import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phdseg import FFEModule, DecoderBlock
from src.dataset import PolypDataset, PolypTestDataset, get_train_transform
from src.losses import StructureLoss
from src.metrics import evaluate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_DATASETS = {
    'Kvasir': 'data/TestDataset/Kvasir',
    'CVC-ClinicDB': 'data/TestDataset/CVC-ClinicDB',
    'CVC-ColonDB': 'data/TestDataset/CVC-ColonDB',
    'ETIS': 'data/TestDataset/ETIS-LaribPolypDB',
    'CVC-300': 'data/TestDataset/CVC-300',
}


def generate_boundary(mask, kernel_size=3):
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    boundary = dilated - eroded
    return boundary.clamp(0, 1)


class PHDSegAblation(nn.Module):

    def __init__(self, use_ffe=True, use_ds=True, use_bnd=True, pretrained=True):
        super().__init__()
        self.use_ffe = use_ffe
        self.use_ds = use_ds
        self.use_bnd = use_bnd

        import segmentation_models_pytorch as smp
        _tmp = smp.Unet(
            encoder_name='mit_b2',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3, classes=1,
        )
        self.encoder = _tmp.encoder
        del _tmp

        enc_ch = [64, 128, 320, 512]
        dec_ch = [64, 128, 256, 512]

        if use_ffe:
            self.ffe = nn.ModuleList([FFEModule(c) for c in enc_ch])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_ch[3], dec_ch[3], 1, bias=False),
            nn.BatchNorm2d(dec_ch[3]),
            nn.ReLU(inplace=True),
        )

        self.up4_proj = nn.Sequential(
            nn.Conv2d(dec_ch[3], dec_ch[2], 1, bias=False),
            nn.BatchNorm2d(dec_ch[2]),
            nn.ReLU(inplace=True),
        )
        self.dec3 = DecoderBlock(dec_ch[2], enc_ch[2], dec_ch[2])

        self.up3_proj = nn.Sequential(
            nn.Conv2d(dec_ch[2], dec_ch[1], 1, bias=False),
            nn.BatchNorm2d(dec_ch[1]),
            nn.ReLU(inplace=True),
        )
        self.dec2 = DecoderBlock(dec_ch[1], enc_ch[1], dec_ch[1])

        self.up2_proj = nn.Sequential(
            nn.Conv2d(dec_ch[1], dec_ch[0], 1, bias=False),
            nn.BatchNorm2d(dec_ch[0]),
            nn.ReLU(inplace=True),
        )
        self.dec1 = DecoderBlock(dec_ch[0], enc_ch[0], dec_ch[0])

        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_ch[0], dec_ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch[0], 1, 1),
        )

        if use_ds:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(dec_ch[2], 1, 1),
                nn.Conv2d(dec_ch[1], 1, 1),
            ])

        if use_bnd:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(dec_ch[0], dec_ch[0] // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch[0] // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dec_ch[0] // 4, 1, 1),
            )

    def forward(self, x):
        B, _, H, W = x.shape

        enc_feats = self.encoder(x)
        f1, f2, f3, f4 = enc_feats[2], enc_feats[3], enc_feats[4], enc_feats[5]

        if self.use_ffe:
            f1 = self.ffe[0](f1)
            f2 = self.ffe[1](f2)
            f3 = self.ffe[2](f3)
            f4 = self.ffe[3](f4)

        d4 = self.bottleneck(f4)

        d4_proj = self.up4_proj(d4)
        d4_up = F.interpolate(d4_proj, size=f3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(d4_up, f3)

        d3_proj = self.up3_proj(d3)
        d3_up = F.interpolate(d3_proj, size=f2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(d3_up, f2)

        d2_proj = self.up2_proj(d2)
        d2_up = F.interpolate(d2_proj, size=f1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(d2_up, f1)

        seg = self.seg_head(d1)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)

        if self.training:
            ds_preds = []
            if self.use_ds:
                ds_preds.append(F.interpolate(self.ds_heads[0](d3), size=(H, W),
                                              mode='bilinear', align_corners=False))
                ds_preds.append(F.interpolate(self.ds_heads[1](d2), size=(H, W),
                                              mode='bilinear', align_corners=False))

            boundary = None
            if self.use_bnd:
                boundary = self.boundary_head(d1)

            return seg, ds_preds, boundary

        return seg


def train_variant(variant_name, use_ffe, use_ds, use_bnd, epochs=100, patience=20,
                  batch_size=16, lr=1e-4, ds_weight=0.3, bnd_weight=0.2):
    save_dir = PROJECT_ROOT / 'results' / f'ablation_{variant_name}'

    if (save_dir / 'test_results.json').exists():
        print(f"\nSKIPPING {variant_name} (already trained)")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ABLATION: {variant_name} (FFE={use_ffe}, DS={use_ds}, BND={use_bnd})")
    print(f"{'='*60}")

    model = PHDSegAblation(use_ffe=use_ffe, use_ds=use_ds, use_bnd=use_bnd,
                           pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    seg_loss_fn = StructureLoss()
    bnd_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_dataset = PolypDataset(
        image_dir=PROJECT_ROOT / 'data/TrainDataset/image',
        mask_dir=PROJECT_ROOT / 'data/TrainDataset/masks',
        transform=get_train_transform(352),
        img_size=352,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    history = {'train_loss': [], 'val_dice': [], 'lr': []}
    best_dice = 0.0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, masks, _ in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False
        ):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)

            seg_pred, ds_preds, boundary = output

            loss = seg_loss_fn(seg_pred, masks)

            if ds_preds:
                ds_loss = sum(seg_loss_fn(dp, masks) for dp in ds_preds) / len(ds_preds)
                loss += ds_weight * ds_loss

            if boundary is not None:
                boundary_gt = generate_boundary(masks)
                bgt = F.interpolate(boundary_gt, size=boundary.shape[2:],
                                    mode='bilinear', align_corners=False)
                loss += bnd_weight * bnd_loss_fn(boundary, bgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        scheduler.step()

        val_dice = quick_eval(model, 'Kvasir')

        history['train_loss'].append(epoch_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"  -> New best! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time
    history['training_time_seconds'] = training_time
    history['best_val_dice'] = best_dice
    history['total_epochs'] = epoch
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    best_path = save_dir / 'best_model.pth'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    all_results = {}
    for ds_name in TEST_DATASETS:
        print(f"Evaluating on {ds_name}...")
        metrics = eval_dataset(model, ds_name)
        all_results[ds_name] = metrics
        print(f"  Dice: {metrics['mean_dice']:.4f} | IoU: {metrics['mean_iou']:.4f}")

    speed = measure_speed(model)
    all_results['inference_fps'] = speed
    all_results['model_info'] = {
        'name': variant_name,
        'use_ffe': use_ffe,
        'use_ds': use_ds,
        'use_bnd': use_bnd,
        'params': n_params,
    }

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{variant_name} done — Best Dice: {best_dice:.4f}, FPS: {speed:.1f}")

    del model, optimizer
    torch.cuda.empty_cache()


def quick_eval(model, dataset_name):
    test_path = PROJECT_ROOT / TEST_DATASETS[dataset_name]
    dataset = PolypTestDataset(
        image_dir=test_path / 'images',
        mask_dir=test_path / 'masks',
        img_size=352,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for image, mask, name, (orig_h, orig_w) in loader:
            image = image.to(DEVICE)
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.sigmoid(output)
            pred = F.interpolate(pred, size=(orig_h.item(), orig_w.item()),
                                 mode='bilinear', align_corners=False)
            pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float64)
            gt = mask.squeeze().numpy().astype(np.float64)
            predictions.append(pred)
            ground_truths.append(gt)

    metrics = evaluate_dataset(predictions, ground_truths)
    return metrics['mean_dice']


def eval_dataset(model, dataset_name):
    test_path = PROJECT_ROOT / TEST_DATASETS[dataset_name]
    dataset = PolypTestDataset(
        image_dir=test_path / 'images',
        mask_dir=test_path / 'masks',
        img_size=352,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for image, mask, name, (orig_h, orig_w) in loader:
            image = image.to(DEVICE)
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.sigmoid(output)
            pred = F.interpolate(pred, size=(orig_h.item(), orig_w.item()),
                                 mode='bilinear', align_corners=False)
            pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float64)
            gt = mask.squeeze().numpy().astype(np.float64)
            predictions.append(pred)
            ground_truths.append(gt)

    return evaluate_dataset(predictions, ground_truths)


def measure_speed(model, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 3, 352, 352).to(DEVICE)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    torch.cuda.synchronize()
    return n_runs / (time.time() - start)


def main():
    ablation_configs = [
        ('base', False, False, False),
        ('ffe_only', True, False, False),
        ('ds_bnd', False, True, True),
    ]

    for name, use_ffe, use_ds, use_bnd in ablation_configs:
        try:
            train_variant(name, use_ffe, use_ds, use_bnd)
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)

    variants = ['base', 'ffe_only', 'ds_bnd']
    phdseg_results = PROJECT_ROOT / 'results' / 'phdseg' / 'test_results.json'
    if phdseg_results.exists():
        variants.append('phdseg')

    print(f"\n{'Variant':<15} {'FFE':>4} {'DS':>4} {'BND':>4} {'Kvasir':>8} {'Clinic':>8} "
          f"{'Colon':>8} {'ETIS':>8} {'CVC300':>8} {'Params':>10} {'FPS':>6}")
    print("-" * 100)

    for var in variants:
        if var == 'phdseg':
            rfile = PROJECT_ROOT / 'results' / 'phdseg' / 'test_results.json'
        else:
            rfile = PROJECT_ROOT / 'results' / f'ablation_{var}' / 'test_results.json'

        if not rfile.exists():
            continue

        with open(rfile) as f:
            r = json.load(f)

        info = r.get('model_info', {})
        ffe = 'Y' if info.get('use_ffe', var in ['ffe_only', 'phdseg']) else 'N'
        ds = 'Y' if info.get('use_ds', var in ['ds_bnd', 'phdseg']) else 'N'
        bnd = 'Y' if info.get('use_bnd', var in ['ds_bnd', 'phdseg']) else 'N'
        params = info.get('params', 'N/A')
        fps = r.get('inference_fps', 0)

        row = f"{var:<15} {ffe:>4} {ds:>4} {bnd:>4}"
        for dataset in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS', 'CVC-300']:
            if dataset in r:
                row += f" {r[dataset]['mean_dice']:>8.4f}"
            else:
                row += f" {'N/A':>8}"
        row += f" {params:>10} {fps:>6.1f}"
        print(row)


if __name__ == '__main__':
    main()
