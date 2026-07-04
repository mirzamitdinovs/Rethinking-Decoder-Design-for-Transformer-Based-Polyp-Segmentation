import json
import time
import random
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phdseg import PHDSeg
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quick_eval(model, dataset_name, img_size=352):
    test_path = PROJECT_ROOT / TEST_DATASETS[dataset_name]
    dataset = PolypTestDataset(
        image_dir=test_path / 'images',
        mask_dir=test_path / 'masks',
        img_size=img_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for image, mask, name, (orig_h, orig_w) in loader:
            image = image.to(DEVICE)
            pred = torch.sigmoid(model(image))
            pred = F.interpolate(pred, size=(orig_h.item(), orig_w.item()),
                                 mode='bilinear', align_corners=False)
            pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float64)
            gt = mask.squeeze().numpy().astype(np.float64)
            predictions.append(pred)
            ground_truths.append(gt)
    return evaluate_dataset(predictions, ground_truths)


def measure_speed(model, img_size=352, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
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


def train_with_seed(seed):
    set_seed(seed)
    save_dir = PROJECT_ROOT / 'results' / f'phdseg_seed{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)

    model = PHDSeg(pretrained=True, use_ffe=False).to(DEVICE)
    seg_loss = StructureLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    train_dataset = PolypDataset(
        image_dir=PROJECT_ROOT / 'data/TrainDataset/image',
        mask_dir=PROJECT_ROOT / 'data/TrainDataset/masks',
        transform=get_train_transform(352),
        img_size=352,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    history = {'train_loss': [], 'val_dice': [], 'lr': []}
    best_dice = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training PHDSeg | Seed: {seed}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        for images, masks, _ in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch}", leave=False):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            pred = model(images)
            loss = seg_loss(pred, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        scheduler.step()

        val_dice = quick_eval(model, 'Kvasir')['mean_dice']
        history['train_loss'].append(epoch_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Seed {seed} | Epoch {epoch} | Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"  -> New best! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time
    history['training_time_seconds'] = training_time
    history['best_val_dice'] = best_dice
    history['total_epochs'] = epoch
    history['seed'] = seed
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(save_dir / 'best_model.pth', map_location=DEVICE))
    model.eval()

    all_results = {}
    for ds_name in TEST_DATASETS:
        print(f"Evaluating on {ds_name}...")
        metrics = quick_eval(model, ds_name)
        all_results[ds_name] = metrics
        print(f"  Dice: {metrics['mean_dice']:.4f} | IoU: {metrics['mean_iou']:.4f}")

    fps = measure_speed(model)
    all_results['inference_fps'] = fps
    all_results['model_info'] = {
        'name': 'PHDSeg',
        'seed': seed,
        'params': sum(p.numel() for p in model.parameters()),
    }

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSeed {seed} done — Best Val Dice: {best_dice:.4f}, FPS: {fps:.1f}")
    print(f"Training time: {training_time/60:.1f} minutes")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    train_with_seed(args.seed)
