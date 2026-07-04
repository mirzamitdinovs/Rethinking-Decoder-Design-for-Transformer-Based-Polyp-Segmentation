import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
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


class PHDSegTrainer:

    def __init__(self, img_size=352, batch_size=16, lr=1e-4,
                 epochs=100, patience=20, use_ffe=False):
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

        self.save_dir = PROJECT_ROOT / 'results' / 'phdseg'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = PHDSeg(pretrained=True, use_ffe=use_ffe).to(DEVICE)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'PHDSeg (FFE={use_ffe}): {n_params:,} parameters')

        self.seg_loss = StructureLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

    def _get_train_loader(self):
        dataset = PolypDataset(
            image_dir=PROJECT_ROOT / 'data/TrainDataset/image',
            mask_dir=PROJECT_ROOT / 'data/TrainDataset/masks',
            transform=get_train_transform(self.img_size),
            img_size=self.img_size,
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )

    def train(self):
        train_loader = self._get_train_loader()
        history = {'train_loss': [], 'val_dice': [], 'lr': []}
        best_dice = 0.0
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training PHDSeg | Epochs: {self.epochs}, BS: {self.batch_size}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for images, masks, _ in tqdm(
                train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False
            ):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                self.optimizer.zero_grad()
                seg_pred = self.model(images)
                loss = self.seg_loss(seg_pred, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            self.scheduler.step()

            val_dice = self._quick_eval('Kvasir')['mean_dice']
            history['train_loss'].append(epoch_loss)
            history['val_dice'].append(val_dice)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            print(f"Epoch {epoch}/{self.epochs} | Loss: {epoch_loss:.4f} | "
                  f"Val Dice: {val_dice:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                print(f"  -> New best! Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break

        training_time = time.time() - start_time
        history['training_time_seconds'] = training_time
        history['best_val_dice'] = best_dice
        history['total_epochs'] = epoch
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete in {training_time/60:.1f} minutes")
        print(f"Best validation Dice: {best_dice:.4f}")

    def _quick_eval(self, dataset_name):
        test_path = PROJECT_ROOT / TEST_DATASETS[dataset_name]
        dataset = PolypTestDataset(
            image_dir=test_path / 'images',
            mask_dir=test_path / 'masks',
            img_size=self.img_size,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        self.model.eval()
        predictions, ground_truths = [], []
        with torch.no_grad():
            for image, mask, name, (orig_h, orig_w) in loader:
                image = image.to(DEVICE)
                pred = torch.sigmoid(self.model(image))
                pred = F.interpolate(pred, size=(orig_h.item(), orig_w.item()),
                                     mode='bilinear', align_corners=False)
                pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float64)
                gt = mask.squeeze().numpy().astype(np.float64)
                predictions.append(pred)
                ground_truths.append(gt)
        return evaluate_dataset(predictions, ground_truths)

    def evaluate_all(self):
        best_path = self.save_dir / 'best_model.pth'
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        self.model.eval()

        all_results = {}
        for name in TEST_DATASETS:
            print(f"Evaluating on {name}...")
            metrics = self._quick_eval(name)
            all_results[name] = metrics
            print(f"  Dice: {metrics['mean_dice']:.4f} | IoU: {metrics['mean_iou']:.4f} | MAE: {metrics['mean_mae']:.4f}")

        speed = self._measure_speed()
        all_results['inference_fps'] = speed
        all_results['model_info'] = {
            'name': 'PHDSeg',
            'encoder': 'mit_b2',
            'params': sum(p.numel() for p in self.model.parameters()),
        }

        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nInference speed: {speed:.1f} FPS")
        return all_results

    def _measure_speed(self, n_runs=100):
        self.model.eval()
        dummy = torch.randn(1, 3, self.img_size, self.img_size).to(DEVICE)
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                self.model(dummy)
        torch.cuda.synchronize()
        return n_runs / (time.time() - start)


def main():
    trainer = PHDSegTrainer(
        img_size=352, batch_size=16, lr=1e-4,
        epochs=100, patience=20, use_ffe=False,
    )
    trainer.train()
    results = trainer.evaluate_all()

    print("\n" + "=" * 60)
    print("PHDSeg RESULTS")
    print("=" * 60)
    for ds in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS', 'CVC-300']:
        r = results[ds]
        print(f"  {ds:<15} Dice: {r['mean_dice']:.4f}  IoU: {r['mean_iou']:.4f}  MAE: {r['mean_mae']:.4f}")


if __name__ == '__main__':
    main()
