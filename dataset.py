import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PolypDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, img_size=352):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform

        self.images = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_candidates = list(self.mask_dir.glob(f"{img_path.stem}.*"))
        if not mask_candidates:
            raise FileNotFoundError(f"No mask found for {img_path.name}")
        mask_path = mask_candidates[0]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0

        mask = (mask > 0.5).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask, img_path.stem


class PolypTestDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_size=352):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size

        self.images = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_candidates = list(self.mask_dir.glob(f"{img_path.stem}.*"))
        if not mask_candidates:
            raise FileNotFoundError(f"No mask found for {img_path.name}")
        mask_path = mask_candidates[0]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        orig_h, orig_w = image.shape[:2]

        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = torch.from_numpy(
            image_resized.transpose(2, 0, 1)
        ).float() / 255.0

        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor, img_path.stem, (orig_h, orig_w)


def get_train_transform(img_size=352):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(img_size=352):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
