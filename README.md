# PHDSeg: Rethinking Decoder Design for Transformer-Based Polyp Segmentation

A polyp segmentation architecture that pairs a Mix Transformer (MiT-B2) encoder with a custom multi-scale decoder and an optional learnable Fourier Frequency Enhancement (FFE) module.

<!-- TODO(sayyod): add paper link once published -->

## Overview

Accurate polyp segmentation in colonoscopy images remains challenging due to variable polyp size, shape, and boundary ambiguity. PHDSeg addresses this with:

1. **MiT-B2 encoder** (via segmentation-models-pytorch) pretrained on ImageNet for hierarchical multi-scale feature extraction.
2. **FFE module** -- a learnable high-pass filter in the Fourier domain that amplifies edge and texture information at each encoder stage, controlled by a per-channel scaling parameter.
3. **Custom decoder** -- a sequence of decoder blocks that fuse upsampled decoder features with skip connections through 1x1 projection and 3x3 convolution pairs, progressively recovering spatial resolution.

## Architecture

```
Input (3 x 352 x 352)
        |
   MiT-B2 Encoder
        |
   f1 (64)   f2 (128)   f3 (320)   f4 (512)
    |           |           |           |
   [FFE]      [FFE]       [FFE]       [FFE]       (optional)
    |           |           |           |
    |           |           |       Bottleneck (512)
    |           |           |           |
    |           |        DecoderBlock3 (256)
    |           |           |
    |        DecoderBlock2 (128)
    |           |
        DecoderBlock1 (64)
            |
       Seg Head --> output (1 x 352 x 352)
```

Each `DecoderBlock` takes an upsampled decoder tensor and a skip connection, applies a 1x1 skip convolution followed by concatenation and two 3x3 convolutions with BatchNorm and ReLU.

## Datasets

The model is trained on a combined training set and evaluated on five standard polyp segmentation benchmarks:

| Dataset | Purpose |
|---|---|
| Kvasir-SEG | Validation during training; test benchmark |
| CVC-ClinicDB | Test benchmark |
| CVC-ColonDB | Test benchmark |
| ETIS-LaribPolypDB | Test benchmark |
| CVC-300 | Test benchmark |

### Expected data layout

```
data/
  TrainDataset/
    image/
    masks/
  TestDataset/
    Kvasir/
      images/
      masks/
    CVC-ClinicDB/
      images/
      masks/
    CVC-ColonDB/
      images/
      masks/
    ETIS-LaribPolypDB/
      images/
      masks/
    CVC-300/
      images/
      masks/
```

## Setup

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for the full list.

### Hardware

Training uses CUDA by default. The code falls back to CPU automatically.

## Training

### Standard training (PHDSeg without FFE)

```bash
python train_phdseg.py
```

### With FFE enabled

Edit `train_phdseg.py` and set `use_ffe=True` in the `PHDSegTrainer` constructor, then run:

```bash
python train_phdseg.py
```

### Seed robustness study

```bash
python train_seed.py --seeds 42 84 123 256 512
```

### Ablation study

```bash
python train_ablation.py
```

### Training configuration

| Parameter | Default |
|---|---|
| Image size | 352 x 352 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight decay 1e-4) |
| Scheduler | Cosine annealing (eta_min 1e-6) |
| Epochs | 100 |
| Early stopping patience | 20 |
| Gradient clipping | max norm 1.0 |

### Data augmentation (training)

HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter, GaussianBlur, ImageNet normalization (via albumentations).

## Evaluation

After training, the best checkpoint is automatically evaluated on all five test datasets. Metrics reported per dataset:

| Metric | Description |
|---|---|
| Dice | Dice similarity coefficient |
| IoU | Intersection over union |
| MAE | Mean absolute error |
| S-measure | Structural similarity |
| E-measure | Enhanced alignment measure |
| Weighted F-measure | Distance-weighted F-score |

### Results

<!-- TODO(sayyod): fill in actual results from your best run -->

| Dataset | Dice | IoU | MAE | S-m | E-m | wF-m |
|---|---|---|---|---|---|---|
| Kvasir | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| CVC-ClinicDB | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| CVC-ColonDB | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| ETIS | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| CVC-300 | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

## Loss function

**StructureLoss** -- a combination of weighted binary cross-entropy and weighted IoU loss, where weights are derived from boundary proximity (5x emphasis near polyp edges via average pooling).

## Metrics implementation

All metrics (Dice, IoU, MAE, S-measure, E-measure, weighted F-measure) are implemented from scratch in `metrics.py` following their original paper definitions. The weighted F-measure uses `scipy.ndimage.distance_transform_edt`.

## Project structure

```
phdseg.py             -- model architecture (PHDSeg, FFEModule, DecoderBlock)
dataset.py            -- PolypDataset, PolypTestDataset, augmentation transforms
losses.py             -- DiceLoss, BCEDiceLoss, StructureLoss
metrics.py            -- all evaluation metrics
train_phdseg.py       -- main training script with early stopping and full evaluation
train_ablation.py     -- ablation study variants
train_seed.py         -- multi-seed robustness evaluation
```

## Citation

<!-- TODO(sayyod): add BibTeX citation once the paper is published -->

## License

<!-- TODO(sayyod): add license -->
