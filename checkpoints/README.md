# Model Checkpoints

## Quick Start

The ConvNeXt-Tiny models (the best performing ensemble) must be downloaded separately as they are too large for the repository.

## Download ConvNeXt Checkpoints

The ConvNeXt-Tiny models (~324MB each) are hosted on Google Drive:

**Download link:** [Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1P3ePrcjLZNyNx85o62SDG4hXa4oDz908?usp=sharing)

Download and place in this folder:
- `best_convnext_tiny_v1.pth`
- `best_convnext_tiny_v2.pth`
- `best_convnext_tiny_v3.pth`

## Recommended Usage

**Best Single Model:** `best_convnext_tiny_v3.pth`

**Best Ensemble:**
```bash
python src/utils/evaluation/ensemble_evaluate.py convnext_tiny_v1 convnext_tiny_v2 convnext_tiny_v3
```

