# Model Checkpoints

## Quick Start

Small checkpoints (EfficientNet-B0) are included in the repository. 
Large checkpoints (ConvNeXt) must be downloaded separately.

## Download ConvNeXt Checkpoints

The ConvNeXt-Tiny models (~324MB each) are hosted on Google Drive:

**Download link:** [Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1P3ePrcjLZNyNx85o62SDG4hXa4oDz908?usp=sharing)

Download and place in this folder:
- `best_convnext_tiny_v1.pth`
- `best_convnext_tiny_v2.pth`
- `best_convnext_tiny_v3.pth`

## Available Models

| Model | Architecture | Size | In Repo | Mean Error |
|-------|--------------|------|---------|------------|
| `best_convnext_tiny_v1.pth` | ConvNeXt-Tiny | 324MB | ❌ Google Drive | 7.50m |
| `best_convnext_tiny_v2.pth` | ConvNeXt-Tiny | 324MB | ❌ Google Drive | 7.55m |
| `best_convnext_tiny_v3.pth` | ConvNeXt-Tiny | 324MB | ❌ Google Drive | 7.46m |
| `best_b0_320_seed42.pth` | EfficientNet-B0 | ~21MB | ✅ Yes | 8.54m |
| `best_b0_320_seed123.pth` | EfficientNet-B0 | ~21MB | ✅ Yes | 8.67m |
| `best_b0_320_seed456.pth` | EfficientNet-B0 | ~21MB | ✅ Yes | 8.71m |

## Recommended Usage

**Best Single Model:** `best_convnext_tiny_v3.pth` (7.46m mean error)

**Best Ensemble (7.16m):**
```bash
python src/utils/ensemble_evaluate.py convnext_tiny_v1 convnext_tiny_v2 convnext_tiny_v3
```

**Hybrid Ensemble (7.13m):**
```bash
python src/utils/ensemble_evaluate.py convnext_tiny_v1 convnext_tiny_v2 b0_320_seed42
```

