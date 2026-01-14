# Repository Guide & Technical Documentation

> **Goal**: Explain the codebase structure, technical implementation, and file purposes.
> **Note**: For setup and quick start, see the main `README.md`.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Configuration](#training-configuration)
5. [Evaluation](#evaluation)
6. [Visualizations](#visualizations)
7. [File Reference](#file-reference)

---

## Project Structure

This section explains the purpose of each folder and key file in the repository.

```
DeepLearningFinalProject/
├── data/
│   ├── images/                   # Resized 320x320 JPGs (3,327 images)
│   ├── gt.csv                    # Ground Truth for evaluation (Required by API)
│   ├── dataset.csv               # Training pool metadata (~2,200 samples)
│   ├── test_dataset.csv          # External test set metadata (~1,023 samples)
│   └── metadata/
│       ├── raw/                  # Per-folder raw EXIF data (13 CSVs)
│       ├── corrections/          # Manual GPS correction batches (5 CSVs)
│       ├── night_holdout.csv     # Night evaluation set (54 samples)
│       └── reference_coords.json # Lat/Lon to Meter conversion reference
├── src/
│   ├── data_prep/                # Scripts for processing raw data
│   ├── model/                    # Neural network architecture & dataset classes
│   ├── training/                 # Training loops and evaluation
│   └── utils/
│       ├── visualization/        # 8 visualization scripts
│       ├── evaluation/           # Ensemble and night evaluation
│       └── data/                 # Data processing and correction
├── checkpoints/                  # Trained model weights (3 ConvNeXt-Tiny models)
│   ├── best_convnext_tiny_v1.pth
│   ├── best_convnext_tiny_v2.pth
│   └── best_convnext_tiny_v3.pth
├── predict.py                    # Main Inference API (Required submission file)
├── requirements.txt              # Python dependencies
└── project_journal.md            # Development log
```

---

## Data Pipeline

### Overview
```
Raw HEIC Photos → Extract Metadata → Convert to JPG → Normalize Coordinates → Train/Val Split
```

### Step 1: Extract Metadata
**Script**: `src/data_prep/extract_metadata.py`

Extracts GPS coordinates, accuracy, and datetime from photo EXIF data.

```bash
# Process all folders
python src/data_prep/extract_metadata.py

# Process only specific new folders (incremental)
python src/data_prep/extract_metadata.py --folders NewFolder1 NewFolder2
```

**Output**: One CSV per folder in `data/metadata/raw/` with columns:
- `filename`, `path`, `datetime`, `lat`, `lon`, `gps_accuracy_m`

---

### Step 2: Convert Images
**Script**: `src/data_prep/convert_images.py`

Converts HEIC to 320x320 JPG. Flattens directory structure (e.g., `LibraryArea/IMG_001.HEIC` → `LibraryArea_IMG_001.jpg`).

```bash
python src/data_prep/convert_images.py
```

**Key Setting**:
- `TARGET_SIZE = 320` — Higher resolution for fine landmark details

**Behavior**: Skips already-processed images (incremental processing).

---

### Step 3: Normalize Coordinates
**Script**: `src/data_prep/normalize_coords.py`

Core preprocessing script that:
1. Loads all metadata CSVs from `data/metadata/raw/`
2. Calculates reference point (campus center)
3. Converts lat/lon to local (x, y) meters
4. **Automatically applies corrections** from `data/metadata/corrections/`
5. Splits data into train/test/night-holdout

```bash
python src/data_prep/normalize_coords.py
```

**Output Files**:
| File | Purpose |
|------|---------|
| `dataset.csv` | Training pool (~2,200 samples) |
| `test_dataset.csv` | External test set (~1,023 samples) |
| `metadata/night_holdout.csv` | 54 night photos for low-light evaluation |
| `metadata/reference_coords.json` | Reference lat/lon for inference |

**Coordinate System**:
- `x_meters`: East-West offset from reference (positive = East)
- `y_meters`: North-South offset from reference (positive = North)

---

### GPS Correction Workflow

For manually correcting inaccurate GPS labels (used for worst predictions with potentially incorrect labels):

1. **Export worst predictions** for review:
   ```bash
   python src/utils/data/export_worst_for_correction.py --num 25
   ```

2. **Import to Google My Maps** → Drag points to correct locations → Export

3. **Convert back to corrections format**:
   ```bash
   python src/utils/data/import_corrections.py exported.csv --output data/metadata/corrections/corrections_batch5.csv
   ```

4. **Regenerate dataset** (corrections are auto-applied):
   ```bash
   python src/data_prep/normalize_coords.py
   ```

---

## Model Architecture

### Backbone: ConvNeXt-Tiny
**File**: `src/model/network.py`

**Why ConvNeXt-Tiny?**
- **Modern CNN (2022):** Designed to compete with Transformers.
- **Large Kernel (7x7):** Captures larger spatial context than EfficientNet's 3x3 kernels.
- **Performance:** 28M parameters. Outperformed EfficientNet-B0 (5.3M) significantly on this task, especially for **night images** (mean error 7.43m vs 9.39m).

### Regression Head: Custom MLP
```
ConvNeXt-Tiny (768 features)
      ↓
Linear(768 → 512) + GELU + Dropout(0.3)
      ↓
Linear(512 → 128) + GELU
      ↓
Linear(128 → 2)  →  [x_meters, y_meters]
```

**Features:**
- Uses **GELU** activation (smoother than ReLU).
- Preserves ConvNeXt's native **LayerNorm** and **Flatten** layers before the head.

---

## Training Configuration

**File**: `src/training/train.py`

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input Size** | 320×320 | Optimal balance of detail and receptive field |
| **Batch Size** | 24 (GPU) | Tuned for GTX 1080 Ti memory limits |
| **Optimizer** | AdamW | Better regularization/weight decay handling than Adam |
| **Learning Rate** | 1e-4 | Standard fine-tuning rate |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Loss Function** | HuberLoss (δ=1.0) | Robust to GPS label noise (linear for >1m error) |
| **Scheduler** | ReduceLROnPlateau | factor=0.5, patience=7 |
| **Epochs** | 100-125 | Models typically converge around epoch 90-100 |

### Data Augmentation

**File**: `src/model/dataset.py`

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| RandomRotation | ±5° | Simulates phone angle variation |
| RandomPerspective | scale=0.2, p=0.5 | Simulates phone tilt (gentle) |
| ColorJitter | B=0.3, C=0.3, S=0.3, H=0.1 | Time-of-day invariance |
| Night Simulation | brightness=0.4-0.7, p=0.2 | Low-light robustness |
| RandomErasing | scale=0.02-0.15, p=0.2 | Occlusion simulation (trees, people) |

---

## Evaluation

### Test Set Structure

| Dataset | Size | Purpose |
|---------|------|---------|
| `test_dataset.csv` | 1,023 | Primary evaluation (photos from all locations) |
| `metadata/night_holdout.csv` | 54 | Low-light performance evaluation |

### Scripts

**Evaluate single model**:
```bash
python src/training/evaluate.py convnext_tiny_v1
```

**Evaluate ensemble (3 models)**:
```bash
python src/utils/evaluation/ensemble_evaluate.py convnext_tiny_v1 convnext_tiny_v2 convnext_tiny_v3
```

**Evaluate night performance**:
```bash
python src/utils/evaluation/evaluate_night.py
```

### Ensemble Strategy

Training 3 models with different random seeds and averaging predictions:

```
Model 1 (seed=42)  ──┐
Model 2 (seed=123) ──┼──→ Mean(x, y) → Final Prediction
Model 3 (seed=456) ──┘
```

---

## Visualizations

Visualization scripts generate figures to `outputs/` (gitignored). Run any script to regenerate visualizations locally.

### Scripts (`src/utils/visualization/`)

| Script | Description |
|--------|-------------|
| `visualize_error_distribution.py` | Histogram of prediction errors with percentile markers |
| `visualize_model_comparison.py` | Compare single models vs ensemble performance |
| `visualize_day_with_maps.py` | Day predictions overlaid on campus map |
| `visualize_night_with_maps.py` | Night predictions overlaid on campus map |
| `visualize_ensemble.py` | Ensemble vs single model comparison |
| `visualize_worst.py` | Worst predictions grid for a single model |
| `visualize_worst_ensemble.py` | Worst predictions for the ensemble |
| `visualize_augmentations.py` | Preview data augmentation pipeline |

---

## File Reference

### Data Preparation (`src/data_prep/`)

| File | Purpose |
|------|---------|
| `extract_metadata.py` | Extract EXIF GPS data from raw photos |
| `convert_images.py` | Convert HEIC → 320×320 JPG |
| `normalize_coords.py` | Generate dataset.csv with local coordinates |
| `extract_problematic.py` | Move high-error test photos to training (Active Learning) |

### Model (`src/model/`)

| File | Purpose |
|------|---------|
| `network.py` | **Main Model** (ConvNeXt-Tiny + MLP head) |
| `dataset.py` | PyTorch Dataset class with augmentation pipeline |

### Training (`src/training/`)

| File | Purpose |
|------|---------|
| `train.py` | Main training loop |
| `evaluate.py` | Evaluate a single model on test set |

### Utilities (`src/utils/`)

#### Visualization (`src/utils/visualization/`)
| File | Purpose |
|------|---------|
| `visualize_error_distribution.py` | Error histogram with statistics |
| `visualize_model_comparison.py` | Compare model performance |
| `visualize_day_with_maps.py` | Day predictions on campus map |
| `visualize_night_with_maps.py` | Night predictions on campus map |
| `visualize_ensemble.py` | Ensemble vs single model analysis |
| `visualize_worst.py` | Plot worst predictions for single model |
| `visualize_worst_ensemble.py` | Plot worst predictions for ensemble |
| `visualize_augmentations.py` | Preview data augmentations |

#### Evaluation (`src/utils/evaluation/`)
| File | Purpose |
|------|---------|
| `ensemble_evaluate.py` | Evaluate ensemble performance |
| `evaluate_night.py` | Evaluate on night holdout set |

#### Data (`src/utils/data/`)
| File | Purpose |
|------|---------|
| `export_worst_for_correction.py` | Export worst predictions for GPS correction |
| `analyze_hardest_samples.py` | Find intersection of failures across models |
| `import_corrections.py` | Convert corrected coords to batch format |
| `generate_gt_csv.py` | Generate ground truth CSV for submission |
