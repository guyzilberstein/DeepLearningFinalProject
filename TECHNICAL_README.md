# Campus Image-to-GPS Localization - Technical Documentation

> **Goal**: Predict GPS coordinates from single campus images with sub-10 meter mean error.  
> **Final Result**: **8.17m mean error** (7.33m median) using an ensemble of 3 models.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Configuration](#training-configuration)
5. [Evaluation](#evaluation)
6. [File Reference](#file-reference)

---

## Project Structure

```
DeepLearningFinalProject/
├── data/
│   ├── raw_photos/               # Original HEIC photos (not in git - see data/README.md)
│   ├── processed_images_320/     # Resized 320x320 JPGs for training
│   ├── metadata_raw/             # Per-folder CSV files with EXIF data
│   ├── dataset.csv               # Training pool (2,248 samples)
│   ├── test_dataset.csv          # External test set (1,023 samples)
│   ├── night_holdout.csv         # Night-specific evaluation set (54 samples)
│   ├── corrections_batch*.csv    # Manual GPS corrections
│   └── reference_coords.json     # Reference lat/lon for coordinate conversion
├── src/
│   ├── data_prep/                # Data processing scripts
│   ├── model/                    # Model and dataset definitions
│   ├── training/                 # Training and evaluation
│   └── utils/                    # Visualization and utilities
├── checkpoints/                  # Saved model weights
├── outputs/                      # Visualizations and results
├── predict.py                    # Single-image inference script
└── project_journal.md            # Development log
```

---

## Data Pipeline

### Overview
```
Raw HEIC Photos → Extract Metadata → Convert to JPG → Normalize Coordinates → Train/Test Split
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

**Output**: One CSV per folder in `data/metadata_raw/` with columns:
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
1. Loads all metadata CSVs
2. Calculates reference point (campus center)
3. Converts lat/lon to local (x, y) meters
4. **Automatically applies corrections** from `corrections_batch*.csv`
5. Splits data into train/test/night-holdout

```bash
python src/data_prep/normalize_coords.py
```

**Output Files**:
| File | Purpose |
|------|---------|
| `dataset.csv` | Training pool (~2,200 samples) |
| `test_dataset.csv` | External test set (~1,000 samples) |
| `night_holdout.csv` | 10% of night photos for low-light evaluation |
| `reference_coords.json` | Reference lat/lon for inference |

**Coordinate System**:
- `x_meters`: East-West offset from reference (positive = East)
- `y_meters`: North-South offset from reference (positive = North)

---

### GPS Correction Workflow

For manually correcting inaccurate GPS labels:

1. **Export worst predictions** for review:
   ```bash
   python src/utils/export_worst_for_audit.py --num 25
   ```

2. **Import to Google My Maps** → Drag points to correct locations → Export

3. **Convert back to corrections format**:
   ```bash
   python src/utils/import_corrections.py exported.csv --output corrections_batch5.csv
   ```

4. **Regenerate dataset** (corrections are auto-applied):
   ```bash
   python src/data_prep/normalize_coords.py
   ```

---

## Model Architecture

### Backbone: EfficientNet-B0
**File**: `src/model/network.py`

**Why EfficientNet-B0?**
- 5.3M parameters — optimal for ~2,000 training images
- Pretrained on ImageNet — strong feature extraction out-of-box
- Efficient inference — suitable for mobile deployment

### Regression Head: Custom MLP
```
EfficientNet-B0 (1280 features)
      ↓
Linear(1280 → 512) + ReLU + Dropout(0.3)
      ↓
Linear(512 → 128) + ReLU
      ↓
Linear(128 → 2)  →  [x_meters, y_meters]
```

**Why MLP instead of Linear?**
- Captures non-linear relationships between visual features and GPS
- Dropout (0.3) prevents overfitting
- 2-layer hidden structure balances expressiveness with regularization

---

## Training Configuration

**File**: `src/training/train.py`

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input Size** | 320×320 | Higher resolution captures fine landmarks (56% more pixels than 256) |
| **Batch Size** | 32 (GPU) / 8 (MPS) | Auto-detected based on hardware |
| **Learning Rate** | 0.0001 | Conservative for fine-tuning pretrained backbone |
| **Weight Decay** | 1e-4 | L2 regularization prevents overfitting |
| **Optimizer** | Adam | Adaptive learning rates per parameter |
| **Loss Function** | HuberLoss (δ=1.0) | Robust to GPS label noise |
| **Scheduler** | ReduceLROnPlateau | factor=0.5, patience=7 |
| **Train/Val Split** | 85/15 | Stratified by location |

### Loss Function: HuberLoss

**Why Huber over MSE?**
- GPS labels can be inaccurate (phone GPS error, obstructed signal)
- HuberLoss is **linear for large errors** (outliers don't dominate training)
- **Quadratic for small errors** (fine-tuning near correct values)

```python
# δ=1.0: Transition from quadratic to linear at 1 meter error
criterion = nn.HuberLoss(delta=1.0)
```

### Learning Rate Scheduler

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
```

**Settings**:
- **patience=7**: Wait 7 epochs before reducing LR (prevents premature reduction)
- **factor=0.5**: Halve LR (gentler than 0.1) to allow continued learning

### Data Augmentation

**File**: `src/model/dataset.py`

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| RandomRotation | ±5° | Simulates phone angle variation |
| RandomPerspective | scale=0.2, p=0.5 | Simulates phone tilt (gentle) |
| ColorJitter | B=0.3, C=0.3, S=0.3, H=0.1 | Time-of-day invariance |
| Night Simulation | brightness=0.4-0.7, p=0.2 | Low-light robustness |
| RandomErasing | scale=0.02-0.15, p=0.2 | Occlusion simulation (trees, people) |

**Design Decisions**:
- Perspective kept gentle (0.2 vs 0.3) — extreme distortion changes apparent location
- Night simulation at 40-70% brightness — avoids fully black images
- Hue shift (0.1) enables time-of-day generalization

---

## Evaluation

### Test Set Structure

| Dataset | Size | Purpose |
|---------|------|---------|
| `test_dataset.csv` | 1,023 | Primary evaluation (photos from all locations) |
| `night_holdout.csv` | 54 | Low-light performance evaluation |

### Scripts

**Evaluate single model**:
```bash
python src/training/evaluate.py b0_320_seed42
```

**Evaluate night performance**:
```bash
python src/utils/evaluate_night.py b0_320_seed42
```

**Evaluate ensemble (3 models)**:
```bash
python src/utils/ensemble_evaluate.py b0_320_seed42 b0_320_seed100 b0_320_seed123
```

### Ensemble Strategy

Training 3 models with different random seeds and averaging predictions:

```
Model 1 (seed=42)  ──┐
Model 2 (seed=123) ──┼──→ Average → Final Prediction
Model 3 (seed=456) ──┘
```

**Why Ensemble?**
- Each model has slightly different "blind spots"
- Averaging reduces variance by ~4-5%
- Final result: 8.17m (ensemble) vs 8.54m (best individual)

---

## File Reference

### Data Preparation (`src/data_prep/`)

| File | Purpose | Usage |
|------|---------|-------|
| `extract_metadata.py` | Extract EXIF GPS data from raw photos | `python extract_metadata.py [--folders F1 F2]` |
| `convert_images.py` | Convert HEIC → 320×320 JPG | `python convert_images.py` |
| `normalize_coords.py` | Generate dataset.csv with local coordinates | `python normalize_coords.py` |
| `extract_problematic.py` | Move high-error test photos to training (Active Learning) | `python extract_problematic.py --threshold 20` |

### Model (`src/model/`)

| File | Purpose |
|------|---------|
| `network.py` | CampusLocator model definition (EfficientNet-B0 + MLP head) |
| `dataset.py` | PyTorch Dataset class with augmentation pipeline |

### Training (`src/training/`)

| File | Purpose | Usage |
|------|---------|-------|
| `train.py` | Main training loop | `python train.py` or with args: `--experiment_name b0_v1 --num_epochs 100 --seed 42` |
| `evaluate.py` | Evaluate on test set | `python evaluate.py [experiment_name]` |

### Utilities (`src/utils/`)

| File | Purpose | Usage |
|------|---------|-------|
| `evaluate_night.py` | Evaluate on night holdout | `python evaluate_night.py [experiment_name]` |
| `ensemble_evaluate.py` | Evaluate ensemble of models | `python ensemble_evaluate.py model1 model2 model3` |
| `visualize_worst.py` | Plot 25 worst predictions with maps | `python visualize_worst.py --experiment b0_v3` |
| `visualize_augmentations.py` | Preview data augmentation effects | `python visualize_augmentations.py --images 5 --augs 10` |
| `visualize_results.py` | Plot best/random/worst samples | `python visualize_results.py` |
| `visualize_ensemble.py` | Visualize ensemble predictions | `python visualize_ensemble.py` |
| `visualize_worst_ensemble.py` | Plot worst ensemble predictions | `python visualize_worst_ensemble.py` |
| `export_worst_for_audit.py` | Export worst predictions for GPS correction | `python export_worst_for_audit.py --num 25` |
| `import_corrections.py` | Convert corrected coords to batch format | `python import_corrections.py input.csv` |

### Root Directory

| File | Purpose |
|------|---------|
| `predict.py` | **Inference script** — Predict GPS from a single image using ensemble |
| `project_journal.md` | Development log with all experiments and decisions |
| `README.md` | Project overview |
| `TECHNICAL_README.md` | This file — detailed technical documentation |

---

## Quick Reference: Training a New Model

```bash
# 1. Prepare data (if new photos added)
python src/data_prep/extract_metadata.py --folders NewFolder
python src/data_prep/convert_images.py
python src/data_prep/normalize_coords.py

# 2. Train (local)
python src/training/train.py --experiment_name my_model --num_epochs 100 --seed 42

# 3. Train (cluster with nohup)
nohup python src/training/train.py --experiment_name my_model --num_epochs 200 > training.log 2>&1 &

# 4. Evaluate
python src/training/evaluate.py my_model
python src/utils/evaluate_night.py my_model

# 5. Visualize errors
python src/utils/visualize_worst.py --experiment my_model
```

---

## Final Model Performance

| Metric | Test Set (1,023) | Night Holdout (54) |
|--------|------------------|-------------------|
| Mean Error | **8.17m** | 9.39m |
| Median Error | **7.33m** | 7.32m |
| Under 10m | ~60% | 59.3% |
| Under 20m | ~95% | 90.7% |
| Over 30m | 1 sample | **0 samples** |

*Achieved with 3-model ensemble (seeds 42, 123, 456), 320×320 resolution, 200 epochs.*

