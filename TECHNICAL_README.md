# Campus Image-to-GPS Localization - Technical Documentation

> **Goal**: Predict GPS coordinates from single campus images with sub-10 meter mean error.  
> **Final Result**: **7.16m mean error** (6.48m median) using a ConvNeXt-Tiny ensemble.

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
│   ├── raw_photos/               # Original HEIC photos (view rawPhotos.md)
│   ├── processed_images_320/     # Resized 320x320 JPGs for training
│   ├── metadata_raw/             # Per-folder CSV files with EXIF data
│   ├── dataset.csv               # Training pool (~2,200 samples)
│   ├── test_dataset.csv          # External test set (1,023 samples)
│   ├── night_holdout.csv         # Night-specific evaluation set (54 samples)
│   ├── corrections_batch*.csv    # Manual GPS corrections (Training set)
│   ├── test_corrections_*.csv    # Manual GPS corrections (Test set)
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
   python src/utils/import_corrections.py exported.csv --output data/corrections_batch5.csv
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
| `night_holdout.csv` | 54 | Low-light performance evaluation |

### Scripts

**Evaluate single model**:
```bash
python src/training/evaluate.py convnext_tiny_v1
```

**Evaluate ensemble (3 models)**:
```bash
# Automatically loads v1, v2, v3
python src/utils/ensemble_evaluate.py convnext_tiny_v1 convnext_tiny_v2 convnext_tiny_v3
```

**Visualize Worst Predictions**:
```bash
python src/utils/visualize_worst.py --experiment convnext_tiny_v1
```

### Ensemble Strategy

Training 3 models with different random seeds and averaging predictions:

```
Model 1 (seed=42)  ──┐
Model 2 (seed=123) ──┼──→ Mean(x, y) → Final Prediction
Model 3 (seed=456) ──┘
```

**Performance Gain:**
- Best Single Model: **7.46m**
- Ensemble: **7.16m** (4% improvement)

---

## Final Model Performance

| Metric | Test Set (1,023) |
|--------|------------------|
| **Mean Error** | **7.16m** |
| **Median Error** | **6.48m** |
| **Under 10m** | **76%** |
| **Under 20m** | **99%** |
| **Catastrophic (>30m)** | **0 samples** |

*Achieved with ConvNeXt-Tiny Ensemble (v1, v2, v3).*

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
| `efficientnet.py` | Legacy Model (EfficientNet-B0) - used for loading old checkpoints |
| `dataset.py` | PyTorch Dataset class with augmentation pipeline |

### Training (`src/training/`)

| File | Purpose |
|------|---------|
| `train.py` | Main training loop |
| `evaluate.py` | Evaluate a single model on test set |

### Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `ensemble_evaluate.py` | Evaluate ensemble (supports mixed architectures) |
| `visualize_worst.py` | Plot 25 worst predictions with maps |
| `export_worst_for_audit.py` | Export worst predictions for GPS correction |
| `analyze_hardest_samples.py` | Find intersection of failures across models |
| `import_corrections.py` | Convert corrected coords to batch format |

