# Project Journal: Campus GPS Regression
**Date:** Dec 26, 2025

## What We Built Today
We built a Deep Learning pipeline that looks at a photo of campus and predicts its exact real-world location (Latitude/Longitude).
*   **Input:** ~350 photos of Building 35.
*   **Output:** GPS coordinates (meters from a center point).
*   **Model:** ResNet-18 (Transfer Learning).

---

## The Pipeline

### 1. Data Processing
*   **Image Standardization:** We convert all raw HEIC photos to standard JPGs and resize them to 256x256 pixels for efficiency.
*   **Coordinate Normalization:** Since neural networks struggle with large numbers like `31.26201`, we calculate a "Center of the World" (Average Lat/Lon) and convert all GPS tags into relative **(X, Y) meters** from that center.

### 2. Dataset Strategy (70/15/15 Split)
We strictly split our 350 images into three groups to ensure valid testing:
*   **Training (70%):** The data the model learns from.
*   **Validation (15%):** Used to tune the model and save the best version.
*   **Test (15%):** Locked away in a "vault" and used *only* for the final performance check.

### 3. The Model & Training
*   **Architecture:** We use a pre-trained **ResNet-18** backbone (which already knows how to "see" features) and replaced the final layer to output 2 coordinate numbers instead of classification classes.
*   **Handling Noisy Data:** We experimented with different ways to handle GPS inaccuracy.
    *   *Manual Filtering:* Initially, we considered manually removing or adjusting "bad" data points.
    *   *Weighted Loss:* We moved to a more robust **Inverse Variance Weighting** strategy. We use the phone's **GPS Accuracy** metadata to dynamically weight the loss. We punish the model heavily for missing precise photos, but are lenient if the original GPS data was fuzzy (e.g., 20m accuracy).
*   **Epoch Tuning:** We experimented with training duration.
    *   **20 Epochs:** Good baseline.
    *   **30 Epochs:** Optimal performance (best validation loss).
    *   **40 Epochs:** Performance degraded (overfitting started), returning to or worsening beyond the 20-epoch baseline.

### 4. Evolution of Validation Strategy
*   **Initial Approach:** We initially used the Weighted Loss for validation as well. While this showed us how well the model fit the "trusted" data, it was not a realistic test of real-world performance.
*   **Refinement:** We switched the Validation phase to use **Standard (Unweighted) MSE**. In the real world (inference time), the model won't have GPS metadata to rely on—it must perform well on any image, regardless of how accurate the ground truth was during collection. This gives us a much more honest estimate of the error in meters.

---

## Future Improvements & Ideas
*   **Data Augmentation:** Introduce random rotations, brightness changes, and crops to make the model robust to different lighting conditions and angles.
*   **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and number of epochs to find the sweet spot for convergence.
*   **Spatial Transformer Networks (STN):** Add a learnable STN layer that can automatically rotate or warp the input image to a canonical "front-facing" view before it hits the ResNet.
*   **Deeper Backbones:** Try ResNet-50 or EfficientNet to see if a larger model can capture more subtle visual details.

---

## Final Results
On completely unseen photos (the Test Set), the model predicts the location with an average error of **~7.5 meters**. Not bad for a first try on a small dataset!

---

**Date:** Dec 27, 2025

## Expanded Dataset & Blind Test
We significantly expanded our dataset by adding photos from multiple new areas of campus (`LibraryArea`, `UnderBuilding26`, `Building28Area`, `Bulding32Area`, `nightWithout26AndLibrary`, etc.). The dataset grew from ~350 images to **1,872 images**.

### Updates
1.  **Data Pipeline:** Updated `convert_images.py` and `normalize_coords.py` to handle nested folder structures and potential filename collisions by namespacing files with their parent folder names.
2.  **Quality Control:** Added automatic filtering to drop samples with missing GPS coordinates (1 sample dropped).
3.  **Blind Testing:** We maintained the **70/15/15 split** strategy. The split was performed randomly across the entire pooled dataset, ensuring the model was tested on images from various sections without any explicit knowledge of the section labels.

### Results
*   **Run 1 (25 Epochs):** Test Mean Error: **29.25 meters** | Test Median Error: **23.05 meters**
*   **Run 2 (50 Epochs):** Test Mean Error: **26.39 meters** | Test Median Error: **22.39 meters**
*   **Run 3 (Day Only, 25 Epochs):** Test Mean Error: **27.34 meters** | Test Median Error: **22.58 meters**
*   **Run 4 (Area-Specific Z-Score Weighting, 25 Epochs):** Test Mean Error: **32.23 meters** | Test Median Error: **28.09 meters**

### Experiment: Area-Specific Z-Score Weighting
We hypothesized that since different areas of campus have different baseline GPS variances, we should penalize outliers *relative to their specific area* rather than using a global standard.
*   **Implementation:** We calculated the Mean and Standard Deviation of the GPS Accuracy for *each* folder (area). We then calculated a Z-Score for every image: $Z = (\text{Accuracy} - \mu_{area}) / \sigma_{area}$.
*   **Weighting:** We introduced a penalty factor $R = 1 / (1 + \max(0, Z))$, reducing the influence of local outliers.
*   **Outcome:** The performance **decreased** (Mean Error rose from ~29m to ~32m). This suggests that the standard weighting (trusting good absolute GPS values regardless of area) is more effective. Penalizing "relative" outliers might have suppressed useful training signals in difficult areas where high-variance GPS is the best we have.

### Visualization Tool
We developed a visualization script (`src/utils/visualize_results.py`) to qualitatively analyze model performance.
*   **Features:** It generates a grid of test samples showing:
    *   The test image.
    *   The GPS Error in meters (color-coded).
    *   A mini-map (OpenStreetMap) showing the True (Green) and Predicted (Red) locations.
*   **Sampling Strategy:** To get a balanced view, the tool selects:
    *   3 Best Predictions (Lowest Error).
    *   3 Random Predictions.
    *   6 Worst Predictions (Highest Error).
*   **Purpose:** This allows us to visually inspect if specific conditions (e.g., night time, specific buildings) correlate with high errors.

## Model Refinement & Breakthrough
We analyzed the failure modes and made two major changes to the training strategy:

### 1. Stratified Data Split (85/10/5)
We realized that a random 70/15/15 split might exclude entire small locations from the training set or test set. We switched to **Stratified Sampling** based on location (folder name) and increased the training data:
*   **Train (85%):** Maximize data exposure.
*   **Val (10%):** Sufficient for model selection.
*   **Test (5%):** ~94 images, ensuring every building/area is represented proportionally.

### 2. Dropping Weighted Loss
We analyzed the `gps_accuracy_m` metadata and found a huge disparity (2m vs 15m). The Inverse Variance Weighting ($1/\sigma^2$) was scaling weights by a factor of **~56x**. This meant "low accuracy" images (which were often still visually valid) were effectively ignored.
*   **Action:** We switched to standard **MSE Loss** for training, treating all images as equally ground truth.
*   **Result:** Immediate and significant improvement. The model stopped ignoring "noisy" data and learned robust features from the entire dataset.

### Latest Results (25 Epochs)
*   **Mean Error:** **19.75 meters** (Previous Best: 26.39m)
*   **Median Error:** **15.95 meters** (Previous Best: 22.39m)
*   **Improvement:** **~25% reduction in error.**

### 3. Adding Learning Rate Scheduler
We noticed the loss spiking in later epochs, suggesting the learning rate was too high for fine-tuning. We implemented `ReduceLROnPlateau` (factor=0.1, patience=3).
*   **Outcome (50 Epochs):**
    *   **Mean Error:** **18.01 meters**
    *   **Median Error:** **14.53 meters**
    *   **Validation Loss:** Stabilized around ~473, indicating ResNet-18 had reached its capacity.

### 4. Upgrade to EfficientNet-B0
Given the limited dataset (~1,600 images), we switched from ResNet-18 (25M params) to **EfficientNet-B0** (5.3M params). EfficientNet is less prone to overfitting and learns more robust features.
*   **Outcome (50 Epochs + Scheduler):**
    *   **Mean Error:** **13.45 meters** (Previous Best: 18.01m)
    *   **Median Error:** **10.95 meters** (Previous Best: 14.53m)
    *   **Validation Loss:** Dropped massively to **~166** (was ~473).
*   **Status:** The model was still improving at Epoch 50, suggesting further training could break the 10-meter barrier.

### 5. Overfitting at 80 Epochs
We attempted to push the EfficientNet-B0 model further by training for 80 epochs.
*   **Outcome:** Performance regressed.
    *   **Mean Error:** **17.28 meters** (Worse than 13.45m)
    *   **Validation Loss:** Spiked to **328** (Doubled from 166).
*   **Analysis:** The model began to overfit the small training set after ~50 epochs. It memorized the training data instead of generalizing.
*   **Conclusion:** Simply training longer is not the solution. We need to make the model smarter (better architecture) or give it better data (higher resolution).

### 6. Failure of ConvNeXt-Tiny (On Mac)
We attempted to upgrade to **ConvNeXt-Tiny** (28M params) to capture better scene features.
*   **Outcome:** Training failed to converge. Loss exploded to ~4000+.
*   **Reason:** The model is too complex for the small batch size and limited compute available on the Mac (MPS). It likely needs hyperparameter tuning (warmup, weight decay) and larger batches.

### 7. Data Augmentation Experiment
We reverted to **EfficientNet-B0** but added robust data augmentation to prevent the overfitting we saw at epoch 80.
*   **Augmentations:** `RandomResizedCrop` (scale 0.8-1.0) and `ColorJitter`.
*   **Results (25 Epochs on Mac):**
    *   **Mean Error:** **18.48 meters**
    *   **Median Error:** **17.56 meters**
    *   **Analysis:** The results are worse than the non-augmented run (~13m), likely because the model needs more epochs to learn from harder data.

---

**Date:** Dec 30, 2025

## GPU Cluster Training & New Best Model

### Infrastructure Upgrade
We gained access to BGU's GPU cluster with a **NVIDIA GTX 1080 Ti** (CUDA). This allowed faster experimentation compared to local Mac training on MPS.

#### Cluster Setup Challenges
1.  **PyTorch CUDA Compatibility:** The latest PyTorch versions drop support for older GPU architectures. We installed `torch==1.12.1+cu116` for GTX 1080 Ti (CUDA 6.1 compute capability).
2.  **NumPy Version Conflict:** PyTorch 1.12 requires NumPy <2.0. Resolved by installing `numpy<2`.
3.  **Data Loading Bottleneck:** Initial training was slow (~5s/epoch) because the GPU was waiting for CPU data loading. Fixed with:
    *   `num_workers=4` for parallel data loading.
    *   `pin_memory=True` for faster GPU memory transfer.
    *   Increased batch size from 16 → 32.

### Smart Checkpoint Management
We implemented a robust checkpointing system that stores `best_loss` alongside the model weights. The training script now:
1.  Loads the previous `best_loss` at startup.
2.  Only saves a new checkpoint if the current validation loss **beats** the previous best.
3.  Prevents accidentally overwriting a good model with a worse one from a subsequent run.

### Baseline Run on GPU (No Augmentation)
We ran a clean 50-epoch baseline with **EfficientNet-B0** and **no augmentation** to establish the true performance on GPU.

*   **Training:** 1590 samples, 50 epochs, batch size 32, LR scheduler (ReduceLROnPlateau).
*   **Validation Loss:** Converged to **170.9** at Epoch 25.
*   **Final Test Results:**
    *   **Mean Error:** **15.27 meters** ⭐ **(Current Best Saved Model)**
    *   **Median Error:** **15.10 meters**
*   **Observations:**
    *   The model improved rapidly in the first 25 epochs (loss: 3081 → 170).
    *   After Epoch 25, the learning rate scheduler kicked in, and validation loss plateaued (~170-200), indicating the model reached its capacity without augmentation.
    *   The gap between the Mac run (13.45m) and GPU run (15.27m) is likely due to different random seeds and data loader shuffling.

---

## Higher Resolution & Model Experiments

### Image Reprocessing to 384×384
We reprocessed all 1,873 HEIC images from raw to **384×384** JPGs to support higher-resolution models. This took ~20 minutes on Mac.

### Experiment: EfficientNet-V2-S at 384×384
We tried the modern **EfficientNet-V2-S** (21M params) with native 384×384 input.

*   **Mac Training (25 Epochs):**
    *   Mean Error: **23.27m**, Median: **21.80m**
    *   Val Loss: 523.98
    *   Model was still learning, needed more epochs.

*   **GPU Training (Resumed, +50 Epochs):**
    *   Train Loss: ~108 (~10m error)
    *   Val Loss: ~452-490 (~21m error)
    *   Mean Error: **20.26m**, Median: **16.12m**

*   **Analysis:** Severe overfitting! Train/val gap of 2x indicates the model is memorizing, not generalizing.

### Diagnosing Overfitting
We identified multiple issues:
1.  **Too many parameters:** 21M params for 1,590 samples (rule of thumb: need 10-100 samples per param).
2.  **No augmentation:** Model saw identical images every epoch.
3.  **Wrong augmentation:** `RandomHorizontalFlip` was enabled but is invalid for GPS regression (would need to flip X coordinate too).

### Fixes Implemented
1.  **Enabled data augmentation:** `ColorJitter`, `RandomGrayscale`, `RandomRotation(±5°)`.
2.  **Removed `RandomHorizontalFlip`:** Invalid for GPS regression.
3.  **Added weight decay:** `weight_decay=1e-4` for L2 regularization.
4.  **Added resume flag:** Training can now continue from checkpoints with `resume=True`.

### Experiment: EfficientNet-B3 at 300×300 with Augmentation
We switched to **EfficientNet-B3** (12M params, native 300×300) - a balance between capacity and overfitting risk.

*   **GPU Training (50 Epochs):**
    *   Best Val Loss: **202.96** (at Epoch 14)
    *   Train Loss: ~140-150 (plateaued after epoch 20)
    *   Val Loss: ~211-215 (plateaued after LR reduction)

*   **Test Results:**
    *   Mean Error: **14.33 meters** ⭐ **(New Best!)**
    *   Median Error: **12.75 meters**

*   **Observations:**
    *   The train/val gap is much smaller (~140 vs ~200) - less overfitting!
    *   LR scheduler kicked in at epoch 14 (patience=3 was too aggressive).
    *   Model plateaued for 30+ epochs after LR reduction.

### Why Training Plateaued
| Epoch | Event |
|-------|-------|
| 1-14 | Rapid learning, val loss: 1819 → 202 |
| 15-18 | Val loss spiked (476, 610), LR scheduler triggered |
| 19-45 | LR reduced 10x (0.001 → 0.0001), stuck at ~211-215 |

The `ReduceLROnPlateau` with `patience=3` reduced learning rate too aggressively. The model couldn't escape its local minimum.

---

## Expert Friend's Suggestions & Best Model Yet

### The Challenge
After extensive experimentation, we were stuck at ~14m mean error. The key issues:
1.  **Model size vs. data:** Larger models (V2-S, B3) overfit despite augmentation.
2.  **Scheduler too aggressive:** `patience=3` killed learning prematurely.
3.  **Suboptimal loss function:** MSE squares errors, making outliers dominate.

### Expert Recommendations Implemented
We received detailed suggestions from an ML expert and implemented all of them:

#### 1. Architecture: EfficientNet-B0 + MLP Head
**Why:** B0 has only ~4M parameters, well-matched to our 1,500 training samples (rule of thumb: ~100-1000 samples per million params). The custom MLP head handles the non-linearity of GPS regression better than a single linear layer.

```python
self.backbone.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
```

#### 2. Image Resolution: 256×256
**Why:** Higher resolution (384×384) didn't improve results for B0 and just slowed training. 256×256 is EfficientNet-B0's native resolution and captures sufficient detail for campus landmarks.

*Action:* Reprocessed all images to `processed_images_256/`.

#### 3. Loss Function: HuberLoss (δ=1.0)
**Why:** MSE squares all errors, so a single 50m outlier dominates the loss. HuberLoss is quadratic for small errors but **linear** for large ones, making it robust to noisy GPS ground truth.

```python
criterion = nn.HuberLoss(delta=1.0)
```

*Effect:* Loss values are much smaller (7.5 vs 200+) because we're not squaring large errors.

#### 4. Data Augmentation (Revised)
**Why:** Previous augmentation included `RandomHorizontalFlip` which is **invalid for GPS regression** (flipping an image doesn't flip the X coordinate in our labels).

**New Pipeline:**
| Transform | Reason |
|-----------|--------|
| `RandomPerspective(0.3, p=0.5)` | Simulates different camera angles |
| `ColorJitter(0.2, 0.2, 0.2, 0.05)` | Handles lighting variation |
| `RandomGrayscale(p=0.1)` | Color-invariant features |
| `RandomRotation(±5°)` | Slight orientation changes |
| `RandomErasing(p=0.2)` | Simulates occlusions |

**Removed:** `RandomHorizontalFlip`, `RandomResizedCrop` (too aggressive).

#### 5. Training Hyperparameters
| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| Learning Rate | 1e-3 | 1e-3 | Keep aggressive start |
| Weight Decay | 1e-4 | 1e-4 | L2 regularization |
| Scheduler Patience | 3 | **7** | Let model explore longer before reducing LR |
| Scheduler Factor | 0.1 | **0.5** | Less aggressive reduction |
| Epochs | 50 | **75** | More time to converge |

#### 6. Data Split: 80/10/10
**Why:** The previous 5% test set (~94 samples) was too small for reliable metrics. Increased to 10% (188 samples) while maintaining stratified sampling across all campus locations.

---

### Results: Best Model! 🏆

**Experiment:** `b0_256_mlp` (75 epochs on GTX 1080 Ti)

| Metric | Value | Previous Best |
|--------|-------|---------------|
| **Mean Error** | **11.94 meters** | 14.33m (B3) |
| **Median Error** | **9.05 meters** ⬇️ | 12.75m (B3) |
| Best Val Loss | 7.50 (HuberLoss) | 202.96 (MSE) |

*Note: Loss values aren't directly comparable due to different loss functions.*

### Key Observations

1.  **Continuous Improvement:** The model saved its best checkpoint on the **final epoch (75)**, meaning it was still learning! No premature plateau.

2.  **Healthy Train/Val Gap:** Train loss (4.3) vs Val loss (7.5) shows ~2x gap. Some overfitting, but much healthier than V2-S (10x gap).

3.  **Median Under 10m!** The median error of **9.05m** breaks the 10-meter barrier for the first time. This means over half of predictions are within 10 meters.

### Visualization Analysis
Looking at the 50-sample visualization:
*   **Best cases (0.5-3m):** The model correctly identifies distinctive buildings with unique architecture.
*   **Worst cases (30-50m+):** Mostly night images and generic hallways/walkways that lack distinctive features.
*   **Night images remain challenging:** Many of the worst predictions are dark/low-light scenes.

---

### Resume Training (Epochs 76-94)

We resumed training from the epoch 75 checkpoint to see if further improvement was possible.

**Results:**
| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 75 | 4.29 | 7.50 | Starting point |
| 77 | 4.32 | **7.10** | New best! ✓ |
| 88 | 3.88 | 7.26 | Train improving, val stuck |
| 94 | 3.89 | 7.18 | Stopped (no improvement) |

**Best Val Loss:** 7.0977 (at epoch 77)

### Conclusion: Resuming Training Provides Diminishing Returns

The model has reached its capacity with the current architecture and data:

1.  **Widening Train/Val Gap:** Train loss continued dropping (4.3 → 3.8) while val loss stayed flat (~7.1-7.3). This indicates the model is memorizing training data rather than learning generalizable features.

2.  **Marginal Improvement:** Only 0.4 loss improvement (7.50 → 7.10) over 19 epochs. The model saved once at epoch 77 and never improved again.

3.  **Diminishing Returns:** Further training would only increase overfitting without improving test accuracy.

**Final Test Results (Epoch 77 Checkpoint):**
| Metric | Value |
|--------|-------|
| Mean Error | **11.63 meters** |
| Median Error | **8.85 meters** |
| Val Loss | 7.10 (HuberLoss) |

**Key Finding:** Resuming training from a converged model yields minimal gains. The EfficientNet-B0 + MLP architecture with 256×256 images has extracted all the learnable signal from our ~1,500 training images. To improve further, we need either **more data** or **better data quality**.

---

### Worst-Case Analysis

We analyzed the 25 worst predictions to identify failure patterns:

| Pattern | Count | Percentage |
|---------|-------|------------|
| Night/low-light images | 12/25 | 48% |
| Generic walkways/corridors | 8/25 | 32% |
| Building 32 Area | 6/25 | 24% |
| Library Area (open spaces) | 5/25 | 20% |

**Key Insights:**
1.  **Night images are the biggest problem** - Nearly half the worst predictions are dark scenes lacking distinctive features.
2.  **Generic architecture confuses the model** - Covered walkways, concrete pillars, and open plazas look identical across campus.
3.  **Some might be GPS label errors** - A few images show distinctive buildings but have huge errors, suggesting incorrect ground truth.

---

## Paths Forward for Sub-10m Mean Error

### Option 1: Light GPS Accuracy Weighting (New Idea)
We have a `gps_accuracy_m` column in the dataset (manually corrected: all 15m+ values set to 7m). We could apply **very light weighting** based on this:

```python
# Example: weight = 1 / (1 + 0.02 * accuracy)
# accuracy = 2m  → weight = 0.96
# accuracy = 7m  → weight = 0.88
# accuracy = 15m → weight = 0.77
# Ratio of best to worst: only ~1.25x (not 56x like before!)
```

**Why this might help:**
*   Gives slight preference to high-confidence GPS labels during training.
*   Doesn't ignore manually-corrected samples (unlike our original 1/σ² weighting).
*   The model can still learn from all data, just with subtle guidance.

### Option 2: Data Quality
The GPS ground truth has inherent noise. Some images may have incorrect labels.

*   **Manual Review:** Check the worst predictions - are they model errors or label errors?
*   **Use Model to Find Bad Labels:** High training loss on a sample = potential mislabel.

### Option 3: More Training Data
With ~1,500 samples and 4M parameters, we're near the data limit.

*   **Collect More Photos:** Especially night scenes and underrepresented areas.
*   **Stronger Night Augmentation:** More aggressive brightness/contrast variation.

### Option 4: Architectural Improvements
*   **Two-Stage Approach:** First classify the region, then regress within it.
*   **Attention Mechanisms:** Help model focus on distinctive landmarks.

### Option 5: Ensemble
*   Train 3-5 models with different random seeds.
*   Average predictions to reduce variance (typically 5-10% improvement).

---

### Current Best Model Summary
| Experiment | Mean Error | Median Error | Val Loss |
|------------|------------|--------------|----------|
| `b0_256_mlp` (epoch 77) | **11.63m** | **8.85m** | 7.10 |
