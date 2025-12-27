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
