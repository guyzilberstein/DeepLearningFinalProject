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
