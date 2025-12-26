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
