# Project 4: Image-to-GPS Regression

## Overview
This repository contains the solution for **Project 4: Image-to-GPS Regression**.
The model predicts exact GPS coordinates from campus photos.

**Method:**
- **Model:** ConvNeXt-Tiny Ensemble (3 models)
- **Architecture:** ConvNeXt backbone with custom MLP regression head
- **Training:** Trained on 320x320 inputs with robust augmentation (Night simulation, rotation, perspective)

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Model Weights:**
   Due to file size limits, the trained models are hosted externally.
   - Read `checkpoints/README.md` for the download link.
   - Place `best_convnext_tiny_v1.pth`, `best_convnext_tiny_v2.pth`, and `best_convnext_tiny_v3.pth` in the `checkpoints/` folder.

## Usage

### Evaluation API
The required function `predict_gps` is implemented in `predict.py`.

```python
from predict import predict_gps
from PIL import Image
import numpy as np

# Load image as numpy array (RGB)
image = np.array(Image.open("data/images/example.jpg").convert('RGB'))

# Predict
coords = predict_gps(image)
print(f"Lat: {coords[0]}, Lon: {coords[1]}")
```

### CLI
```bash
python predict.py data/images/LibraryArea_IMG_8164.jpg
```

## Dataset Structure
- `data/images/`: Contains all project images.
- `data/gt.csv`: Ground truth file (`image_name`, `Latitude`, `Longitude`).

## Documentation
- `REPO_GUIDE.md`: Deep dive into the codebase structure, technical implementation, and file purposes.
- `data/README.md`: Dataset collection, raw photo download links, and processing details.
