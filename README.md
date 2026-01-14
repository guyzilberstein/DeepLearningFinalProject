# Campus Image-to-GPS Localization

## Overview
This repository contains a deep learning solution for **Image-to-GPS Regression**. The goal is to predict the precise real-world location (Latitude, Longitude) of a photo taken within a predefined area of the university campus, using only visual features.

**Method:**
- **Model:** ConvNeXt-Tiny Ensemble (3 models)
- **Architecture:** ConvNeXt backbone with custom MLP regression head
- **Training:** Trained on 320x320 inputs with robust augmentation (Night simulation, rotation, perspective)
- **Loss:** Huber Loss (robust to outliers/noisy GPS labels)

## Setup

**Python Version:** 3.9+ recommended (tested on 3.9, 3.10)

1. **Create Environment:**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using Conda
   conda create -n campus-gps python=3.10
   conda activate campus-gps
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights:**
   Due to file size limits, the trained models are hosted externally.
   - See **[Checkpoints README](checkpoints/README.md)** for the download link.
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
- **[Repository Guide](REPO_GUIDE.md)**: Deep dive into the codebase structure, technical implementation, and file purposes.
- **[Data README](data/README.md)**: Dataset collection, raw photo download links, and processing details.
