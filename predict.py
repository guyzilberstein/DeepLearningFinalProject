"""
Project 4: Image-to-GPS Regression
==================================
Evaluation API Implementation.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# --- CONFIGURATION ---

# 1. Models to Load (ConvNeXt-Tiny Ensemble)
MODEL_FILENAMES = [
    'best_convnext_tiny_v1.pth',
    'best_convnext_tiny_v2.pth',
    'best_convnext_tiny_v3.pth'
]

INPUT_SIZE = 320

# Reference point (Center of Area)
REF_LAT = 31.261976298245617
REF_LON = 34.80279455555555
METERS_PER_LAT = 111132.0
METERS_PER_LON = 111132.0 * np.cos(np.radians(REF_LAT))

# Global cache
_MODELS = []
_DEVICE = None

def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                             else "mps" if torch.backends.mps.is_available() 
                             else "cpu")
    return _DEVICE

def load_models_cached():
    """
    Loads the ConvNeXt ensemble. 
    Raises specific error if weights are missing (as they are external).
    """
    global _MODELS
    if _MODELS:
        return _MODELS

    device = get_device()
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    # Ensure src is in path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from src.model.network import CampusLocator
    except ImportError:
        # Fallback for evaluation environments
        sys.path.append(project_root)
        from src.model.network import CampusLocator

    loaded_models = []
    missing_files = []

    for name in MODEL_FILENAMES:
        path = os.path.join(checkpoint_dir, name)
        if not os.path.exists(path):
            missing_files.append(name)
            continue
        
        # Initialize ConvNeXt Architecture
        model = CampusLocator().to(device)
        
        # Load weights
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        loaded_models.append(model)

    if missing_files:
        error_msg = (
            "\n" + "="*60 + "\n"
            "CRITICAL ERROR: Model weights not found!\n"
            "The following files are missing from the 'checkpoints/' directory:\n"
            + "\n".join([f" - {f}" for f in missing_files]) + "\n\n"
            "Please download them from the Google Drive link in 'checkpoints/README.md'\n"
            "and place them in the 'checkpoints/' folder before running.\n"
            + "="*60 + "\n"
        )
        raise FileNotFoundError(error_msg)
    
    _MODELS = loaded_models
    return _MODELS

def meters_to_latlon(x_meters, y_meters):
    lat = REF_LAT + (y_meters / METERS_PER_LAT)
    lon = REF_LON + (x_meters / METERS_PER_LON)
    return lat, lon

# --- REQUIRED API FUNCTION ---
def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS latitude and longitude from a single RGB image.
    
    Args:
        image: numpy.ndarray of shape (H, W, 3) in RGB format, dtype=uint8
        
    Returns:
        np.array([latitude, longitude], dtype=float32)
    """
    # 1. Setup
    device = get_device()
    models = load_models_cached()
    
    # 2. Preprocess
    pil_img = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # 3. Inference (Ensemble Average)
    predictions = []
    with torch.no_grad():
        for model in models:
            output = model(img_tensor)
            predictions.append(output.cpu().numpy()[0])
            
    avg_pred = np.mean(predictions, axis=0)
    
    # 4. Convert to Lat/Lon
    lat, lon = meters_to_latlon(avg_pred[0], avg_pred[1])
    
    return np.array([lat, lon], dtype=np.float32)

# --- CLI FOR MANUAL TESTING ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    path = sys.argv[1]
    if os.path.exists(path):
        # Emulate the API call with exact types
        pil_image = Image.open(path).convert('RGB')
        img_arr = np.array(pil_image, dtype=np.uint8)
        
        try:
            result = predict_gps(img_arr)
            # Only print the result array for testing purposes
            print(result)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
