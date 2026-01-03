"""
Campus GPS Prediction - Ensemble Model
=======================================
This script predicts GPS coordinates from a campus image using an ensemble
of 3 models trained with different random seeds.

Usage:
    python predict.py <image_path>
    
Output:
    Predicted latitude and longitude
"""
import torch
import sys
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model.network import CampusLocator

# Configuration
INPUT_SIZE = 320
MODEL_NAMES = [
    'best_b0_320_seed42.pth',
    'best_b0_320_seed123.pth', 
    'best_b0_320_seed456.pth'
]

# Reference point for coordinate conversion (center of training data)
REF_LAT = 31.261976298245617
REF_LON = 34.80279455555555
METERS_PER_LAT = 111132.0
METERS_PER_LON = 111132.0 * np.cos(np.radians(REF_LAT))


def load_models(device):
    """Load all ensemble models."""
    models = []
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    for name in MODEL_NAMES:
        model_path = os.path.join(checkpoint_dir, name)
        if not os.path.exists(model_path):
            print(f"Warning: {name} not found, skipping...")
            continue
            
        model = CampusLocator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        models.append(model)
    
    return models


def preprocess_image(image_path):
    """Load and preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


def meters_to_latlon(x_meters, y_meters):
    """Convert meters offset to latitude/longitude."""
    lat = REF_LAT + (y_meters / METERS_PER_LAT)
    lon = REF_LON + (x_meters / METERS_PER_LON)
    return lat, lon


def predict(image_path):
    """
    Predict GPS coordinates for an image using ensemble of models.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        (latitude, longitude) tuple
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    
    # Load models
    models = load_models(device)
    if len(models) == 0:
        raise RuntimeError("No models found! Check checkpoints directory.")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Get predictions from each model
    predictions = []
    with torch.no_grad():
        for model in models:
            output = model(image_tensor)
            predictions.append(output.cpu().numpy()[0])  # [x_meters, y_meters]
    
    # Average predictions (ensemble)
    ensemble_pred = np.mean(predictions, axis=0)
    x_meters, y_meters = ensemble_pred[0], ensemble_pred[1]
    
    # Convert to lat/lon
    lat, lon = meters_to_latlon(x_meters, y_meters)
    
    return lat, lon


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/processed_images_320/LibraryArea_IMG_7288.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    lat, lon = predict(image_path)
    
    print(f"Predicted Location:")
    print(f"  Latitude:  {lat:.6f}")
    print(f"  Longitude: {lon:.6f}")
    print(f"  Google Maps: https://www.google.com/maps?q={lat},{lon}")

