"""
Visualize worst 25 ensemble predictions on external test set.
Uses test_dataset.csv (1,023 samples from TestPhotos).
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import contextily as ctx
import pandas as pd
from PIL import Image

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

def denormalize_to_latlon(x_meters, y_meters, ref_lat, ref_lon):
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    lat = ref_lat + (y_meters / METERS_PER_LAT)
    lon = ref_lon + (x_meters / METERS_PER_LON)
    return lat, lon

def latlon_to_web_mercator(lat, lon):
    r = 6378137.0
    x = np.radians(lon) * r
    y = np.log(np.tan(np.pi/4 + np.radians(lat)/2)) * r
    return x, y

def load_model(model_path, device):
    model = CampusLocator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def visualize_worst_ensemble(num_worst=25):
    # Setup - use external test set
    test_csv = os.path.join(project_root, 'data', 'test_dataset.csv')
    train_csv = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    output_plot = os.path.join(project_root, 'outputs', 'worst_ensemble_visualization.png')
    
    model_names = ['convnext_tiny_v1', 'convnext_tiny_v2', 'convnext_tiny_v3']
    
    # Load reference point from training set (same as used during training)
    df_train = pd.read_csv(train_csv)
    ref_lat = df_train['lat'].mean()
    ref_lon = df_train['lon'].mean()
    
    # Load external test set
    test_dataset = CampusDataset(csv_file=test_csv, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Loaded external test set: {len(test_dataset)} samples")
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models = []
    for name in model_names:
        model_path = os.path.join(checkpoint_dir, f'best_{name}.pth')
        if os.path.exists(model_path):
            models.append(load_model(model_path, device))
            print(f"Loaded {name}")
    
    # Run ensemble inference
    all_preds = []
    all_labels = []
    
    print("Running ensemble inference...")
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            inputs = inputs.to(device)
            
            # Get ensemble prediction (average of all models)
            model_outputs = []
            for model in models:
                outputs = model(inputs)
                model_outputs.append(outputs.cpu().numpy())
            
            ensemble_pred = np.mean(model_outputs, axis=0)
            all_preds.append(ensemble_pred)
            all_labels.append(labels.numpy())
    
    # Stack predictions
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Get image names from test dataset
    all_img_names = test_dataset.data_frame['filename'].tolist()
    all_sources = test_dataset.data_frame.get('source_file', pd.Series(['Unknown'] * len(test_dataset))).tolist()
    
    # Calculate errors
    ensemble_errors = np.linalg.norm(all_preds - all_labels, axis=1)
    
    # Select worst samples
    sorted_idxs = np.argsort(ensemble_errors)
    worst_idxs = sorted_idxs[-num_worst:][::-1]  # Worst first
    
    # Plot - 5x5 grid
    cols = 5
    rows = 5
    
    fig = plt.figure(figsize=(25, 30))
    gs = gridspec.GridSpec(rows * 2, cols, height_ratios=[3, 2] * rows)
    
    print(f"\nPlotting {num_worst} worst ensemble predictions...")
    print("\n" + "="*80)
    print(f"WORST {num_worst} ENSEMBLE PREDICTIONS (sorted by error)")
    print(f"Test Set: {len(test_dataset)} samples | Mean: {ensemble_errors.mean():.2f}m | Median: {np.median(ensemble_errors):.2f}m")
    print("="*80)
    
    for i, idx in enumerate(worst_idxs):
        row = (i // cols) * 2
        col = i % cols
        
        ensemble_pred = all_preds[idx]
        true_local = all_labels[idx]
        error_m = ensemble_errors[idx]
        
        # Get image info
        img_name = all_img_names[idx]
        source_file = all_sources[idx] if idx < len(all_sources) else 'Unknown'
        img_path = os.path.join(img_dir, img_name)
        
        # Print to console
        print(f"#{i+1:2d} | Error: {error_m:5.1f}m | {source_file:30s} | {img_name}")
        
        try:
            image = Image.open(img_path)
        except:
            image = None
        
        # Coordinates
        p_lat, p_lon = denormalize_to_latlon(ensemble_pred[0], ensemble_pred[1], ref_lat, ref_lon)
        t_lat, t_lon = denormalize_to_latlon(true_local[0], true_local[1], ref_lat, ref_lon)
        
        px, py = latlon_to_web_mercator(p_lat, p_lon)
        tx, ty = latlon_to_web_mercator(t_lat, t_lon)
        
        # Plot image
        ax_img = fig.add_subplot(gs[row, col])
        if image:
            ax_img.imshow(image)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        color = 'orange' if error_m < 30 else 'red'
        ax_img.set_title(f"#{i+1} | {error_m:.1f}m\n{source_file[:20]}", color=color, fontweight='bold', fontsize=10)
        
        # Plot map
        ax_map = fig.add_subplot(gs[row + 1, col])
        ax_map.scatter(tx, ty, c='lime', s=100, edgecolors='black', label='True', zorder=5)
        ax_map.scatter(px, py, c='red', s=100, edgecolors='black', label='Ensemble', zorder=5)
        ax_map.plot([tx, px], [ty, py], 'k--', alpha=0.6, zorder=4)
        
        margin = max(100, error_m * 1.5)
        center_x, center_y = (tx + px) / 2, (ty + py) / 2
        ax_map.set_xlim(center_x - margin, center_x + margin)
        ax_map.set_ylim(center_y - margin, center_y + margin)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        
        try:
            ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857', zoom='auto')
        except:
            pass
        
        if i == 0:
            ax_map.legend(loc='upper right', fontsize='x-small')
    
    plt.suptitle(f"WORST {num_worst} ENSEMBLE PREDICTIONS\nMean: {ensemble_errors.mean():.2f}m | Median: {np.median(ensemble_errors):.2f}m", 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_plot}")

if __name__ == "__main__":
    visualize_worst_ensemble(25)

