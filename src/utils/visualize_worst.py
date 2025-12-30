import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

def denormalize_to_latlon(x_meters, y_meters, ref_lat, ref_lon):
    """Convert local metric coordinates back to Lat/Lon"""
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    
    lat = ref_lat + (y_meters / METERS_PER_LAT)
    lon = ref_lon + (x_meters / METERS_PER_LON)
    return lat, lon

def latlon_to_web_mercator(lat, lon):
    """Convert Lat/Lon to Web Mercator (EPSG:3857) for contextily"""
    r = 6378137.0
    x = np.radians(lon) * r
    y = np.log(np.tan(np.pi/4 + np.radians(lat)/2)) * r
    return x, y

def visualize_worst_samples(num_worst=25):
    # 1. Setup
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images_256')
    model_path = os.path.join(project_root, 'checkpoints', 'best_b0_256_mlp.pth')
    indices_path = os.path.join(project_root, 'outputs', 'test_indices.npy')
    output_plot = os.path.join(project_root, 'outputs', 'worst_samples_visualization.png')
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return
    if not os.path.exists(indices_path):
        print("Test indices not found.")
        return

    # Load Reference Point
    df_temp = pd.read_csv(csv_file)
    ref_lat = df_temp['lat'].mean()
    ref_lon = df_temp['lon'].mean()

    # 2. Load Data
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    test_indices = np.load(indices_path)
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Run Inference on ALL Test Data
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_indices = []
    
    print("Running inference on test set...")
    current_idx = 0
    with torch.no_grad():
        for inputs, labels, weights in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            batch_size = inputs.size(0)
            batch_indices = test_indices[current_idx : current_idx + batch_size]
            current_idx += batch_size
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.extend(batch_indices)
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_indices = np.array(all_indices)
    
    # Calculate Errors
    errors = np.linalg.norm(all_preds - all_labels, axis=1)
    
    # 4. Select ONLY Worst samples
    sorted_idxs = np.argsort(errors)
    worst_idxs = sorted_idxs[-num_worst:][::-1]  # Reverse to show worst first
    
    # 5. Plotting - 5 columns, 5 rows = 25 samples
    cols = 5
    rows = 5
    
    fig = plt.figure(figsize=(25, 30))
    gs = gridspec.GridSpec(rows * 2, cols, height_ratios=[3, 2] * rows)
    
    print(f"Plotting {num_worst} worst samples...")
    
    # Also print details to console
    print("\n" + "="*80)
    print("WORST 25 PREDICTIONS (sorted by error, worst first)")
    print("="*80)
    
    for i, idx_local in enumerate(worst_idxs):
        row = (i // cols) * 2
        col = i % cols
        
        # Data
        idx_global = all_indices[idx_local]
        pred_local = all_preds[idx_local]
        true_local = all_labels[idx_local]
        error_m = errors[idx_local]
        
        # Get filename and source
        img_name = full_dataset.data_frame.iloc[idx_global]['filename']
        source_file = full_dataset.data_frame.iloc[idx_global].get('source_file', 'Unknown')
        img_path = os.path.join(img_dir, img_name)
        
        # Print to console
        print(f"#{i+1:2d} | Error: {error_m:6.1f}m | Source: {source_file:30s} | File: {img_name}")
        
        try:
            image = Image.open(img_path)
        except Exception:
            image = None
            
        # Coordinates
        p_lat, p_lon = denormalize_to_latlon(pred_local[0], pred_local[1], ref_lat, ref_lon)
        t_lat, t_lon = denormalize_to_latlon(true_local[0], true_local[1], ref_lat, ref_lon)
        
        px, py = latlon_to_web_mercator(p_lat, p_lon)
        tx, ty = latlon_to_web_mercator(t_lat, t_lon)
        
        # --- Plot Image ---
        ax_img = fig.add_subplot(gs[row, col])
        if image:
            ax_img.imshow(image)
        else:
            ax_img.text(0.5, 0.5, "Img Not Found", ha='center')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Color title based on error
        color = 'orange' if error_m < 40 else 'red'
        ax_img.set_title(f"#{i+1} | {error_m:.1f}m\n{source_file[:20]}", color=color, fontweight='bold', fontsize=10)
        
        # --- Plot Map ---
        ax_map = fig.add_subplot(gs[row + 1, col])
        
        ax_map.scatter(tx, ty, c='lime', s=100, edgecolors='black', label='True', zorder=5)
        ax_map.scatter(px, py, c='red', s=100, edgecolors='black', label='Pred', zorder=5)
        ax_map.plot([tx, px], [ty, py], 'k--', alpha=0.6, zorder=4)
        
        # Dynamic margin based on error
        margin = max(100, error_m * 1.5)
        center_x = (tx + px) / 2
        center_y = (ty + py) / 2
        
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

    plt.suptitle(f"WORST {num_worst} PREDICTIONS\nMean Error: {errors.mean():.1f}m | Median: {np.median(errors):.1f}m", 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_plot}")

if __name__ == "__main__":
    visualize_worst_samples(25)

