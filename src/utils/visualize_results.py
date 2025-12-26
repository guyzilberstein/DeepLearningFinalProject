import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import contextily as ctx
import pandas as pd

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
    # Earth radius
    r = 6378137.0
    
    x = np.radians(lon) * r
    y = np.log(np.tan(np.pi/4 + np.radians(lat)/2)) * r
    return x, y

def visualize_results():
    # 1. Setup
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images')
    model_path = os.path.join(project_root, 'checkpoints', 'best_campus_locator.pth')
    output_plot = os.path.join(project_root, 'outputs', 'results_visualization.png')
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Run train.py first.")
        return

    # Get Reference Point for Coordinate Conversion
    df_temp = pd.read_csv(csv_file)
    ref_lat = df_temp['lat'].mean()
    ref_lon = df_temp['lon'].mean()

    # 2. Load Data
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    
    # Select Random Indices
    num_samples = 20
    # Ensure we don't ask for more samples than exist
    num_samples = min(num_samples, len(full_dataset))
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    
    dataset = torch.utils.data.Subset(full_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds_local = []
    all_labels_local = []
    
    print("Running inference for visualization...")
    
    with torch.no_grad():
        for inputs, labels, weights in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds_local.append(outputs.cpu().numpy())
            all_labels_local.append(labels.numpy())
            
    all_preds_local = np.vstack(all_preds_local)
    all_labels_local = np.vstack(all_labels_local)
    
    # 4. Convert Local Metrics -> Lat/Lon -> Web Mercator (EPSG:3857)
    # We need Web Mercator for the Map Tiles
    
    # Calculate Lat/Lon
    pred_lat, pred_lon = denormalize_to_latlon(all_preds_local[:,0], all_preds_local[:,1], ref_lat, ref_lon)
    true_lat, true_lon = denormalize_to_latlon(all_labels_local[:,0], all_labels_local[:,1], ref_lat, ref_lon)
    ref_lat_wm, ref_lon_wm = latlon_to_web_mercator(ref_lat, ref_lon)
    
    # Calculate Web Mercator X/Y
    pred_x_wm, pred_y_wm = latlon_to_web_mercator(pred_lat, pred_lon)
    true_x_wm, true_y_wm = latlon_to_web_mercator(true_lat, true_lon)
    
    # 5. Plotting
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    # --- Main Map Plot ---
    ax_map = fig.add_subplot(gs[0])
    
    # Calculate errors in meters (using the original local coordinates is accurate enough)
    errors = np.linalg.norm(all_preds_local - all_labels_local, axis=1)
    
    # Plot Actual GPS (Blue circles)
    # Increased size (s=150)
    ax_map.scatter(true_x_wm, true_y_wm, c='blue', alpha=0.5, label='Actual GPS', s=150, edgecolors='white', linewidth=2, zorder=3)
    
    # Plot Predicted GPS (Color-coded by error)
    # Increased size (s=150)
    sc = ax_map.scatter(pred_x_wm, pred_y_wm, c=errors, cmap='RdYlGn_r', 
                        alpha=0.9, label='Predicted', s=150, edgecolors='black', linewidth=1.5, vmin=0, vmax=30, zorder=4)
    
    # Draw error lines
    for i in range(len(true_x_wm)):
        ax_map.plot([true_x_wm[i], pred_x_wm[i]], [true_y_wm[i], pred_y_wm[i]], 
                    color='black', alpha=0.4, linestyle='--', linewidth=1, zorder=2)
        
    ax_map.set_title(f"Campus Localization Results (N={num_samples})", fontsize=14)
    ax_map.set_xlabel("Longitude (Web Mercator)", fontsize=10)
    ax_map.set_ylabel("Latitude (Web Mercator)", fontsize=10)
    
    # Add 20% margin to zoom out
    x_min, x_max = ax_map.get_xlim()
    y_min, y_max = ax_map.get_ylim()
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2
    ax_map.set_xlim(x_min - margin_x, x_max + margin_x)
    ax_map.set_ylim(y_min - margin_y, y_max + margin_y)
    
    # Turn off axis numbers as they are large Web Mercator coordinates
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    
    # Add Base Map (OpenStreetMap)
    # crs=3857 tells contextily we are providing Web Mercator coordinates
    try:
        ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857', zoom='auto')
    except Exception as e:
        print(f"Could not fetch map tiles: {e}")
        print("Continuing without background map...")
    
    ax_map.legend()
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax_map)
    cbar.set_label('Error Distance (meters)', rotation=270, labelpad=15)
    
    # --- Error Distribution Histogram ---
    ax_hist = fig.add_subplot(gs[1])
    
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    ax_hist.hist(errors, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.1f}m')
    ax_hist.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_error:.1f}m')
    
    ax_hist.set_title("Error Distribution", fontsize=14)
    ax_hist.set_xlabel("Error (meters)", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Visualization saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    visualize_results()
