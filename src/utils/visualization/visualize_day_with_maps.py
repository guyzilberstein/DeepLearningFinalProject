"""
Visualize day/test predictions with maps.
Shows 7 best, 7 medium, and 7 worst predictions for analysis.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import contextily as ctx
import pandas as pd
from PIL import Image
import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from torch.utils.data import DataLoader
from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

OUTPUT_DIR = os.path.join(project_root, 'outputs', 'visualizations_final')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def latlon_to_web_mercator(lat, lon):
    """Convert Lat/Lon to Web Mercator (EPSG:3857)."""
    r = 6378137.0
    x = np.radians(lon) * r
    y = np.log(np.tan(np.pi/4 + np.radians(lat)/2)) * r
    return x, y


def denormalize_to_latlon(x_meters, y_meters, ref_lat, ref_lon):
    """Convert local metric coordinates back to Lat/Lon."""
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    lat = ref_lat + (y_meters / METERS_PER_LAT)
    lon = ref_lon + (x_meters / METERS_PER_LON)
    return lat, lon


def get_ensemble_predictions(data_loader, device):
    """Get ensemble predictions."""
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    model_names = ['convnext_tiny_v1', 'convnext_tiny_v2', 'convnext_tiny_v3']
    
    models = []
    for name in model_names:
        model_path = os.path.join(checkpoint_dir, f'best_{name}.pth')
        if os.path.exists(model_path):
            model = CampusLocator().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            models.append(model)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            batch_preds = [model(inputs).cpu().numpy() for model in models]
            ensemble_pred = np.mean(batch_preds, axis=0)
            all_preds.append(ensemble_pred)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def visualize_day_with_maps(num_per_category=7):
    """Create day/test predictions visualization with maps."""
    test_csv = os.path.join(project_root, 'data', 'test_dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
    ref_coords_path = os.path.join(project_root, 'data', 'metadata', 'reference_coords.json')
    
    with open(ref_coords_path, 'r') as f:
        ref_coords = json.load(f)
    ref_lat, ref_lon = ref_coords['ref_lat'], ref_coords['ref_lon']
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = CampusDataset(csv_file=test_csv, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Running inference on test set ({len(test_dataset)} samples)...")
    predictions, labels = get_ensemble_predictions(test_loader, device)
    errors = np.linalg.norm(predictions - labels, axis=1)
    
    # Sort by error
    sorted_idxs = np.argsort(errors)
    
    # Select: best, medium, worst
    n = min(num_per_category, len(sorted_idxs) // 3)
    best_idxs = sorted_idxs[:n]
    worst_idxs = sorted_idxs[-n:]
    mid_start = len(sorted_idxs) // 2 - n // 2
    medium_idxs = sorted_idxs[mid_start:mid_start + n]
    
    # Combine with labels
    selected = []
    categories = []
    colors_cat = []
    
    for idx in best_idxs:
        selected.append(idx)
        categories.append('BEST')
        colors_cat.append('#27AE60')  # Green
    for idx in medium_idxs:
        selected.append(idx)
        categories.append('MEDIUM')
        colors_cat.append('#F39C12')  # Orange
    for idx in worst_idxs:
        selected.append(idx)
        categories.append('WORST')
        colors_cat.append('#E74C3C')  # Red
    
    # Create figure
    num_samples = len(selected)
    cols = n
    rows = 3  # Best, Medium, Worst
    
    fig = plt.figure(figsize=(5 * cols, 6 * rows), facecolor='white')
    gs = gridspec.GridSpec(rows * 2, cols, height_ratios=[3, 2] * rows, hspace=0.15, wspace=0.1)
    
    print(f"Plotting {num_samples} test samples...")
    
    for i, (idx, category, cat_color) in enumerate(zip(selected, categories, colors_cat)):
        row_group = i // cols  # 0, 1, or 2
        col = i % cols
        
        pred = predictions[idx]
        true = labels[idx]
        error_m = errors[idx]
        
        img_name = test_dataset.data_frame.iloc[idx]['filename']
        img_path = os.path.join(img_dir, img_name)
        
        try:
            image = Image.open(img_path)
        except Exception:
            image = None
        
        # Coordinates
        p_lat, p_lon = denormalize_to_latlon(pred[0], pred[1], ref_lat, ref_lon)
        t_lat, t_lon = denormalize_to_latlon(true[0], true[1], ref_lat, ref_lon)
        
        px, py = latlon_to_web_mercator(p_lat, p_lon)
        tx, ty = latlon_to_web_mercator(t_lat, t_lon)
        
        # Plot Image
        ax_img = fig.add_subplot(gs[row_group * 2, col])
        if image:
            ax_img.imshow(image)
        else:
            ax_img.text(0.5, 0.5, "Image not found", ha='center', va='center')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Add border and title
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_color(cat_color)
            spine.set_linewidth(3)
        
        ax_img.set_title(f'{error_m:.1f}m', fontsize=11, fontweight='bold', color=cat_color)
        
        # Add category label on first column
        if col == 0:
            ax_img.text(-0.15, 0.5, category, transform=ax_img.transAxes,
                       fontsize=14, fontweight='bold', color=cat_color,
                       va='center', ha='right', rotation=90)
        
        # Plot Map
        ax_map = fig.add_subplot(gs[row_group * 2 + 1, col])
        
        # Actual location (green circle)
        ax_map.scatter(tx, ty, c='#27AE60', s=120, edgecolors='white', 
                      linewidth=2, label='Actual', zorder=5, marker='o')
        # Predicted location (red X)
        ax_map.scatter(px, py, c='#E74C3C', s=120, edgecolors='white',
                      linewidth=2, label='Predicted', zorder=5, marker='X')
        # Connection line
        ax_map.plot([tx, px], [ty, py], 'k--', alpha=0.6, linewidth=2, zorder=4)
        
        # Dynamic margin based on error
        margin = max(50, error_m * 2)
        center_x = (tx + px) / 2
        center_y = (ty + py) / 2
        
        ax_map.set_xlim(center_x - margin, center_x + margin)
        ax_map.set_ylim(center_y - margin, center_y + margin)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        
        try:
            ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik,
                           crs='EPSG:3857', zoom='auto')
        except:
            pass
        
        # Add legend only to first map
        if i == 0:
            ax_map.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    fig.suptitle(f'Test Set (Day) Predictions Analysis\n'
                 f'Mean Error: {mean_error:.2f}m | Median: {median_error:.2f}m | {len(test_dataset)} samples',
                fontsize=16, fontweight='bold', color='#2C3E50', y=0.98)
    
    # Legend at bottom
    legend_text = '● Actual Location (Green)     ✕ Predicted Location (Red)     --- Error Distance'
    fig.text(0.5, 0.01, legend_text, ha='center', fontsize=11, color='#666666')
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, 'day_predictions_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=7, help='Number per category (best/medium/worst)')
    args = parser.parse_args()
    visualize_day_with_maps(num_per_category=args.num)
