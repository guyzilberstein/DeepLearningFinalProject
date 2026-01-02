"""
Export worst predictions to CSV for Google Maps import.
Output format: name, latitude, longitude (simple 3-column format)

Usage:
1. Run this script to generate outputs/worst_for_maps.csv
2. Import to Google My Maps (mymaps.google.com)
3. Drag points to correct locations
4. Export from Google Maps and use import_corrections.py to convert back
"""
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
import pandas as pd

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator


def export_for_maps(num_worst: int = 25, experiment_name: str = "b0_256_mlp"):
    """
    Export worst predictions as simple CSV for Google Maps import.
    
    Args:
        num_worst: Number of worst predictions to export
        experiment_name: Model experiment name
    """
    # Setup paths
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images_320')
    model_path = os.path.join(project_root, 'checkpoints', f'best_{experiment_name}.pth')
    indices_path = os.path.join(project_root, 'outputs', 'test_indices.npy')
    output_csv = os.path.join(project_root, 'outputs', 'worst_for_maps.csv')
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(indices_path):
        print("Test indices not found. Run training first.")
        return

    # Load data
    df_full = pd.read_csv(csv_file)
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    test_indices = np.load(indices_path)
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Run inference
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
    
    # Calculate errors
    errors = np.linalg.norm(all_preds - all_labels, axis=1)
    
    # Select worst
    sorted_idxs = np.argsort(errors)
    worst_idxs = sorted_idxs[-num_worst:][::-1]
    
    # Build simple output
    rows = []
    print(f"\nExporting {num_worst} worst predictions:")
    print("="*60)
    
    for i, idx_local in enumerate(worst_idxs):
        idx_global = all_indices[idx_local]
        error_m = errors[idx_local]
        
        row_data = df_full.iloc[idx_global]
        img_name = row_data['filename']
        lat = row_data['lat']
        lon = row_data['lon']
        
        rows.append({
            'name': img_name,
            'latitude': lat,
            'longitude': lon
        })
        
        print(f"#{i+1:2d} | {error_m:5.1f}m | {img_name}")
    
    # Save CSV (Google Maps compatible format)
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_csv, index=False)
    
    print("="*60)
    print(f"\nSaved: {output_csv}")
    print("\nNext steps:")
    print("1. Go to mymaps.google.com")
    print("2. Create new map → Import → Select this CSV")
    print("3. Drag points to correct locations")
    print("4. Export as KML or use the coordinates shown")
    print("5. Run import_corrections.py with the exported data")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export worst predictions for Google Maps")
    parser.add_argument("--num", "-n", type=int, default=25, help="Number of worst predictions")
    parser.add_argument("--model", "-m", default="b0_256_mlp", help="Experiment name")
    args = parser.parse_args()
    
    export_for_maps(num_worst=args.num, experiment_name=args.model)
