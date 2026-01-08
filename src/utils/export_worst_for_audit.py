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
from torch.utils.data import DataLoader
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


def export_for_maps(num_worst: int = 25, experiment_name: str = "convnext_tiny_v1", model_type: str = "convnext"):
    """
    Export worst predictions as simple CSV for Google Maps import.
    
    Args:
        num_worst: Number of worst predictions to export
        experiment_name: Model experiment name
        model_type: "convnext" or "efficientnet" (determines which model class to load)
    """
    # Setup paths - use external test set
    test_csv = os.path.join(project_root, 'data', 'test_dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images_320')
    model_path = os.path.join(project_root, 'checkpoints', f'best_{experiment_name}.pth')
    output_csv = os.path.join(project_root, 'outputs', 'worst_for_maps.csv')
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(test_csv):
        print(f"Test dataset not found: {test_csv}")
        return

    # Load data directly from test_dataset.csv
    df_test = pd.read_csv(test_csv)
    test_dataset = CampusDataset(csv_file=test_csv, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Testing on {len(test_dataset)} samples from test_dataset.csv")
    
    # Load model based on model_type
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    
    if model_type == "convnext":
        from src.model.network import CampusLocator
        model = CampusLocator().to(device)
    elif model_type == "swin":
        from src.model.swin_network import CampusSwin
        model = CampusSwin().to(device)
    else:  # efficientnet (legacy)
        from src.model.network import CampusLocator
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
    
    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels, weights in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate errors
    errors = np.linalg.norm(all_preds - all_labels, axis=1)
    
    # Select worst
    sorted_idxs = np.argsort(errors)
    worst_idxs = sorted_idxs[-num_worst:][::-1]
    
    # Build simple output
    rows = []
    print(f"\nExporting {num_worst} worst predictions:")
    print("="*60)
    
    for i, idx in enumerate(worst_idxs):
        error_m = errors[idx]
        
        row_data = df_test.iloc[idx]
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
    parser.add_argument("--model", "-m", default="convnext_tiny_v1", help="Experiment name")
    parser.add_argument("--type", "-t", default="convnext", choices=["convnext", "swin", "efficientnet"],
                        help="Model type (convnext, swin, or efficientnet)")
    args = parser.parse_args()
    
    export_for_maps(num_worst=args.num, experiment_name=args.model, model_type=args.type)
