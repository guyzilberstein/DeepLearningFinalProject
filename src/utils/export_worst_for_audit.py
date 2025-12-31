"""
Export worst predictions to CSV for manual GPS label audit.
Includes Google Maps links for easy verification.
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

def denormalize_to_latlon(x_meters, y_meters, ref_lat, ref_lon):
    """Convert local metric coordinates back to Lat/Lon"""
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    
    lat = ref_lat + (y_meters / METERS_PER_LAT)
    lon = ref_lon + (x_meters / METERS_PER_LON)
    return lat, lon

def google_maps_link(lat, lon):
    """Generate a Google Maps link for a coordinate"""
    return f"https://www.google.com/maps?q={lat},{lon}"

def export_worst_for_audit(num_worst=25):
    # 1. Setup
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images_256')
    model_path = os.path.join(project_root, 'checkpoints', 'best_b0_256_mlp.pth')
    indices_path = os.path.join(project_root, 'outputs', 'test_indices.npy')
    output_csv = os.path.join(project_root, 'outputs', 'worst_predictions_audit.csv')
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return
    if not os.path.exists(indices_path):
        print("Test indices not found.")
        return

    # Load Reference Point
    df_full = pd.read_csv(csv_file)
    ref_lat = df_full['lat'].mean()
    ref_lon = df_full['lon'].mean()

    # 2. Load Data
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    test_indices = np.load(indices_path)
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Run Inference
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
    
    # 4. Select Worst samples
    sorted_idxs = np.argsort(errors)
    worst_idxs = sorted_idxs[-num_worst:][::-1]  # Worst first
    
    # 5. Build audit data
    audit_rows = []
    
    print(f"\nExporting {num_worst} worst predictions for audit...")
    print("="*100)
    
    for i, idx_local in enumerate(worst_idxs):
        idx_global = all_indices[idx_local]
        pred_local = all_preds[idx_local]
        true_local = all_labels[idx_local]
        error_m = errors[idx_local]
        
        # Get original data
        row_data = df_full.iloc[idx_global]
        img_name = row_data['filename']
        source_file = row_data.get('source_file', 'Unknown')
        original_lat = row_data['lat']
        original_lon = row_data['lon']
        gps_accuracy = row_data.get('gps_accuracy_m', 'N/A')
        
        # Predicted coordinates
        p_lat, p_lon = denormalize_to_latlon(pred_local[0], pred_local[1], ref_lat, ref_lon)
        t_lat, t_lon = denormalize_to_latlon(true_local[0], true_local[1], ref_lat, ref_lon)
        
        audit_rows.append({
            'rank': i + 1,
            'filename': img_name,
            'source_file': source_file,
            'error_meters': round(error_m, 1),
            'gps_accuracy_m': gps_accuracy,
            'true_lat': round(t_lat, 6),
            'true_lon': round(t_lon, 6),
            'pred_lat': round(p_lat, 6),
            'pred_lon': round(p_lon, 6),
            'true_maps_link': google_maps_link(t_lat, t_lon),
            'pred_maps_link': google_maps_link(p_lat, p_lon),
            'dataset_row_index': idx_global,
            'label_correct': '',  # Empty column for manual annotation
            'corrected_lat': '',  # Empty column for corrections
            'corrected_lon': '',  # Empty column for corrections
            'notes': ''  # Empty column for notes
        })
        
        print(f"#{i+1:2d} | Error: {error_m:5.1f}m | {img_name}")
        print(f"     True: {google_maps_link(t_lat, t_lon)}")
        print(f"     Pred: {google_maps_link(p_lat, p_lon)}")
        print()
    
    # 6. Save to CSV
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_csv, index=False)
    
    print("="*100)
    print(f"\nSaved audit CSV to: {output_csv}")
    print("\nInstructions:")
    print("1. Open the CSV in Excel/Google Sheets")
    print("2. Click the 'true_maps_link' and 'pred_maps_link' to compare locations")
    print("3. If the PREDICTED location looks correct (model is right, label is wrong):")
    print("   - Set 'label_correct' to 'NO'")
    print("   - Copy pred_lat/pred_lon to corrected_lat/corrected_lon")
    print("4. If the TRUE location is correct, set 'label_correct' to 'YES'")
    print("5. Add any notes in the 'notes' column")

if __name__ == "__main__":
    export_worst_for_audit(25)

