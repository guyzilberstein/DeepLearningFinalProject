"""
Extract photos with high prediction error from test set and move to training.
These represent "blind spots" in the model that should be trained on.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import shutil

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator


def extract_problematic_photos(experiment_name="b0_256_v3", error_threshold=20.0, batch_name="ProblematicPhotos2"):
    """
    Find test photos with error > threshold and move them to training data.
    
    Args:
        experiment_name: Name of the model checkpoint to use
        error_threshold: Move photos with error > this value (meters)
        batch_name: Name for the new batch (e.g., ProblematicPhotos2)
    """
    # Paths
    test_csv = os.path.join(project_root, 'data', 'test_dataset.csv')
    train_csv = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
    metadata_dir = os.path.join(project_root, 'data', 'metadata_raw')
    
    checkpoint_filename = f'best_{experiment_name}.pth'
    model_path = os.path.join(project_root, 'checkpoints', checkpoint_filename)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataset
    test_dataset = CampusDataset(csv_file=test_csv, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Run inference and collect errors
    all_errors = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            outputs = model(inputs.to(device))
            errors = torch.norm(outputs.cpu() - labels, dim=1).numpy()
            all_errors.extend(errors)
    
    all_errors = np.array(all_errors)
    
    # Identify problematic photos (error > threshold)
    problematic_mask = all_errors > error_threshold
    problematic_indices = np.where(problematic_mask)[0]
    
    print(f"\nFound {len(problematic_indices)} photos with error > {error_threshold}m")
    
    # Load the CSVs
    test_df = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv)
    
    # Load original TestPhotos metadata to get datetime
    test_meta_path = os.path.join(metadata_dir, 'TestPhotos.csv')
    test_meta = pd.read_csv(test_meta_path) if os.path.exists(test_meta_path) else None
    
    # Get problematic rows
    problematic_df = test_df.iloc[problematic_indices].copy()
    
    # Update source_file to indicate these are now training photos from problematic set
    problematic_df['source_file'] = f'{batch_name}.csv'
    
    # Rename the files: TestPhotos_* -> {batch_name}_*
    print(f"\nRenaming files to {batch_name}_*...")
    renamed_filenames = []
    metadata_rows = []
    
    for idx, row in problematic_df.iterrows():
        old_filename = row['filename']
        new_filename = old_filename.replace('TestPhotos_', f'{batch_name}_')
        renamed_filenames.append(new_filename)
        
        old_path = os.path.join(img_dir, old_filename)
        new_path = os.path.join(img_dir, new_filename)
        
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
        
        # Build metadata row with correct format: filename,path,datetime,lat,lon,gps_accuracy_m
        orig_heic_name = old_filename.replace('TestPhotos_', '').replace('.jpg', '.HEIC')
        datetime_val = ''
        if test_meta is not None:
            orig_row = test_meta[test_meta['filename'] == orig_heic_name]
            if len(orig_row) > 0:
                datetime_val = orig_row.iloc[0]['datetime']
        
        metadata_rows.append({
            'filename': orig_heic_name,
            'path': f'data/raw_photos/{batch_name}/{orig_heic_name}',
            'datetime': datetime_val,
            'lat': row['lat'],
            'lon': row['lon'],
            'gps_accuracy_m': row['gps_accuracy_m']
        })
    
    problematic_df['filename'] = renamed_filenames
    
    # Create metadata CSV with correct format (same as other metadata files)
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(metadata_dir, f'{batch_name}.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Created metadata: {metadata_path}")
    
    # Remove problematic photos from test set
    remaining_test_df = test_df.drop(test_df.index[problematic_indices])
    remaining_test_df.to_csv(test_csv, index=False)
    print(f"Updated test_dataset.csv: {len(remaining_test_df)} samples remaining")
    
    # Add problematic photos to training set
    updated_train_df = pd.concat([train_df, problematic_df], ignore_index=True)
    updated_train_df.to_csv(train_csv, index=False)
    print(f"Updated dataset.csv: {len(updated_train_df)} samples (added {len(problematic_df)})")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Moved {len(problematic_indices)} problematic photos to training")
    print(f"Test set: {len(test_df)} -> {len(remaining_test_df)} samples")
    print(f"Training pool: {len(train_df)} -> {len(updated_train_df)} samples")
    print(f"Files renamed: TestPhotos_* -> {batch_name}_*")
    print("="*50)
    
    # Show error distribution of moved photos
    moved_errors = all_errors[problematic_mask]
    print(f"\nMoved photos error distribution:")
    print(f"  {error_threshold}-30m:  {sum((moved_errors > error_threshold) & (moved_errors <= 30))}")
    print(f"  30-50m:  {sum((moved_errors > 30) & (moved_errors <= 50))}")
    print(f"  50-100m: {sum((moved_errors > 50) & (moved_errors <= 100))}")
    print(f"  >100m:   {sum(moved_errors > 100)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='b0_256_v3', help='Experiment name')
    parser.add_argument('--threshold', type=float, default=20.0, help='Error threshold in meters')
    parser.add_argument('--batch', default='ProblematicPhotos2', help='Batch name for the extracted photos')
    args = parser.parse_args()
    
    extract_problematic_photos(experiment_name=args.experiment, error_threshold=args.threshold, batch_name=args.batch)

