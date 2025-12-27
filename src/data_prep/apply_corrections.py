"""
Script to apply GPS corrections from CSV files to the main dataset.csv
Sets gps_accuracy_m to 7 meters for all corrected samples.
"""

import pandas as pd
import os
import json
import numpy as np

def apply_gps_corrections(dataset_path, corrections_paths, ref_coords_path, default_accuracy=7.0):
    """
    Apply GPS corrections from one or more CSV files to the dataset.
    
    Args:
        dataset_path: Path to the main dataset.csv
        corrections_paths: List of paths to correction CSV files
        ref_coords_path: Path to reference_coords.json
        default_accuracy: GPS accuracy to set for corrected samples (default 7.0m)
    """
    # Backup original dataset (only if backup doesn't exist)
    backup_path = dataset_path.replace('.csv', '_before_corrections.csv')
    if not os.path.exists(backup_path):
        df_backup = pd.read_csv(dataset_path)
        df_backup.to_csv(backup_path, index=False)
        print(f"Created backup at {backup_path}")
    else:
        print(f"Backup already exists at {backup_path}")
    
    # Load main dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from dataset")
    
    # Load reference coordinates
    with open(ref_coords_path, 'r') as f:
        ref_data = json.load(f)
    ref_lat = ref_data['ref_lat']
    ref_lon = ref_data['ref_lon']
    
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    
    total_updated = 0
    not_found = []
    
    for corrections_path in corrections_paths:
        print(f"\nProcessing: {os.path.basename(corrections_path)}")
        corrections_df = pd.read_csv(corrections_path)
        print(f"  Found {len(corrections_df)} corrections")
        
        updated_count = 0
        for _, row in corrections_df.iterrows():
            # Extract folder and filename from path
            # e.g., "data/raw_photos/UnderBuilding26/IMG_2840.HEIC" -> "UnderBuilding26_IMG_2840.jpg"
            path_parts = row['path'].split('/')
            folder_name = path_parts[-2]  # e.g., "UnderBuilding26"
            original_filename = os.path.splitext(path_parts[-1])[0]  # e.g., "IMG_2840"
            filename_to_match = f"{folder_name}_{original_filename}.jpg"
            
            # Find the row in the main dataset
            match_idx = df[df['filename'] == filename_to_match].index
            if not match_idx.empty:
                idx = match_idx[0]
                df.loc[idx, 'lat'] = row['lat']
                df.loc[idx, 'lon'] = row['lon']
                df.loc[idx, 'gps_accuracy_m'] = default_accuracy
                
                # Recalculate x_meters and y_meters
                df.loc[idx, 'x_meters'] = (row['lon'] - ref_lon) * METERS_PER_LON
                df.loc[idx, 'y_meters'] = (row['lat'] - ref_lat) * METERS_PER_LAT
                updated_count += 1
            else:
                not_found.append(filename_to_match)
        
        print(f"  Updated {updated_count} samples")
        total_updated += updated_count
    
    if not_found:
        print(f"\nWarning: {len(not_found)} files not found in dataset:")
        for f in not_found[:10]:  # Show first 10
            print(f"  - {f}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")
    
    # Recalculate Z-scores per area
    print("\nRecalculating Z-scores per area...")
    stats = df.groupby('source_file')['gps_accuracy_m'].agg(['mean', 'std']).reset_index()
    stats.rename(columns={'mean': 'acc_mean', 'std': 'acc_std'}, inplace=True)
    df = df.drop(columns=['gps_z_score'], errors='ignore')
    df = df.merge(stats, on='source_file', how='left')
    df['acc_std'] = df['acc_std'].fillna(1.0).replace(0, 1.0)
    df['gps_z_score'] = (df['gps_accuracy_m'] - df['acc_mean']) / df['acc_std']
    df.drop(columns=['acc_mean', 'acc_std'], inplace=True)
    
    # Save updated dataset
    df.to_csv(dataset_path, index=False)
    print(f"\n{'='*50}")
    print(f"Total updated: {total_updated} samples")
    print(f"GPS accuracy set to {default_accuracy}m for all corrected samples")
    print(f"Saved updated dataset to {dataset_path}")
    print(f"Original data preserved at {backup_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    dataset_csv = os.path.join(project_root, 'data', 'dataset.csv')
    ref_coords_json = os.path.join(project_root, 'data', 'reference_coords.json')
    
    # List of correction files to apply
    corrections_files = [
        os.path.join(project_root, 'data', 'corrections_batch1.csv'),
        os.path.join(project_root, 'data', 'corrections_batch2.csv'),
        os.path.join(project_root, 'data', 'corrections_batch3.csv'),
    ]
    
    # Filter to only existing files
    existing_corrections = [f for f in corrections_files if os.path.exists(f)]
    
    if not existing_corrections:
        print("No correction files found!")
        print("Expected files:")
        for f in corrections_files:
            print(f"  - {f}")
    else:
        print(f"Found {len(existing_corrections)} correction file(s)")
        apply_gps_corrections(dataset_csv, existing_corrections, ref_coords_json)


