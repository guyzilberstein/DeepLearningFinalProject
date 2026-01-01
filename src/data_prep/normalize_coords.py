"""
Normalize GPS coordinates and generate dataset.csv from metadata_raw files.
Automatically applies all corrections from corrections_batch*.csv files.
"""
import pandas as pd
import numpy as np
import os
import glob
import json


def create_flattened_filename(row):
    """Convert raw photo path to flattened JPG filename."""
    full_path = row['path']  # e.g. data/raw_photos/Folder/Image.HEIC
    # Normalize to forward slashes just in case
    full_path = full_path.replace('\\', '/')
    
    # We expect data/raw_photos/ to be in the path
    token = 'data/raw_photos/'
    if token in full_path:
        rel_part = full_path.split(token)[1]
        # Replace / with _ and change extension to .jpg
        flat_name = rel_part.replace('/', '_')
        base_name = os.path.splitext(flat_name)[0]
        return base_name + ".jpg"
    else:
        # Fallback
        return os.path.splitext(os.path.basename(full_path))[0] + ".jpg"


def load_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load and merge all metadata CSV files."""
    csv_files = glob.glob(os.path.join(metadata_dir, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {metadata_dir}. Run extract_metadata.py first.")
    
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        temp_df['source_file'] = os.path.basename(file)
        
        if 'path' in temp_df.columns:
            temp_df['filename'] = temp_df.apply(create_flattened_filename, axis=1)
        elif 'filename' in temp_df.columns:
            temp_df['filename'] = temp_df['filename'].apply(lambda x: os.path.splitext(x)[0] + ".jpg")
        
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Shuffle data (reproducible)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Filter out samples with missing GPS coordinates
    initial_len = len(df)
    df = df.dropna(subset=['lat', 'lon'])
    print(f"Dropped {initial_len - len(df)} samples with missing GPS coordinates.")
    print(f"Total samples loaded: {len(df)}")
    
    return df


def apply_corrections(df: pd.DataFrame, corrections_dir: str, ref_lat: float, ref_lon: float, 
                      meters_per_lat: float, meters_per_lon: float) -> pd.DataFrame:
    """Apply all corrections from corrections_batch*.csv files."""
    correction_files = sorted(glob.glob(os.path.join(corrections_dir, 'corrections_batch*.csv')))
    
    if not correction_files:
        print("No correction files found.")
        return df
    
    print(f"\nApplying corrections from {len(correction_files)} file(s)...")
    total_updated = 0
    not_found = []
    
    for corrections_path in correction_files:
        corrections_df = pd.read_csv(corrections_path)
        print(f"  {os.path.basename(corrections_path)}: {len(corrections_df)} corrections")
        
        updated_count = 0
        for _, row in corrections_df.iterrows():
            # Extract folder and filename from path
            # e.g., "data/raw_photos/UnderBuilding26/IMG_2840.HEIC" -> "UnderBuilding26_IMG_2840.jpg"
            path_parts = row['path'].split('/')
            folder_name = path_parts[-2]
            original_filename = os.path.splitext(path_parts[-1])[0]
            filename_to_match = f"{folder_name}_{original_filename}.jpg"
            
            # Find the row in the main dataset
            match_idx = df[df['filename'] == filename_to_match].index
            if not match_idx.empty:
                idx = match_idx[0]
                df.loc[idx, 'lat'] = row['lat']
                df.loc[idx, 'lon'] = row['lon']
                df.loc[idx, 'gps_accuracy_m'] = 7.0  # Corrected samples get 7m accuracy
                
                # Recalculate x_meters and y_meters
                df.loc[idx, 'x_meters'] = (row['lon'] - ref_lon) * meters_per_lon
                df.loc[idx, 'y_meters'] = (row['lat'] - ref_lat) * meters_per_lat
                updated_count += 1
            else:
                not_found.append(filename_to_match)
        
        total_updated += updated_count
    
    if not_found:
        print(f"  Warning: {len(not_found)} files not found in dataset")
        for f in not_found[:5]:
            print(f"    - {f}")
        if len(not_found) > 5:
            print(f"    ... and {len(not_found) - 5} more")
    
    print(f"  Total updated: {total_updated} samples")
    return df




def normalize_and_save(project_root: str = None):
    """
    Main function to normalize coordinates and generate dataset files.
    
    Creates three output files:
    - dataset.csv: Training data (excludes TestPhotos and night holdout)
    - test_dataset.csv: TestPhotos only (external test set)
    - night_holdout.csv: 10% of night photos (for night-specific evaluation)
    """
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    
    metadata_dir = os.path.join(project_root, 'data', 'metadata_raw')
    data_dir = os.path.join(project_root, 'data')
    output_file = os.path.join(data_dir, 'dataset.csv')
    test_output_file = os.path.join(data_dir, 'test_dataset.csv')
    night_holdout_file = os.path.join(data_dir, 'night_holdout.csv')
    ref_file = os.path.join(data_dir, 'reference_coords.json')
    
    # 1. Load metadata
    df = load_metadata(metadata_dir)
    
    # 2. Calculate reference point (center of campus) - using ALL data for consistent coordinates
    ref_lat = df['lat'].mean()
    ref_lon = df['lon'].mean()
    print(f"\nReference Point (Mean): Lat={ref_lat:.6f}, Lon={ref_lon:.6f}")
    
    # Save reference point for inference
    ref_data = {'ref_lat': ref_lat, 'ref_lon': ref_lon}
    with open(ref_file, 'w') as f:
        json.dump(ref_data, f)
    print(f"Saved reference coordinates to {ref_file}")
    
    # 3. Calculate conversion constants
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    
    # 4. Convert to local coordinates
    df['x_meters'] = (df['lon'] - ref_lon) * METERS_PER_LON
    df['y_meters'] = (df['lat'] - ref_lat) * METERS_PER_LAT
    
    # 5. Apply corrections
    df = apply_corrections(df, data_dir, ref_lat, ref_lon, METERS_PER_LAT, METERS_PER_LON)
    
    # 6. Split into test set, night holdout, and training data
    print("\n--- Splitting Data ---")
    
    # TestPhotos = external test set
    is_test = df['source_file'] == 'TestPhotos.csv'
    test_df = df[is_test].copy()
    print(f"Test set (TestPhotos): {len(test_df)} samples")
    
    # Night photos for holdout (10%)
    night_sources = ['NightImagesLibraryArea.csv', 'nightWithout26AndLibrary.csv']
    is_night = df['source_file'].isin(night_sources)
    night_df = df[is_night & ~is_test].copy()
    
    # Randomly select 10% of night photos for holdout
    night_holdout_df = night_df.sample(frac=0.1, random_state=42)
    night_holdout_indices = night_holdout_df.index
    print(f"Night holdout (10%): {len(night_holdout_df)} samples (from {len(night_df)} night photos)")
    
    # Training data = everything except TestPhotos and night holdout
    train_df = df[~is_test & ~df.index.isin(night_holdout_indices)].copy()
    print(f"Training pool: {len(train_df)} samples")
    
    # 7. Save all datasets
    cols_to_keep = ['filename', 'lat', 'lon', 'x_meters', 'y_meters', 
                    'gps_accuracy_m', 'source_file']
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    train_df[cols_to_keep].to_csv(output_file, index=False)
    print(f"\nSaved training data to {output_file} ({len(train_df)} samples)")
    
    test_df[cols_to_keep].to_csv(test_output_file, index=False)
    print(f"Saved test data to {test_output_file} ({len(test_df)} samples)")
    
    night_holdout_df[cols_to_keep].to_csv(night_holdout_file, index=False)
    print(f"Saved night holdout to {night_holdout_file} ({len(night_holdout_df)} samples)")
    
    # Summary
    print("\n--- Summary ---")
    print(f"Training pool: {len(train_df)} (will be split 85/15 train/val)")
    print(f"Test set: {len(test_df)}")
    print(f"Night holdout: {len(night_holdout_df)}")
    print(f"Total: {len(train_df) + len(test_df) + len(night_holdout_df)}")


if __name__ == "__main__":
    normalize_and_save()
