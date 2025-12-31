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
    Main function to normalize coordinates and generate dataset.csv.
    
    1. Loads all metadata from metadata_raw/
    2. Calculates reference point (center of campus)
    3. Converts lat/lon to local x/y meters
    4. Applies all corrections from corrections_batch*.csv
    5. Recalculates Z-scores
    6. Saves to dataset.csv
    """
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    
    metadata_dir = os.path.join(project_root, 'data', 'metadata_raw')
    data_dir = os.path.join(project_root, 'data')
    output_file = os.path.join(data_dir, 'dataset.csv')
    ref_file = os.path.join(data_dir, 'reference_coords.json')
    
    # 1. Load metadata
    df = load_metadata(metadata_dir)
    
    # 2. Calculate reference point (center of campus)
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
    
    # 6. Save dataset
    cols_to_keep = ['filename', 'path', 'lat', 'lon', 'x_meters', 'y_meters', 
                    'gps_accuracy_m', 'source_file']
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    df[cols_to_keep].to_csv(output_file, index=False)
    print(f"\nPreprocessing complete. Data saved to {output_file}")
    print(f"Total samples: {len(df)}")
    print(df[['filename', 'lat', 'lon', 'x_meters', 'y_meters']].head())


if __name__ == "__main__":
    normalize_and_save()
