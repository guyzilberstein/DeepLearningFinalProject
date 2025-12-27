import pandas as pd
import os
import json
import numpy as np

def update_dataset():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    original_csv_path = os.path.join(project_root, 'data', 'dataset.csv')
    corrections_csv_path = os.path.join(project_root, 'data', 'corrections_batch_2.csv')
    output_csv_path = os.path.join(project_root, 'data', 'dataset_corrected.csv')
    ref_coords_path = os.path.join(project_root, 'data', 'reference_coords.json')

    print(f"Loading original dataset: {original_csv_path}")
    df_main = pd.read_csv(original_csv_path)
    
    print(f"Loading corrections: {corrections_csv_path}")
    df_corr = pd.read_csv(corrections_csv_path)
    
    # Load reference coords
    with open(ref_coords_path, 'r') as f:
        ref_data = json.load(f)
        ref_lat = ref_data['ref_lat']
        ref_lon = ref_data['ref_lon']
        
    print(f"Reference Point: {ref_lat}, {ref_lon}")
    
    # Pre-process corrections to match main dataset filenames
    # Main dataset filenames are like: "UnderBuilding26_IMG_2857.jpg"
    # Corrections have "path": "data/raw_photos/UnderBuilding26/IMG_2857.HEIC"
    
    updates_count = 0
    
    for idx, row in df_corr.iterrows():
        path = row['path'] # e.g. data/raw_photos/UnderBuilding26/IMG_2857.HEIC
        
        # Convert path to expected filename in main dataset
        # 1. Strip 'data/raw_photos/'
        if 'data/raw_photos/' in path:
            rel_path = path.split('data/raw_photos/')[1]
        else:
            rel_path = path # Fallback
            
        # 2. Replace / with _ and change extension to .jpg
        flat_name = rel_path.replace('/', '_')
        target_filename = os.path.splitext(flat_name)[0] + ".jpg"
        
        # Find this filename in main dataframe
        mask = df_main['filename'] == target_filename
        
        if mask.any():
            # Update values
            # Use loc to update
            df_main.loc[mask, 'lat'] = row['lat']
            df_main.loc[mask, 'lon'] = row['lon']
            df_main.loc[mask, 'gps_accuracy_m'] = 7.0
            updates_count += 1
        else:
            print(f"Warning: Could not find {target_filename} in main dataset.")

    print(f"Updated {updates_count} rows.")
    
    # Recalculate x_meters and y_meters for ALL rows (to be safe, though only updated changed)
    # Actually only updated rows changed x/y relative to ref
    
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    
    df_main['x_meters'] = (df_main['lon'] - ref_lon) * METERS_PER_LON
    df_main['y_meters'] = (df_main['lat'] - ref_lat) * METERS_PER_LAT
    
    # Save
    df_main.to_csv(output_csv_path, index=False)
    print(f"Saved corrected dataset to {output_csv_path}")

if __name__ == "__main__":
    update_dataset()

