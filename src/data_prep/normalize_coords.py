import pandas as pd
import numpy as np
import os
import glob
import json

# 1. Load your data
# Define the path to your CSV files
# Since this script is in src/data_prep/, we look in ../../data/metadata_raw/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
csv_path = os.path.join(project_root, 'data', 'metadata_raw')
csv_files = glob.glob(os.path.join(csv_path, '*.csv'))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_path}. Run extract_metadata.py first.")

print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")

# Read and concatenate all CSV files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    # Add a column for the source file if needed
    temp_df['source_file'] = os.path.basename(file)
    
    # Fix Filename Mismatch: Ensure extension is .jpg to match preprocess_images.py output
    # And handle the flattened directory structure (Folder_Image.jpg)
    def create_flattened_filename(row):
        full_path = row['path'] # e.g. data/raw_photos/Folder/Image.HEIC
        # Normalize to forward slashes just in case
        full_path = full_path.replace('\\', '/')
        
        # We expect data/raw_photos/ to be in the path. 
        # We want everything AFTER raw_photos/
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

    if 'path' in temp_df.columns:
        temp_df['filename'] = temp_df.apply(create_flattened_filename, axis=1)
    elif 'filename' in temp_df.columns:
         temp_df['filename'] = temp_df['filename'].apply(lambda x: os.path.splitext(x)[0] + ".jpg")
        
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# SHUFFLE DATA (Reproducible)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Filter out samples with missing GPS coordinates
initial_len = len(df)
df = df.dropna(subset=['lat', 'lon'])
print(f"Dropped {initial_len - len(df)} samples with missing GPS coordinates.")

print(f"Total samples loaded: {len(df)}")

# 2. Define the "Center" of your campus area (The Reference Point)
# Using the mean of the entire combined dataset
ref_lat = df['lat'].mean()
ref_lon = df['lon'].mean()

print(f"Reference Point (Mean): Lat={ref_lat:.6f}, Lon={ref_lon:.6f}")

# SAVE REFERENCE POINT for later inference
ref_data = {'ref_lat': ref_lat, 'ref_lon': ref_lon}
ref_file = os.path.join(project_root, 'data', 'reference_coords.json')
with open(ref_file, 'w') as f:
    json.dump(ref_data, f)
print(f"Saved reference coordinates to {ref_file}")

# 3. Conversion Constants (Approximate for small areas)
# 1 degree of latitude is ~111,132 meters
# 1 degree of longitude depends on the latitude (cos(lat) * 111,132)
METERS_PER_LAT = 111132.0
METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))

# 4. Calculate local coordinates (Meters from Center)
df['x_meters'] = (df['lon'] - ref_lon) * METERS_PER_LON
df['y_meters'] = (df['lat'] - ref_lat) * METERS_PER_LAT

# 4b. Calculate Z-Score for GPS Accuracy per Area
print("Calculating GPS Accuracy Z-Scores per Area...")
# Group by source_file (e.g., 'Building35Lower.csv')
stats = df.groupby('source_file')['gps_accuracy_m'].agg(['mean', 'std']).reset_index()
stats.rename(columns={'mean': 'acc_mean', 'std': 'acc_std'}, inplace=True)

df = df.merge(stats, on='source_file', how='left')

# Handle NaN std (single sample groups) -> set std to 1 (so z-score becomes 0)
df['acc_std'] = df['acc_std'].fillna(1.0)
df['acc_std'] = df['acc_std'].replace(0, 1.0) # Avoid division by zero if all values are same

df['gps_z_score'] = (df['gps_accuracy_m'] - df['acc_mean']) / df['acc_std']

# Clean up temporary columns
df.drop(columns=['acc_mean', 'acc_std'], inplace=True)

# 5. Save the prepared data
# Save in data/ folder
output_file = os.path.join(project_root, 'data', 'dataset.csv')

# Ensure we keep 'gps_accuracy_m' and 'gps_z_score'
cols_to_keep = ['filename', 'path', 'lat', 'lon', 'x_meters', 'y_meters', 'gps_accuracy_m', 'source_file', 'gps_z_score']
# Filter columns that actually exist
cols_to_keep = [c for c in cols_to_keep if c in df.columns]

df[cols_to_keep].to_csv(output_file, index=False)
print(f"Preprocessing complete. Data saved to {output_file}")
print(df[['filename', 'lat', 'lon', 'x_meters', 'y_meters']].head())
