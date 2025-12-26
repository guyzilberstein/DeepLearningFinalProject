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
    if 'filename' in temp_df.columns:
        temp_df['filename'] = temp_df['filename'].apply(lambda x: os.path.splitext(x)[0] + ".jpg")
        
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# SHUFFLE DATA (Reproducible)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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

# 5. Save the prepared data
# Save in data/ folder
output_file = os.path.join(project_root, 'data', 'dataset.csv')

# Ensure we keep 'gps_accuracy_m'
cols_to_keep = ['filename', 'path', 'lat', 'lon', 'x_meters', 'y_meters', 'gps_accuracy_m', 'source_file']
# Filter columns that actually exist
cols_to_keep = [c for c in cols_to_keep if c in df.columns]

df[cols_to_keep].to_csv(output_file, index=False)
print(f"Preprocessing complete. Data saved to {output_file}")
print(df[['filename', 'lat', 'lon', 'x_meters', 'y_meters']].head())
