import pandas as pd
import numpy as np
import os
import glob

# 1. Load your data
# Define the path to your CSV files
# Since this script is in Preprocessing/, we look in ../PhotoExtraction/
csv_folder = '../PhotoExtraction'  
csv_path = os.path.join(os.path.dirname(__file__), csv_folder) # robust path handling
csv_files = glob.glob(os.path.join(csv_path, '*.csv'))

if not csv_files:
    # Fallback to current directory or manual check if running from different root
    print(f"No CSVs found in {csv_path}, trying absolute search or checking paths...")
    # Just in case the script is run from project root, try direct path
    csv_files = glob.glob('DeepLearningFinalProject/PhotoExtraction/*.csv')

if not csv_files:
    raise FileNotFoundError(f"No CSV files found. Checked {csv_path} and fallback paths.")

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
print(f"Total samples loaded: {len(df)}")

# 2. Define the "Center" of your campus area (The Reference Point)
# Using the mean of the entire combined dataset
ref_lat = df['lat'].mean()
ref_lon = df['lon'].mean()

print(f"Reference Point (Mean): Lat={ref_lat:.6f}, Lon={ref_lon:.6f}")

# 3. Conversion Constants (Approximate for small areas)
# 1 degree of latitude is ~111,132 meters
# 1 degree of longitude depends on the latitude (cos(lat) * 111,132)
METERS_PER_LAT = 111132.0
METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))

# 4. Calculate local coordinates (Meters from Center)
df['x_meters'] = (df['lon'] - ref_lon) * METERS_PER_LON
df['y_meters'] = (df['lat'] - ref_lat) * METERS_PER_LAT

# 5. Save the prepared data
# Save in the same folder as the script or a specific output folder
output_file = os.path.join(os.path.dirname(__file__), 'processed_data.csv')
df.to_csv(output_file, index=False)
print(f"Preprocessing complete. Data saved to {output_file}")
print(df[['filename', 'lat', 'lon', 'x_meters', 'y_meters']].head())
