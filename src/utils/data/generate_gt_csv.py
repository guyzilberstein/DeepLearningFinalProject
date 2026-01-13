import pandas as pd
import os
import sys

# Define paths robustly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

dataset_path = os.path.join(project_root, 'data', 'dataset.csv')
gt_path = os.path.join(project_root, 'data', 'gt.csv')

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} does not exist.")
    exit(1)

# Read dataset
df = pd.read_csv(dataset_path)

# Create gt dataframe with required columns
gt_df = pd.DataFrame()
gt_df['image_name'] = df['filename']
gt_df['Latitude'] = df['lat']
gt_df['Longitude'] = df['lon']

# Save to csv
gt_df.to_csv(gt_path, index=False)
print(f"Created {gt_path} with {len(gt_df)} rows.")
