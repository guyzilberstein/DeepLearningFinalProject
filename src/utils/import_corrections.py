"""
Convert corrected coordinates from Google Maps export to corrections_batch format.

Usage:
1. After adjusting points in Google My Maps, export as CSV or note coordinates
2. Create a simple CSV with columns: name, latitude, longitude
3. Run: python import_corrections.py input.csv --output corrections_batch5.csv
"""
import pandas as pd
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))


def import_corrections(input_csv: str, output_name: str = "corrections_batch_new.csv"):
    """
    Convert simple coordinate CSV to corrections_batch format.
    
    Input format (from Google Maps):
        name,latitude,longitude
        UnderBuilding26_IMG_7146.jpg,31.2620788,34.8023207
    
    Output format (corrections_batch):
        WKT,filename,path,lat,lon,gps_accuracy_m,datetime,original_csv_file
    """
    # Load input
    df_input = pd.read_csv(input_csv)
    
    # Validate columns
    required = ['name', 'latitude', 'longitude']
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        # Try alternate column names
        alt_names = {'Name': 'name', 'Latitude': 'latitude', 'Longitude': 'longitude',
                     'lat': 'latitude', 'lon': 'longitude', 'filename': 'name'}
        for old, new in alt_names.items():
            if old in df_input.columns:
                df_input.rename(columns={old: new}, inplace=True)
        
        missing = [c for c in required if c not in df_input.columns]
        if missing:
            print(f"Error: Missing required columns: {missing}")
            print(f"Found columns: {list(df_input.columns)}")
            return
    
    # Load main dataset to get paths
    dataset_path = os.path.join(project_root, 'data', 'dataset.csv')
    df_main = pd.read_csv(dataset_path)
    
    # Build corrections
    corrections = []
    not_found = []
    
    for _, row in df_input.iterrows():
        filename = row['name']
        lat = row['latitude']
        lon = row['longitude']
        
        # Find original path in dataset
        match = df_main[df_main['filename'] == filename]
        if match.empty:
            not_found.append(filename)
            continue
        
        original_row = match.iloc[0]
        path = original_row['path']
        source_file = original_row.get('source_file', '')
        
        # Determine original_csv_file from source_file
        original_csv = f"data/metadata_raw/{source_file}" if source_file else ""
        
        corrections.append({
            'WKT': f'"POINT ({lon} {lat})"',
            'filename': os.path.basename(path),
            'path': path,
            'lat': lat,
            'lon': lon,
            'gps_accuracy_m': 7.0,
            'datetime': '',
            'original_csv_file': original_csv
        })
    
    if not_found:
        print(f"Warning: {len(not_found)} files not found in dataset:")
        for f in not_found[:10]:
            print(f"  - {f}")
    
    if not corrections:
        print("No corrections to save.")
        return
    
    # Save
    output_path = os.path.join(project_root, 'data', output_name)
    df_output = pd.DataFrame(corrections)
    df_output.to_csv(output_path, index=False)
    
    print(f"Created {output_path} with {len(corrections)} corrections")
    print("\nNext: Run normalize_coords.py to regenerate dataset.csv with these corrections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Google Maps export to corrections format")
    parser.add_argument("input", help="Input CSV with corrected coordinates")
    parser.add_argument("--output", "-o", default="corrections_batch_new.csv", 
                        help="Output filename (saved in data/ folder)")
    args = parser.parse_args()
    
    import_corrections(args.input, args.output)

