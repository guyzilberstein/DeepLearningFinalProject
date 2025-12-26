import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import folium
import pandas as pd
from Preprocessing.img_to_tensor import CampusDataset
from model import CampusLocator

def denormalize_coordinates(x_meters, y_meters, ref_lat, ref_lon):
    """
    Convert (x, y) meters back to (lat, lon).
    Using the same constants as coordinates_normalizer.py
    """
    METERS_PER_LAT = 111132.0
    METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))
    
    lat = ref_lat + (y_meters / METERS_PER_LAT)
    lon = ref_lon + (x_meters / METERS_PER_LON)
    return lat, lon

def visualize_on_map():
    # 1. Setup
    csv_file = 'Preprocessing/processed_data.csv'
    img_dir = 'ProcessedImages'
    model_path = 'best_campus_locator.pth'
    output_map = 'campus_map_visualization.html'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Run train.py first.")
        return

    # 2. Get Reference Point
    df = pd.read_csv(csv_file)
    ref_lat = df['lat'].mean()
    ref_lon = df['lon'].mean()
    print(f"Reference Point: {ref_lat}, {ref_lon}")

    # 3. Load Data & Select Examples
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    print(f"Selecting 15 random examples...")
    indices = np.random.choice(len(full_dataset), 15, replace=False)
    
    # 4. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CampusLocator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 5. Initialize Map with Satellite Imagery
    # Esri WorldImagery allows for much higher zoom levels than standard OSM
    m = folium.Map(
        location=[ref_lat, ref_lon],
        zoom_start=19,
        max_zoom=21,  # Allow zooming in very close
        tiles=None    # We will add custom tiles below
    )
    
    # Add Esri Satellite Layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite (Esri)',
        overlay=False,
        control=True
    ).add_to(m)

    # Add OpenStreetMap as an option (for reference)
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    print("Running inference and plotting...")
    
    for i, idx in enumerate(indices):
        image_tensor, labels, _ = full_dataset[idx]
        inputs = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            
        pred_x, pred_y = outputs.cpu().numpy()[0]
        true_x, true_y = labels.numpy()
        
        pred_lat, pred_lon = denormalize_coordinates(pred_x, pred_y, ref_lat, ref_lon)
        true_lat, true_lon = denormalize_coordinates(true_x, true_y, ref_lat, ref_lon)
        
        error_m = np.linalg.norm([pred_x - true_x, pred_y - true_y])
        
        # Color coding error: Green (<3m), Orange (<10m), Red (>10m)
        color = 'green' if error_m < 3 else 'orange' if error_m < 10 else 'red'
        
        fg = folium.FeatureGroup(name=f"Ex {i+1} (Err: {error_m:.1f}m)")
        
        # True Location (Circle Marker - better for zooming)
        folium.CircleMarker(
            location=[true_lat, true_lon],
            radius=6,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=f"<b>Actual</b><br>File: {full_dataset.data_frame.iloc[idx]['filename']}"
        ).add_to(fg)
        
        # Predicted Location (X Marker via HTML div icon or just a different color Circle)
        # We'll use a Red Circle with a dot
        folium.CircleMarker(
            location=[pred_lat, pred_lon],
            radius=6,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=f"<b>Predicted</b><br>Error: {error_m:.2f}m"
        ).add_to(fg)
        
        # Line
        folium.PolyLine(
            locations=[[true_lat, true_lon], [pred_lat, pred_lon]],
            color=color,
            weight=3,
            opacity=0.8
        ).add_to(fg)
        
        fg.add_to(m)

    folium.LayerControl().add_to(m)
    
    m.save(output_map)
    print(f"Map saved to {output_map}")

if __name__ == "__main__":
    visualize_on_map()
