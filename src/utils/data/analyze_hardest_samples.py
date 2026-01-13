"""
Identify the 'Hardest of the Hard' samples.
Finds the intersection of the worst predictions across multiple models.
"""
import torch
import numpy as np
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

def get_worst_indices(model_name, num=50):
    """Get indices and errors of worst predictions for a model."""
    print(f"Analyzing {model_name}...")
    
    # Paths
    csv_file = os.path.join(project_root, 'data', 'test_dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
    checkpoint_path = os.path.join(project_root, 'checkpoints', f'best_{model_name}.pth')
    
    # Load Data
    test_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CampusLocator().to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found for {model_name}")
        return {}
        
    model.eval()
    
    # Inference
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate errors
    errors = np.linalg.norm(all_preds - all_labels, axis=1)
    
    # Get worst
    sorted_idxs = np.argsort(errors)[::-1]  # Descending
    worst_idxs = sorted_idxs[:num]
    
    worst_files = {}
    for idx in worst_idxs:
        fname = test_dataset.data_frame.iloc[idx]['filename']
        worst_files[fname] = errors[idx]
        
    return worst_files

def analyze_intersection():
    models = ['convnext_tiny_v1', 'convnext_tiny_v2', 'convnext_tiny_v3']
    
    results = []
    for m in models:
        results.append(get_worst_indices(m, num=50))
        
    # Find intersection
    set1 = set(results[0].keys())
    set2 = set(results[1].keys())
    set3 = set(results[2].keys())
    
    intersection = set1.intersection(set2).intersection(set3)
    
    print(f"\n{'='*60}")
    print(f"HARDEST IMAGES (Failed by ALL 3 models)")
    print(f"Intersection of top 50 worst from v1, v2, v3")
    print(f"{'='*60}")
    print(f"Found {len(intersection)} images common to all 3 worst-lists:\n")
    
    # Sort by average error across models
    hardest_list = []
    for fname in intersection:
        err1 = results[0][fname]
        err2 = results[1][fname]
        err3 = results[2][fname]
        avg_err = (err1 + err2 + err3) / 3
        hardest_list.append((fname, avg_err, err1, err2, err3))
        
    # Sort by average error
    hardest_list.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Filename':<30} | {'Avg':<6} | {'v1':<6} | {'v2':<6} | {'v3':<6}")
    print("-" * 65)
    
    for item in hardest_list:
        fname, avg, e1, e2, e3 = item
        print(f"{fname:<30} | {avg:.1f}m  | {e1:.1f}m  | {e2:.1f}m  | {e3:.1f}m")

    # Save to CSV for inspection
    df = pd.DataFrame(hardest_list, columns=['filename', 'avg_error', 'v1_error', 'v2_error', 'v3_error'])
    df.to_csv('outputs/hardest_images_intersection.csv', index=False)
    print(f"\nSaved list to outputs/hardest_images_intersection.csv")

if __name__ == "__main__":
    analyze_intersection()

