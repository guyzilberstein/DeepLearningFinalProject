import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from Preprocessing.img_to_tensor import CampusDataset
from model import CampusLocator

def visualize_results():
    # 1. Setup
    csv_file = 'Preprocessing/processed_data.csv'
    img_dir = 'ProcessedImages'
    model_path = 'best_campus_locator.pth'
    output_plot = 'results_visualization.png'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Run train.py first.")
        return

    # 2. Load Data
    # We visualize the entire dataset to see the map coverage
    dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running inference for visualization...")
    
    with torch.no_grad():
        for inputs, labels, weights in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 4. Plotting
    plt.figure(figsize=(12, 10))
    
    # Plot Reference Point (Center)
    plt.plot(0, 0, 'g+', markersize=15, markeredgewidth=3, label='Reference Point (0,0)')
    
    # Plot Actual vs Predicted
    # We use scatter with some transparency
    plt.scatter(all_labels[:, 0], all_labels[:, 1], c='blue', alpha=0.6, label='Actual GPS', s=50, edgecolors='k')
    plt.scatter(all_preds[:, 0], all_preds[:, 1], c='red', alpha=0.6, label='Predicted', s=50, edgecolors='k')
    
    # Draw error lines connecting Actual -> Predicted
    print("Drawing error vectors...")
    for i in range(len(all_labels)):
        start = all_labels[i]
        end = all_preds[i]
        plt.plot([start[0], end[0]], [start[1], end[1]], color='gray', alpha=0.3, linestyle='-', linewidth=1)
        
    plt.title(f"Campus Localization Results (N={len(dataset)})\nBlue=Truth, Red=Pred, Line=Error", fontsize=14)
    plt.xlabel("Meters East/West (X)", fontsize=12)
    plt.ylabel("Meters North/South (Y)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure aspect ratio is equal so 1 meter looks like 1 meter
    plt.axis('equal')
    
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    visualize_results()

