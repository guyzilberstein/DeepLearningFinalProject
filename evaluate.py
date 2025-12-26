import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from Preprocessing.img_to_tensor import CampusDataset
from model import CampusLocator

def evaluate_model():
    # 1. Setup
    csv_file = 'Preprocessing/processed_data.csv'
    img_dir = 'ProcessedImages'
    model_path = 'best_campus_locator.pth'
    indices_path = 'test_indices.npy'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Run train.py first.")
        return
        
    if not os.path.exists(indices_path):
        print(f"Test indices {indices_path} not found. Run train.py first to generate split.")
        return

    # 2. Load Data and Select Test Set
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    test_indices = np.load(indices_path)
    
    # Create the test subset using the saved indices
    test_dataset = Subset(full_dataset, test_indices)
    
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} UNSEEN test images...")
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    distances = []
    
    print("\nRunning Evaluation on Test Set...")
    print(f"{'Actual (x, y)':<25} | {'Predicted (x, y)':<25} | {'Error (m)':<10} | {'GPS Acc (m)':<10}")
    print("-" * 80)
    
    with torch.no_grad():
        for i, (inputs, labels, weights) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) # Shape: [1, 2]
            
            # weights is 1/sigma^2. recover sigma for display
            weight_val = weights.item()
            gps_acc = np.sqrt(1.0 / weight_val) if weight_val > 0 else 0
            
            outputs = model(inputs)    # Shape: [1, 2]
            
            # Move back to CPU for numpy math
            pred = outputs.cpu().numpy()[0]
            actual = labels.cpu().numpy()[0]
            
            # Euclidean distance
            dist = np.linalg.norm(pred - actual)
            distances.append(dist)
            
            # Print first 10 results detailed
            if i < 15:
                print(f"({actual[0]:6.1f}, {actual[1]:6.1f})     | ({pred[0]:6.1f}, {pred[1]:6.1f})     | {dist:6.2f}     | {gps_acc:6.1f}")
    
    mean_error = np.mean(distances)
    median_error = np.median(distances)
    max_error = np.max(distances)
    
    print("-" * 80)
    print(f"\nFinal Results on {len(test_dataset)} Test images:")
    print(f"Mean Error:   {mean_error:.2f} meters")
    print(f"Median Error: {median_error:.2f} meters")
    print(f"Max Error:    {max_error:.2f} meters")

if __name__ == "__main__":
    evaluate_model()
