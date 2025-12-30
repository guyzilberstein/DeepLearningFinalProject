import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

def evaluate_model(experiment_name="default"):
    """
    Evaluate the model on the test set.
    Args:
        experiment_name: Name of the experiment to evaluate (matches checkpoint name)
    """
    # 1. Setup
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images_256')
    
    # Checkpoint path based on experiment name
    checkpoint_filename = f'best_{experiment_name}.pth' if experiment_name != "default" else 'best_campus_locator.pth'
    model_path = os.path.join(project_root, 'checkpoints', checkpoint_filename)
    indices_path = os.path.join(project_root, 'outputs', 'test_indices.npy')
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Run train.py first.")
        return
    if not os.path.exists(indices_path):
        print(f"Test indices {indices_path} not found. Run train.py first.")
        return

    # 2. Load Data
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    
    # Load the EXACT SAME test indices used during training
    test_indices = np.load(indices_path)
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"=== Evaluating Experiment: {experiment_name} ===")
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    
    # Handle both old format (state_dict only) and new format (dict with model_state_dict)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with best_loss: {checkpoint.get('best_loss', 'N/A')}")
        if 'experiment' in checkpoint:
            print(f"Experiment name in checkpoint: {checkpoint['experiment']}")
    else:
        # Old format: checkpoint IS the state_dict
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 4. Evaluation Loop
    total_error = 0.0
    errors = []
    
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Calculate Euclidean distance for each point in batch
            diff = outputs - labels
            batch_errors = torch.norm(diff, dim=1)
            
            total_error += batch_errors.sum().item()
            errors.extend(batch_errors.cpu().numpy())
            
    mean_error = total_error / len(test_dataset)
    median_error = np.median(errors)
    
    print("-" * 30)
    print(f"Test Set Mean Error:   {mean_error:.2f} meters")
    print(f"Test Set Median Error: {median_error:.2f} meters")
    print("-" * 30)
    
    return mean_error, median_error

if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        evaluate_model(experiment_name=sys.argv[1])
    else:
        evaluate_model()
