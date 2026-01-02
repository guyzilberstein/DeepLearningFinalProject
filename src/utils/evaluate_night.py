"""
Evaluate model specifically on night holdout set.
This tests the model's performance on low-light conditions.
"""
import torch
from torch.utils.data import DataLoader
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


def evaluate_night(experiment_name="default"):
    """
    Evaluate the model on the night holdout set (night_holdout.csv).
    Args:
        experiment_name: Name of the experiment to evaluate (matches checkpoint name)
    """
    # 1. Setup - Use night_holdout.csv
    night_csv = os.path.join(project_root, 'data', 'night_holdout.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images_320')
    
    # Checkpoint path based on experiment name
    checkpoint_filename = f'best_{experiment_name}.pth' if experiment_name != "default" else 'best_campus_locator.pth'
    model_path = os.path.join(project_root, 'checkpoints', checkpoint_filename)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Run train.py first.")
        return
    if not os.path.exists(night_csv):
        print(f"Night holdout {night_csv} not found. Run normalize_coords.py first.")
        return

    # 2. Load Night Holdout Data
    night_dataset = CampusDataset(csv_file=night_csv, root_dir=img_dir, is_train=False)
    night_loader = DataLoader(night_dataset, batch_size=32, shuffle=False)
    
    print(f"=== Night Holdout Evaluation: {experiment_name} ===")
    print(f"Evaluating on {len(night_dataset)} night samples (from night_holdout.csv)")
    
    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with best_loss: {checkpoint.get('best_loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 4. Evaluation Loop
    total_error = 0.0
    errors = []
    
    with torch.no_grad():
        for inputs, labels, _ in night_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Calculate Euclidean distance for each point in batch
            diff = outputs - labels
            batch_errors = torch.norm(diff, dim=1)
            
            total_error += batch_errors.sum().item()
            errors.extend(batch_errors.cpu().numpy())
            
    mean_error = total_error / len(night_dataset)
    median_error = np.median(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print("-" * 40)
    print(f"Night Holdout Mean Error:   {mean_error:.2f} meters")
    print(f"Night Holdout Median Error: {median_error:.2f} meters")
    print(f"Night Holdout Min Error:    {min_error:.2f} meters")
    print(f"Night Holdout Max Error:    {max_error:.2f} meters")
    print("-" * 40)
    
    # Distribution analysis
    under_10m = sum(1 for e in errors if e < 10)
    under_20m = sum(1 for e in errors if e < 20)
    over_30m = sum(1 for e in errors if e > 30)
    
    print(f"Under 10m: {under_10m}/{len(errors)} ({100*under_10m/len(errors):.1f}%)")
    print(f"Under 20m: {under_20m}/{len(errors)} ({100*under_20m/len(errors):.1f}%)")
    print(f"Over 30m:  {over_30m}/{len(errors)} ({100*over_30m/len(errors):.1f}%)")
    
    return mean_error, median_error


if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_night(experiment_name=sys.argv[1])
    else:
        evaluate_night()

