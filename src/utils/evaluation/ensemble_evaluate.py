"""
Ensemble evaluation: Load multiple models and average their predictions.
This reduces variance and typically improves accuracy by 5-10%.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

def load_model(model_path, device, model_name=""):
    """
    Load a single model from checkpoint.
    """
    model = CampusLocator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint.get('best_loss', 'N/A')
    else:
        model.load_state_dict(checkpoint)
        best_loss = 'N/A'
    
    model.eval()
    return model, best_loss


def ensemble_evaluate(model_names=None):
    """
    Evaluate an ensemble of models by averaging their predictions.
    
    Args:
        model_names: List of experiment names (e.g., ['b0_256_mlp_seed42', 'b0_256_mlp_seed100'])
                    If None, uses default ensemble
    """
    # Default ensemble if not specified
    if model_names is None:
        model_names = [
            'convnext_tiny_v1',
            'convnext_tiny_v2',
            'convnext_tiny_v3'
        ]
    
    # 1. Setup
    test_csv = os.path.join(project_root, 'data', 'test_dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    if not os.path.exists(test_csv):
        print(f"Test dataset {test_csv} not found. Run normalize_coords.py first.")
        return
    
    # 2. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    models = []
    valid_names = []
    for name in model_names:
        checkpoint_filename = f'best_{name}.pth' if name != "default" else 'best_campus_locator.pth'
        model_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_path} not found. Skipping.")
            continue
        
        try:
            print(f"Loading: {name}")
            model, best_loss = load_model(model_path, device, model_name=name)
            models.append(model)
            valid_names.append(name)
            print(f"  Success (val_loss: {best_loss})")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    
    if len(models) == 0:
        print("ERROR: No models loaded. Cannot evaluate ensemble.")
        return
    
    print(f"\nEnsemble size: {len(models)} models")
    print(f"{'='*60}\n")
    
    # 3. Load Test Data (external test set)
    test_dataset = CampusDataset(csv_file=test_csv, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # 4. Ensemble Inference
    all_ensemble_preds = []
    all_labels = []
    
    # Also track individual model predictions for comparison
    individual_preds = {i: [] for i in range(len(models))}
    
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            
            # Get predictions from each model
            batch_preds = []
            for i, model in enumerate(models):
                outputs = model(inputs)
                batch_preds.append(outputs.cpu().numpy())
                individual_preds[i].append(outputs.cpu().numpy())
            
            # Average predictions: (x1+x2+x3)/3, (y1+y2+y3)/3
            ensemble_pred = np.mean(batch_preds, axis=0)
            
            all_ensemble_preds.append(ensemble_pred)
            all_labels.append(labels.numpy())
    
    # Combine all batches
    all_ensemble_preds = np.vstack(all_ensemble_preds)
    all_labels = np.vstack(all_labels)
    
    # 5. Calculate Ensemble Errors
    ensemble_errors = np.linalg.norm(all_ensemble_preds - all_labels, axis=1)
    ensemble_mean = np.mean(ensemble_errors)
    ensemble_median = np.median(ensemble_errors)
    
    # 6. Calculate Individual Model Errors (for comparison)
    print("\n" + "-"*60)
    print("INDIVIDUAL MODEL RESULTS:")
    print("-"*60)
    
    for i in range(len(models)):
        model_preds = np.vstack(individual_preds[i])
        model_errors = np.linalg.norm(model_preds - all_labels, axis=1)
        model_mean = np.mean(model_errors)
        model_median = np.median(model_errors)
        print(f"  Model {i+1} ({valid_names[i]}): Mean={model_mean:.2f}m, Median={model_median:.2f}m")
    
    # 7. Print Ensemble Results
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS:")
    print("="*60)
    print(f"  Mean Error:   {ensemble_mean:.2f} meters")
    print(f"  Median Error: {ensemble_median:.2f} meters")
    print("="*60)
    
    # Calculate improvement
    if len(models) > 1:
        # Compare to best individual model
        best_individual_mean = min(
            np.mean(np.linalg.norm(np.vstack(individual_preds[i]) - all_labels, axis=1))
            for i in range(len(models))
        )
        improvement = best_individual_mean - ensemble_mean
        improvement_pct = (improvement / best_individual_mean) * 100
        print(f"\nImprovement over best individual model: {improvement:.2f}m ({improvement_pct:.1f}%)")
    
    return ensemble_mean, ensemble_median


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Use provided model names
        model_names = sys.argv[1:]
        ensemble_evaluate(model_names)
    else:
        # Use default ensemble
        ensemble_evaluate()
