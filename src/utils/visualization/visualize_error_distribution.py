"""
Visualize prediction error distribution as a histogram.
Shows percentile markers and key statistics.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from torch.utils.data import DataLoader
from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

# Output directory
OUTPUT_DIR = os.path.join(project_root, 'outputs', 'visualizations_final')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_ensemble_predictions(test_loader, device):
    """Load ensemble models and get predictions."""
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    model_names = ['convnext_tiny_v1', 'convnext_tiny_v2', 'convnext_tiny_v3']
    
    models = []
    for name in model_names:
        model_path = os.path.join(checkpoint_dir, f'best_{name}.pth')
        if os.path.exists(model_path):
            model = CampusLocator().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            models.append(model)
    
    if len(models) == 0:
        raise RuntimeError("No model checkpoints found!")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            batch_preds = []
            for model in models:
                outputs = model(inputs)
                batch_preds.append(outputs.cpu().numpy())
            
            ensemble_pred = np.mean(batch_preds, axis=0)
            all_preds.append(ensemble_pred)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def visualize_error_distribution():
    """Create error distribution histogram with percentile markers."""
    # Setup
    test_csv = os.path.join(project_root, 'data', 'test_dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    test_dataset = CampusDataset(csv_file=test_csv, root_dir=img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Running ensemble inference on {len(test_dataset)} test samples...")
    predictions, labels = get_ensemble_predictions(test_loader, device)
    
    # Calculate errors (Euclidean distance in meters)
    errors = np.linalg.norm(predictions - labels, axis=1)
    
    # Statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)
    
    # Percentile thresholds
    within_5m = np.sum(errors <= 5) / len(errors) * 100
    within_10m = np.sum(errors <= 10) / len(errors) * 100
    within_15m = np.sum(errors <= 15) / len(errors) * 100
    within_20m = np.sum(errors <= 20) / len(errors) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    # Histogram
    bins = np.arange(0, min(max(errors) + 5, 80), 2)
    n, bins_edges, patches = ax.hist(errors, bins=bins, color='#3498DB', 
                                      edgecolor='white', alpha=0.8, linewidth=0.5)
    
    # Color bars by error severity
    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center <= 10:
            patch.set_facecolor('#27AE60')  # Green - excellent
        elif bin_center <= 20:
            patch.set_facecolor('#F39C12')  # Orange - good
        else:
            patch.set_facecolor('#E74C3C')  # Red - poor
    
    # Add vertical lines for key statistics
    ax.axvline(mean_error, color='#2C3E50', linestyle='--', linewidth=2)
    ax.axvline(median_error, color='#8E44AD', linestyle='--', linewidth=2)
    ax.axvline(p90, color='#E74C3C', linestyle=':', linewidth=2)
    
    # Labels and title
    ax.set_xlabel('Prediction Error (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    
    # Use suptitle for main title with more control over positioning
    fig.suptitle('Prediction Error Distribution\nConvNeXt-Tiny Ensemble (3 Models)', 
                 fontsize=14, fontweight='bold', color='#2C3E50', y=0.98)
    
    # Stats text below the title
    stats_text = f'Mean: {mean_error:.2f}m  |  Median: {median_error:.2f}m  |  90th%: {p90:.1f}m'
    fig.text(0.5, 0.90, stats_text, fontsize=11, ha='center', color='#555555')
    
    # Add percentile text box - centered in the right portion of the plot
    textstr = (f'Samples within threshold:\n'
               f'  ≤5m:  {within_5m:.1f}%\n'
               f'  ≤10m: {within_10m:.1f}%\n'
               f'  ≤15m: {within_15m:.1f}%\n'
               f'  ≤20m: {within_20m:.1f}%')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7', alpha=0.9)
    ax.text(0.78, 0.55, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center', bbox=props,
            family='monospace')
    
    # Add color legend at the bottom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', label='≤10m (Excellent)'),
        Patch(facecolor='#F39C12', label='10-20m (Good)'),
        Patch(facecolor='#E74C3C', label='>20m (Poor)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=3, 
              fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    
    ax.set_xlim(0, min(max(errors) + 5, 60))
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'error_distribution.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    plt.close()
    
    return errors, mean_error, median_error


if __name__ == "__main__":
    visualize_error_distribution()
