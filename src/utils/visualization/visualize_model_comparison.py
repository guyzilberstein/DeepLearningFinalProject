"""
Visualize model comparison between different architectures.
Bar chart comparing EfficientNet-B0 vs ConvNeXt-Tiny performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

OUTPUT_DIR = os.path.join(project_root, 'outputs', 'visualizations_updated')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_model_comparison():
    """Create bar chart comparing model architectures."""
    # Data from project_journal.md
    models = ['EfficientNet-B0\nEnsemble', 'ConvNeXt-Tiny\nEnsemble']
    
    # Test set performance
    test_mean = [8.17, 7.16]
    test_median = [7.33, 6.82]
    
    # Night performance
    night_mean = [9.39, 7.43]
    night_median = [7.32, 6.66]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    # Colors
    colors = {
        'mean': '#3498DB',
        'median': '#9B59B6'
    }
    
    # Test Set Performance
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, test_mean, width, label='Mean Error', color=colors['mean'], edgecolor='white')
    bars2 = ax1.bar(x + width/2, test_median, width, label='Median Error', color=colors['median'], edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Error (meters)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Set Performance', fontsize=13, fontweight='bold', color='#2C3E50')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, max(test_mean) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Add improvement annotation
    improvement = ((test_mean[0] - test_mean[1]) / test_mean[0]) * 100
    ax1.annotate(f'↓ {improvement:.1f}%', xy=(1, test_mean[1]), 
                xytext=(1.3, test_mean[0]),
                fontsize=12, color='#27AE60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    
    # Night Performance
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, night_mean, width, label='Mean Error', color=colors['mean'], edgecolor='white')
    bars4 = ax2.bar(x + width/2, night_median, width, label='Median Error', color=colors['median'], edgecolor='white')
    
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Error (meters)', fontsize=12, fontweight='bold')
    ax2.set_title('Night Holdout Performance', fontsize=13, fontweight='bold', color='#2C3E50')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, max(night_mean) * 1.3)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add improvement annotation for night
    night_improvement = ((night_mean[0] - night_mean[1]) / night_mean[0]) * 100
    ax2.annotate(f'↓ {night_improvement:.1f}%', xy=(1, night_mean[1]), 
                xytext=(1.3, night_mean[0]),
                fontsize=12, color='#27AE60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    
    fig.suptitle('Model Architecture Comparison\nEfficientNet-B0 vs ConvNeXt-Tiny (Both 320×320 input)',
                 fontsize=14, fontweight='bold', color='#2C3E50', y=1.02)
    
    # Add model info
    info_text = ('EfficientNet-B0: 5.3M params  |  ConvNeXt-Tiny: 28M params\n'
                 'ConvNeXt uses 7×7 kernels for larger receptive field')
    fig.text(0.5, -0.02, info_text, ha='center', fontsize=10, color='#666666', style='italic')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_model_comparison()
