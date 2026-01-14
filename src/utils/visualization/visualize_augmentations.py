"""
Visualize data augmentations applied during training.
Shows original image alongside each individual augmentation effect.
Designed for report/presentation quality output.
"""
import os
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as transforms
from PIL import Image
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

# Match dataset.py exactly
INPUT_SIZE = 320

# Base transform (resize only)
base_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])

# Individual augmentation transforms (for demonstration)
AUGMENTATIONS = {
    "Original": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ]),
    "Rotation (±5°)": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
    ]),
    "Perspective": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),  # p=1 to always show effect
        transforms.ToTensor(),
    ]),
    "Color Jitter": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
    ]),
    "Night Simulation": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ColorJitter(brightness=(0.4, 0.7), contrast=(0.6, 0.9), saturation=0.2),
        transforms.ToTensor(),
    ]),
    "Random Erasing": transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.15)),  # p=1 to always show effect
    ]),
}

# Color palette for augmentation categories
COLORS = {
    "Original": "#2C3E50",      # Dark blue-gray
    "Rotation (±5°)": "#E74C3C",       # Red
    "Perspective": "#9B59B6",   # Purple
    "Color Jitter": "#F39C12",  # Orange
    "Night Simulation": "#1ABC9C",  # Teal
    "Random Erasing": "#3498DB",  # Blue
}


def tensor_to_image(tensor):
    """Convert tensor back to displayable image (reverse ToTensor)."""
    return tensor.permute(1, 2, 0).numpy().clip(0, 1)


def visualize_augmentations(num_images: int = 3, output_name: str = "augmentation_visualization"):
    """
    Create a professional visualization showing each augmentation effect.
    
    Args:
        num_images: Number of different sample images to show (rows)
        output_name: Name for the output file (without extension)
    """
    img_dir = os.path.join(project_root, 'data', 'images')
    
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return
    
    # Get random images
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if len(all_images) == 0:
        print("No images found!")
        return
    
    # Select diverse images (try to get different areas)
    random.seed(42)  # Reproducible selection
    selected = random.sample(all_images, min(num_images, len(all_images)))
    
    aug_names = list(AUGMENTATIONS.keys())
    num_augs = len(aug_names)
    
    # Create figure with custom styling
    fig, axes = plt.subplots(
        num_images, num_augs,
        figsize=(2.8 * num_augs, 3.2 * num_images),
        facecolor='white'
    )
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Process each image
    for row, img_name in enumerate(selected):
        img_path = os.path.join(img_dir, img_name)
        original = Image.open(img_path).convert('RGB')
        
        for col, aug_name in enumerate(aug_names):
            transform = AUGMENTATIONS[aug_name]
            aug_tensor = transform(original)
            
            ax = axes[row, col]
            ax.imshow(tensor_to_image(aug_tensor))
            ax.axis('off')
            
            # Add colored border based on augmentation type
            color = COLORS[aug_name]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)
            
            # Add title only to first row
            if row == 0:
                ax.set_title(aug_name, fontsize=11, fontweight='bold', 
                           color=color, pad=8)
            
            # Add image label on left side
            if col == 0:
                # Shorten filename for display
                short_name = img_name.split('_')[0][:12]
                ax.text(-0.15, 0.5, f"Sample {row + 1}", 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='center', rotation=90,
                       fontweight='bold', color='#555555')
    
    # Main title
    fig.suptitle(
        "Training Data Augmentation Pipeline",
        fontsize=16, fontweight='bold', y=0.98, color='#2C3E50'
    )
    
    # Subtitle with details
    subtitle = "Each column shows the isolated effect of a single augmentation technique used during training."
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=10, 
             color='#666666', style='italic')
    
    # Create legend with augmentation descriptions
    legend_text = [
        ("Rotation (±5°)", "Simulates slight phone angle variation"),
        ("Perspective", "Simulates phone tilt (distortion=0.2, p=0.5)"),
        ("Color Jitter", "Time-of-day invariance (B/C/S=0.3, H=0.1)"),
        ("Night Simulation", "Low-light robustness (brightness=0.4-0.7, p=0.2)"),
        ("Random Erasing", "Occlusion simulation (scale=0.02-0.15, p=0.2)"),
    ]
    
    # Add legend at bottom
    legend_y = 0.02
    legend_x_start = 0.08
    spacing = 0.18
    
    for i, (name, desc) in enumerate(legend_text):
        x_pos = legend_x_start + (i * spacing)
        color = COLORS[name]
        
        # Colored square
        fig.patches.append(mpatches.FancyBboxPatch(
            (x_pos - 0.015, legend_y + 0.005), 0.012, 0.015,
            boxstyle="round,pad=0.002",
            facecolor=color, edgecolor='none',
            transform=fig.transFigure, figure=fig
        ))
        
        # Text
        fig.text(x_pos, legend_y + 0.012, name, fontsize=8, fontweight='bold',
                color=color, va='center')
        fig.text(x_pos, legend_y - 0.005, desc, fontsize=7,
                color='#888888', va='center')
    
    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.91])
    
    # Save
    output_path = os.path.join(project_root, 'outputs', f'{output_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Saved to: {output_path}")
    plt.show()


def visualize_augmentations_compact(num_images: int = 3, output_name: str = "augmentation_compact"):
    """
    Create a more compact visualization (fewer columns) for presentations.
    Shows: Original | Geometric (Rotation+Perspective) | Photometric (Color+Night) | Combined
    """
    img_dir = os.path.join(project_root, 'data', 'images')
    
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return
    
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if len(all_images) == 0:
        print("No images found!")
        return
    
    random.seed(42)
    selected = random.sample(all_images, min(num_images, len(all_images)))
    
    # Compact augmentation groups
    compact_augs = {
        "Original": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]),
        "Geometric\n(Rotation + Perspective)": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
            transforms.ToTensor(),
        ]),
        "Photometric\n(Color Jitter)": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
        ]),
        "Night\nSimulation": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ColorJitter(brightness=(0.4, 0.7), contrast=(0.6, 0.9), saturation=0.2),
            transforms.ToTensor(),
        ]),
        "Occlusion\n(Random Erasing)": transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.15)),
        ]),
    }
    
    compact_colors = {
        "Original": "#2C3E50",
        "Geometric\n(Rotation + Perspective)": "#E74C3C",
        "Photometric\n(Color Jitter)": "#F39C12",
        "Night\nSimulation": "#1ABC9C",
        "Occlusion\n(Random Erasing)": "#3498DB",
    }
    
    aug_names = list(compact_augs.keys())
    num_augs = len(aug_names)
    
    fig, axes = plt.subplots(
        num_images, num_augs,
        figsize=(3.5 * num_augs, 3.5 * num_images),
        facecolor='white'
    )
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for row, img_name in enumerate(selected):
        img_path = os.path.join(img_dir, img_name)
        original = Image.open(img_path).convert('RGB')
        
        for col, aug_name in enumerate(aug_names):
            transform = compact_augs[aug_name]
            aug_tensor = transform(original)
            
            ax = axes[row, col]
            ax.imshow(tensor_to_image(aug_tensor))
            ax.axis('off')
            
            color = compact_colors[aug_name]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(4)
            
            if row == 0:
                ax.set_title(aug_name, fontsize=13, fontweight='bold',
                           color=color, pad=12)
    
    fig.suptitle(
        "Data Augmentation Pipeline",
        fontsize=18, fontweight='bold', y=0.97, color='#2C3E50'
    )
    
    subtitle = "Augmentations improve model robustness to lighting, viewing angle, and occlusions"
    fig.text(0.5, 0.92, subtitle, ha='center', fontsize=11, color='#666666', style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.89])
    
    output_path = os.path.join(project_root, 'outputs', f'{output_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize training data augmentations")
    parser.add_argument("--images", "-i", type=int, default=3, 
                        help="Number of sample images to show")
    parser.add_argument("--compact", "-c", action="store_true",
                        help="Use compact layout (grouped augmentations)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (without extension)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for image selection")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.compact:
        output_name = args.output or "augmentation_compact"
        visualize_augmentations_compact(num_images=args.images, output_name=output_name)
    else:
        output_name = args.output or "augmentation_visualization"
        visualize_augmentations(num_images=args.images, output_name=output_name)
