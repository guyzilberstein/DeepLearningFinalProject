"""
Visualize data augmentations applied during training.
Shows original image alongside multiple augmented versions.
"""
import os
import sys
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

# Same augmentation pipeline as dataset.py (but without normalization for visualization)
INPUT_SIZE = 256

# Current production augmentations
augmentation_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=(0.1, 0.4), contrast=(0.1, 0.4), saturation=0.1, hue=0.01),
    ], p=0.25),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

# CURRENT: Rotation + gentle perspective, color variety
experimental_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    
    # Slight rotation (±5°) for angle variation
    transforms.RandomRotation(degrees=5),
    
    # Gentle perspective (0.2 instead of 0.3) - simulates phone tilt
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    
    # Color variety (brightness, contrast, saturation, hue for time-of-day)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    
    # Gentler night simulation: 0.4-0.7 brightness (40-70% of original)
    transforms.RandomApply([
        transforms.ColorJitter(brightness=(0.4, 0.7), contrast=(0.6, 0.9), saturation=0.2),
    ], p=0.2),
    
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

no_augmentation = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])


def tensor_to_image(tensor):
    """Convert tensor back to displayable image (reverse ToTensor)."""
    return tensor.permute(1, 2, 0).numpy()


def visualize_augmentations(num_images: int = 3, num_augmentations: int = 5, experimental: bool = False):
    """
    Show original images alongside augmented versions.
    
    Args:
        num_images: Number of different images to show
        num_augmentations: Number of augmented versions per image
        experimental: Use experimental transforms (gentler night, more color)
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
    
    selected = random.sample(all_images, min(num_images, len(all_images)))
    
    # Choose transform pipeline
    transforms_to_use = experimental_transforms if experimental else augmentation_transforms
    mode_label = "EXPERIMENTAL" if experimental else "CURRENT"
    
    # Create figure
    fig, axes = plt.subplots(num_images, num_augmentations + 1, 
                             figsize=(2.5 * (num_augmentations + 1), 2.5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for row, img_name in enumerate(selected):
        img_path = os.path.join(img_dir, img_name)
        original = Image.open(img_path).convert('RGB')
        
        # Show original
        orig_tensor = no_augmentation(original)
        axes[row, 0].imshow(tensor_to_image(orig_tensor))
        axes[row, 0].set_title("Original", fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Show augmented versions
        for col in range(1, num_augmentations + 1):
            aug_tensor = transforms_to_use(original)
            axes[row, col].imshow(tensor_to_image(aug_tensor).clip(0, 1))
            axes[row, col].set_title(f"Aug #{col}", fontsize=9)
            axes[row, col].axis('off')
    
    if experimental:
        title = "EXPERIMENTAL Augmentations\n(Gentler night: 0.3-0.6 brightness, More color: hue=0.1, sat=0.4)"
    else:
        title = "CURRENT Augmentations\n(Perspective, ColorJitter, Night Simulation, RandomErasing)"
    
    plt.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    # Save and show
    suffix = "_experimental" if experimental else ""
    output_path = os.path.join(project_root, 'outputs', f'augmentation_examples{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize data augmentations")
    parser.add_argument("--images", "-i", type=int, default=3, help="Number of images")
    parser.add_argument("--augs", "-a", type=int, default=5, help="Augmentations per image")
    parser.add_argument("--experimental", "-e", action="store_true", 
                        help="Use experimental transforms (gentler night, more color)")
    args = parser.parse_args()
    
    visualize_augmentations(num_images=args.images, num_augmentations=args.augs, 
                           experimental=args.experimental)

