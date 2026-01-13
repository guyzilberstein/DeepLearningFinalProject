import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import os

# Input resolution 
INPUT_SIZE = 320

class CampusDataset(Dataset):
    def __init__(self, csv_file, root_dir, is_train=False):
        """
        Args:
            csv_file (string): Path to the csv file (e.g., 'dataset.csv')
            root_dir (string): Directory with all the actual images.
            is_train (bool): If True, apply data augmentation.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        if is_train:
            # Training Pipeline with Augmentation
            self.transform = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                
                # Slight rotation (±5°)
                transforms.RandomRotation(degrees=5),
                
                # Gentle perspective (simulates phone tilt, not location change)
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                
                # Photometric Augmentation (Lighting & color invariance), simulates different times of day
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                
                # Night/Low-light Simulation (20% chance)
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=(0.4, 0.7), contrast=(0.6, 0.9), saturation=0.2),
                ], p=0.2),
                
                transforms.ToTensor(),
                
                # Regularization (Simulates occlusions like trees, poles, people)
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
                
                # Standard ImageNet Normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test Pipeline (Deterministic)
            self.transform = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image filename and path
        img_name = self.data_frame.iloc[idx]['filename']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Open and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Get GPS labels (meters from center)
        label_x = self.data_frame.iloc[idx]['x_meters']
        label_y = self.data_frame.iloc[idx]['y_meters']
        labels = torch.tensor([label_x, label_y], dtype=torch.float32)
        
        # Return image and labels
        return image_tensor, labels
