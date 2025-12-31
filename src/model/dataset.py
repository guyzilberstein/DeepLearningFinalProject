import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import os

# Input resolution for EfficientNet-B0 (close to native 224x224)
INPUT_SIZE = 256

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
            # Training Pipeline with Advanced Augmentation
            self.transform = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                
                # Geometric Augmentation (Crucial for localization)
                # Simulates viewing buildings from different angles/tilts
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                
                # Photometric Augmentation (Lighting invariance)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                
                # Night Simulation (25% of images brutally darkened)
                # Addresses the 48% night image failure rate
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=(0.1, 0.4), contrast=(0.1, 0.4), saturation=0.1, hue=0.01),
                ], p=0.25),
                
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
        
        # Weight (kept for compatibility, not currently used)
        gps_accuracy = self.data_frame.iloc[idx].get('gps_accuracy_m', 10.0)
        if pd.isna(gps_accuracy):
            gps_accuracy = 10.0
        sigma = max(float(gps_accuracy), 1.0)
        weight = torch.tensor([1.0 / (sigma**2)], dtype=torch.float32)
        
        return image_tensor, labels, weight
