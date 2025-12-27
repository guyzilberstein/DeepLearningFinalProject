import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import os
import math

class CampusDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file (e.g., 'dataset.csv')
            root_dir (string): Directory with all the actual images.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # This pipeline prepares the image for the model
        self.transform = transforms.Compose([
            # 1. Resize every image to 224x224 (Standard for ResNet)
            transforms.Resize((224, 224)),
            
            # 2. Convert from [0, 255] pixels to [0.0, 1.0] Tensor
            transforms.ToTensor(),
            
            # 3. Normalize color channels (Standard math for pre-trained models)
            # These specific numbers are the Mean and Std of the ImageNet dataset
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # The model asks: "How many items do I have to learn?"
        return len(self.data_frame)

    def __getitem__(self, idx):
        # The model asks: "Give me item #42"
        
        # A. Find the filename in the CSV at row 'idx'
        img_name = self.data_frame.iloc[idx]['filename']
        img_path = os.path.join(self.root_dir, img_name)
        
        # B. Open the image file
        # .convert('RGB') ensures it has 3 channels (Red, Green, Blue)
        # Handle FileNotFoundError gracefully? Or let it crash so we know something is wrong?
        # Better to let it crash or filter dataset beforehand.
        image = Image.open(img_path).convert('RGB')
        
        # C. Apply the transforms (Resize -> Tensor -> Normalize)
        image_tensor = self.transform(image)
        
        # D. Get the labels (The "Right Answer")
        # We grab the meters_x and meters_y we calculated in Step 2
        label_x = self.data_frame.iloc[idx]['x_meters']
        label_y = self.data_frame.iloc[idx]['y_meters']
        
        # Get GPS accuracy for weighting
        # Default to a moderate value if missing (e.g. 10m)
        gps_accuracy = self.data_frame.iloc[idx].get('gps_accuracy_m', 10.0)
        if pd.isna(gps_accuracy):
            gps_accuracy = 10.0
            
        # Get Z-Score (default to 0 if missing)
        z_score = self.data_frame.iloc[idx].get('gps_z_score', 0.0)
        if pd.isna(z_score):
            z_score = 0.0
            
        # Convert labels to a Tensor (float32 is standard for regression)
        labels = torch.tensor([label_x, label_y], dtype=torch.float32)
        
        # Inverse variance weighting is common: weight = 1 / (sigma^2)
        # Avoid division by zero by clamping min accuracy
        sigma = max(float(gps_accuracy), 1.0)
        base_weight = 1.0 / (sigma**2)
        
        # Reliability Penalty: Weighted by Z-Score
        # If z_score > 0 (worse than average for area), reduce weight
        # If z_score <= 0 (better than average), keep full weight (factor = 1.0)
        reliability_factor = 1.0 / (1.0 + max(0.0, float(z_score)))
        
        final_weight = base_weight * reliability_factor
        
        weight = torch.tensor([final_weight], dtype=torch.float32)
        
        return image_tensor, labels, weight
