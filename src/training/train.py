import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import os
import copy
import numpy as np
import sys
import pandas as pd

# Ensure we can import from src
# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.dataset import CampusDataset
from src.model.network import CampusLocator

# Custom Weighted MSE Loss
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, prediction, target, weights):
        # weights shape might be [batch_size, 1]
        # prediction, target shape: [batch_size, 2]
        
        squared_errors = (prediction - target) ** 2
        # Sum errors for x and y, then apply weight
        weighted_errors = squared_errors * weights
        
        # Mean over the batch
        return weighted_errors.mean()

def train_model(num_epochs=25, batch_size=16, learning_rate=0.001):
    # 1. Setup Paths
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'processed_images')
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    output_dir = os.path.join(project_root, 'outputs')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Processed data not found at {csv_file}. Run normalize_coords.py first.")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Processed images not found at {img_dir}. Run convert_images.py first.")

    # 2. Prepare Dataset
    # Using is_train=False for both to disable data augmentation (restore best baseline)
    full_dataset_train = CampusDataset(csv_file=csv_file, root_dir=img_dir, is_train=False)
    full_dataset_val   = CampusDataset(csv_file=csv_file, root_dir=img_dir, is_train=False)
    
    # Read CSV for stratification
    df = pd.read_csv(csv_file)
    indices = np.arange(len(df))
    
    # We use 'source_file' column for stratification to ensure all locations are represented
    stratify_labels = df['source_file'].values
    
    # First split: Separate Test (5%) from the rest (95%)
    # Stratify by location (source_file)
    from sklearn.model_selection import train_test_split
    
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=0.05, 
        stratify=stratify_labels, 
        random_state=42
    )
    
    # Get labels for the remaining train_val set for the next split
    train_val_labels = stratify_labels[train_val_idx]
    
    # Second split: Separate Validation (10% of total) from the remaining 95%
    # The remaining 85% of total becomes the Training set.
    # Val share of remaining = 0.10 / 0.95 = ~0.1053
    val_share_of_remaining = 0.10 / 0.95
    
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=val_share_of_remaining, 
        stratify=train_val_labels, 
        random_state=42
    )
    
    # Create Subsets
    # Use the Augmentation dataset for Train, and the Clean dataset for Val/Test
    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_val, val_idx)
    test_dataset = Subset(full_dataset_val, test_idx)
    
    # Verify split distribution in Test set (Optional, for logging)
    print("\nTest Set Distribution by Location:")
    test_df = df.iloc[test_idx]
    print(test_df['source_file'].value_counts())
    print("-" * 30)
    
    # Save the test indices so evaluate.py knows which images are in the test set
    test_indices = test_idx  # test_idx is already a numpy array of indices
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    print(f"Saved {len(test_indices)} test indices to outputs/test_indices.npy")
    
    print(f"Dataset Split: {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    
    # Define Loss function
    # User Experiment: Switch to Standard MSE (Ignore GPS Accuracy weights)
    # We found that 1/sigma^2 was too aggressive (56x difference between best and worst)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler: Reduce LR by factor of 0.1 if val_loss doesn't improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    # Load previous best loss if checkpoint exists (to avoid overwriting a better model)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_campus_locator.pth')
    if os.path.exists(checkpoint_path):
        try:
            existing_checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(existing_checkpoint, dict) and 'best_loss' in existing_checkpoint:
                best_loss = existing_checkpoint['best_loss']
                print(f"Loaded existing best_loss: {best_loss:.4f} (will only save if we beat this)")
            else:
                print("Existing checkpoint found but no best_loss recorded. Will overwrite if any improvement.")
        except Exception as e:
            print(f"Could not load existing checkpoint: {e}. Starting fresh.")
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            
            for inputs, labels, weights in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # We ignore weights now
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Step the scheduler on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)
            
            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(checkpoint_dir, 'best_campus_locator.pth')
                # Save both model weights and best_loss so we can compare across runs
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss
                }, save_path)
                print(f"New best model saved to {save_path} (loss: {best_loss:.4f})")

    print(f'Best val loss: {best_loss:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    train_model()
