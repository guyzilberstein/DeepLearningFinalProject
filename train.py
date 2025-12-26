import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import os
import copy
import numpy as np
from Preprocessing.img_to_tensor import CampusDataset
from model import CampusLocator

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

def train_model(num_epochs=20, batch_size=16, learning_rate=0.001):
    # 1. Setup Paths
    # Current directory is project root
    csv_file = 'Preprocessing/processed_data.csv'
    img_dir = 'ProcessedImages'
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Processed data not found at {csv_file}. Run coordinates_normalizer.py first.")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Processed images not found at {img_dir}. Run preprocess_images.py first.")

    # 2. Prepare Dataset
    full_dataset = CampusDataset(csv_file=csv_file, root_dir=img_dir)
    total_len = len(full_dataset)
    
    # Define Split Ratios: 70% Train, 15% Val, 15% Test
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    # Fix the seed for reproducibility so Test set remains constant across runs
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Save the test indices so evaluate.py knows which images are in the test set
    # We can save indices to a numpy file
    test_indices = test_dataset.indices
    np.save('test_indices.npy', test_indices)
    print(f"Saved {len(test_indices)} test indices to test_indices.npy")
    
    print(f"Dataset Split: {train_size} Train, {val_size} Val, {test_size} Test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CampusLocator().to(device)
    
    # Define BOTH loss functions
    train_criterion = WeightedMSELoss() # For learning (trust good data more)
    val_criterion = nn.MSELoss()        # For evaluation (treat all test points equally)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
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
                weights = weights.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    if phase == 'train':
                        # Training: Use Weighted Loss
                        loss = train_criterion(outputs, labels, weights)
                        loss.backward()
                        optimizer.step()
                    else:
                        # Validation: Use Standard MSE Loss
                        # We ignore weights here, simulating real-world usage where we don't know accuracy
                        loss = val_criterion(outputs, labels)
                        
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_campus_locator.pth')
                print("New best model saved!")

    print(f'Best val loss: {best_loss:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    train_model()
