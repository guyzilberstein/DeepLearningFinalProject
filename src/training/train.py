import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import GradScaler, autocast
import os
import copy
import numpy as np
import random
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


def get_device_config():
    """
    Auto-detect device and return optimal settings for that hardware.
    Returns: (device, batch_size, num_workers, pin_memory, use_amp)
    """
    if torch.cuda.is_available():
        # GPU cluster - enable all optimizations
        device = torch.device("cuda")
        # ConvNeXt is heavier than B0. 
        # 32 might OOM on 1080 Ti (11GB). 24 is safer, 16 is safest.
        batch_size = 24  
        num_workers = 4  # Multi-process data loading
        pin_memory = True  # Faster CPU→GPU transfer
        use_amp = True  # Mixed precision for ~2x speedup
        # Optimize convolution algorithms
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        # Mac with Apple Silicon
        device = torch.device("mps")
        batch_size = 8  # Conservative for MPS memory
        num_workers = 0  # MPS doesn't benefit from multiprocessing
        pin_memory = False
        use_amp = False  # MPS doesn't support AMP well
    else:
        # CPU fallback
        device = torch.device("cpu")
        batch_size = 16
        num_workers = 2
        pin_memory = False
        use_amp = False
    
    return device, batch_size, num_workers, pin_memory, use_amp


def train_model(num_epochs=25, batch_size=None, learning_rate=0.0001, experiment_name="default", resume=False, seed=42):
    """
    Train the model.
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size (auto-detected if None)
        learning_rate: Initial learning rate
        experiment_name: Name for this experiment (used for checkpoint naming)
        resume: If True, load weights from existing checkpoint and continue training
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 1. Setup Device & Config
    device, auto_batch_size, num_workers, pin_memory, use_amp = get_device_config()
    
    # Use provided batch_size or auto-detected
    if batch_size is None:
        batch_size = auto_batch_size
    
    print(f"=== Experiment: {experiment_name} ===")
    print(f"Device: {device}")
    print(f"Config: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, use_amp={use_amp}")
    print(f"Resume: {resume}, Seed: {seed}")
    
    # 2. Setup Paths
    csv_file = os.path.join(project_root, 'data', 'dataset.csv')
    img_dir = os.path.join(project_root, 'data', 'images')
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

    # 3. Prepare Dataset
    # Training dataset with augmentation, validation without
    full_dataset_train = CampusDataset(csv_file=csv_file, root_dir=img_dir, is_train=True)
    full_dataset_val   = CampusDataset(csv_file=csv_file, root_dir=img_dir, is_train=False)
    
    # Read CSV for stratification
    df = pd.read_csv(csv_file)
    indices = np.arange(len(df))
    
    # We use 'source_file' column for stratification to ensure all locations are represented
    stratify_labels = df['source_file'].values
    
    # 85/15 Train/Val split (Test set is now external in test_dataset.csv)
    from sklearn.model_selection import train_test_split
    
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.15, 
        stratify=stratify_labels, 
        random_state=seed
    )
    
    # Create Subsets
    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_val, val_idx)
    
    # Verify split distribution
    print("\nValidation Set Distribution by Location:")
    val_df = df.iloc[val_idx]
    print(val_df['source_file'].value_counts())
    print("-" * 30)
    
    print(f"Dataset Split: {len(train_dataset)} Train, {len(val_dataset)} Val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    # 4. Initialize Model, Loss, Optimizer
    model = CampusLocator().to(device)
    
    # HuberLoss handles outliers better than MSELoss
    # Linear for large errors (noisy GPS), quadratic for small errors (fine-tuning)
    criterion = nn.HuberLoss(delta=1.0)
    
    # Use AdamW instead of Adam for ConvNeXt
    # AdamW separates weight decay from the gradient update. 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler: Reduce LR by factor of 0.5 if val_loss doesn't improve for 7 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    # Mixed Precision Scaler (only used if use_amp=True)
    scaler = GradScaler(enabled=use_amp)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    start_epoch = 0
    
    # Checkpoint path based on experiment name
    checkpoint_filename = f'best_{experiment_name}.pth' if experiment_name != "default" else 'best_campus_locator.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            existing_checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(existing_checkpoint, dict) and 'best_loss' in existing_checkpoint:
                best_loss = existing_checkpoint['best_loss']
                
                if resume:
                    # Resume training: load model weights and optimizer state
                    model.load_state_dict(existing_checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in existing_checkpoint:
                        optimizer.load_state_dict(existing_checkpoint['optimizer_state_dict'])
                    if 'epoch' in existing_checkpoint:
                        start_epoch = existing_checkpoint['epoch'] + 1
                    print(f"RESUMING from epoch {start_epoch}, best_loss: {best_loss:.4f}")
                else:
                    print(f"Loaded existing best_loss: {best_loss:.4f} (will only save if we beat this)")
            else:
                print("Existing checkpoint found but no best_loss recorded. Will overwrite if any improvement.")
        except Exception as e:
            print(f"Could not load existing checkpoint: {e}. Starting fresh.")
    elif resume:
        print(f"WARNING: resume=True but no checkpoint found at {checkpoint_path}. Starting fresh.")
    
    # 5. Training Loop
    for epoch in range(num_epochs):
        actual_epoch = start_epoch + epoch if resume else epoch
        print(f'Epoch {actual_epoch+1}/{start_epoch + num_epochs if resume else num_epochs}')
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
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        # Scaled backward pass for mixed precision
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
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
                # Save model weights, optimizer state, epoch, and best_loss
                torch.save({
                    'epoch': actual_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'experiment': experiment_name
                }, checkpoint_path)
                print(f"New best model saved to {checkpoint_path} (loss: {best_loss:.4f})")

    print(f'Best val loss: {best_loss:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    train_model()
