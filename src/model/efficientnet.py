import torch.nn as nn
from torchvision import models

class EfficientNetLocator(nn.Module):
    """
    EfficientNet-B0 based locator (Legacy architecture).
    Used for loading older checkpoints and for hybrid ensembles.
    """
    def __init__(self):
        super(EfficientNetLocator, self).__init__()
        # EfficientNet-B0 backbone with pretrained ImageNet weights
        # Best fit for ~1800 images (5.3M params)
        # Note: We don't load weights='DEFAULT' here because we'll load from checkpoint
        self.backbone = models.efficientnet_b0(weights=None)
        
        # Get input features of the final layer (1280 for B0)
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with a Regression MLP Head
        # MLP allows non-linear relationships between features and GPS
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),          # Regularization
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)         # Output: x_meters, y_meters
        )

    def forward(self, x):
        return self.backbone(x)

