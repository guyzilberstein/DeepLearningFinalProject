import torch.nn as nn
from torchvision import models

class CampusLocator(nn.Module):
    def __init__(self):
        super(CampusLocator, self).__init__()
        # Use ConvNeXt Tiny (28M params)
        # Weights: IMAGENET1K_V1 (Standard pre-training)
        self.backbone = models.convnext_tiny(weights='DEFAULT')
        
        # ConvNeXt classifier structure in torchvision:
        # Sequential(
        #   [0] LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
        #   [1] Flatten(start_dim=1, end_dim=-1),
        #   [2] Linear(in_features=768, out_features=1000, bias=True)
        # )
        
        # We need to access the input features of the final Linear layer [2]
        in_features = self.backbone.classifier[2].in_features  # Should be 768
        
        # Create Custom Regression Head
        # Using GELU to match ConvNeXt's internal activation function
        regression_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),          # Regularization
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2)         # Output: x_meters, y_meters
        )
        
        # Replace only the final Linear layer, keeping LayerNorm and Flatten
        self.backbone.classifier[2] = regression_head

    def forward(self, x):
        return self.backbone(x)
