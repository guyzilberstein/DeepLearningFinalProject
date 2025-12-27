import torch
import torch.nn as nn
import torchvision.models as models

class CampusLocator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load a pre-trained EfficientNet-B0
        # More efficient and accurate than ResNet-18 for small datasets
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        # 2. Modify the output layer ("The Head")
        # EfficientNet's classifier is a Sequential block, the final linear layer is [1]
        # It typically has 1280 input features
        input_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(input_features, 2)
        
    def forward(self, x):
        # This function defines the flow of data
        # Image Tensor -> Backbone -> 2 Coordinates
        return self.backbone(x)
