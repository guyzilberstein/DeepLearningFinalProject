import torch
import torch.nn as nn
import torchvision.models as models

class CampusLocator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load a pre-trained ResNet-18
        # "weights='DEFAULT'" downloads a version that has already seen millions of images.
        # This is called "Transfer Learning" - it already knows how to "see".
        self.backbone = models.resnet18(weights='DEFAULT')
        
        # 2. Modify the output layer ("The Head")
        # The original .fc (fully connected) layer expects to output 1000 classes.
        # We check how many inputs it usually takes (512 for ResNet18)
        # And force it to output only 2 numbers: X_meters and Y_meters.
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, 2)
        
    def forward(self, x):
        # This function defines the flow of data
        # Image Tensor -> Backbone -> 2 Coordinates
        return self.backbone(x)
