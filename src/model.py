import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(DeepfakeDetector, self).__init__()
        
        # CNN backbone - updated for new torchvision
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = 1280  # EfficientNet-B0 features
        
        # Remove classification layer
        self.backbone.classifier = nn.Identity()
        
        # Forensic features branch (simplified)
        self.forensic_fc = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined features
        self.combined_fc = nn.Sequential(
            nn.Linear(num_ftrs + 32, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, forensic_features):
        # Extract CNN features
        cnn_features = self.backbone(x)
        
        # Process forensic features
        forensic_out = self.forensic_fc(forensic_features)
        
        # Combine features
        combined = torch.cat([cnn_features, forensic_out], dim=1)
        output = self.combined_fc(combined)
        
        return output

# Simple version without forensic features
class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)