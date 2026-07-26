"""
GeoTree Model Architecture Definition for Hugging Face Loading
"""
import torch
import torch.nn as nn

class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)

class TreeDetectorModel(nn.Module):
    """GeoTree Residual Convolutional Neural Network for Tree Crown Detection."""
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            ConvResidualBlock(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            ConvResidualBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        detections = self.detector(features)
        return torch.cat([
            detections[:, :1],
            torch.sigmoid(detections[:, 1:])
        ], dim=1)
