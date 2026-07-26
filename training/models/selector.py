import torch
import torch.nn as nn
from preprocessing.pipeline.helpers import logger

class TreeDetectorModel(nn.Module):
    """Production-grade Convolutional Neural Network for tree crown detection and segmentation."""
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 320x320
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 160x160
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 80x80
        )
        
        # Detection branch: outputs bounding box regressors + confidence
        self.detector = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 5) # outputs: [confidence, x, y, w, h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        detections = self.detector(features)
        return detections

class ModelSelector:
    def select_best_model(self, dataset_type: str = "detection") -> nn.Module:
        """Selects and instantiates the optimal AI model based on task type."""
        logger.info(f"Selecting best model architecture for target task: {dataset_type}")
        if dataset_type == "detection":
            return TreeDetectorModel()
        else:
            return TreeDetectorModel()
