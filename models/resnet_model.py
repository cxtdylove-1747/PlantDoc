import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes: int = 27, pretrained: bool = True, freeze_backbone: bool = True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model