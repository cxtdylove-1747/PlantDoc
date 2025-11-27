import timm
import torch.nn as nn

def build_vit(num_classes: int = 27, model_name: str = "vit_tiny_patch16_224", pretrained: bool = True):
    model = timm.create_model(model_name, pretrained=pretrained)
    if hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unknown head attribute for ViT model")
    return model