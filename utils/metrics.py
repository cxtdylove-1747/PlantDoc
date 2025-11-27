import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    acc = (all_labels == all_preds).mean()
    cm = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, output_dict=True)

    return acc, cm, cls_report, all_labels, all_preds, all_probs