import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2

def plot_training_curves(history, save_path: str):
    """
    history: dict with keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "bo-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "bo-", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "ro-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path: str, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def grad_cam(model, img_tensor, target_layer_name: str, class_idx: int, device="cuda"):
    """
    简单 Grad-CAM 实现，用于可视化卷积层关注区域。
    img_tensor: (1, C, H, W)
    target_layer_name: e.g., "layer4" for ResNet
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # 获取目标层
    target_layer = dict(model.named_modules())[target_layer_name]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    scores = model(img_tensor)
    if class_idx is None:
        class_idx = torch.argmax(scores, dim=1).item()
    score = scores[:, class_idx]

    model.zero_grad()
    score.backward()

    act = activations["value"]       # (N, C, H, W)
    grad = gradients["value"]        # (N, C, H, W)
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
    cam = (weights * act).sum(dim=1, keepdim=True)  # (N, 1, H, W)
    cam = torch.relu(cam)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam


def overlay_cam_on_image(image_np, cam, save_path: str):
    """
    image_np: HxWx3, [0,1]
    cam: HxW, [0,1]
    """
    h, w, _ = image_np.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * heatmap / 255.0 + 0.5 * image_np
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()