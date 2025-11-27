import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
import pandas as pd

from models.resnet_model import build_resnet50
from configs.base_config import BaseConfig

def main():
    cfg = BaseConfig
    # 指定使用哪个实验的最佳模型
    exp_name = "resnet50_finetune_aug"
    cfg.experiment_name = exp_name
    exp_dir = cfg.get_experiment_dir()
    best_model_path = os.path.join(exp_dir, "best_model.pt")

    device = cfg.device

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root=cfg.test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    class_names = test_dataset.classes

    model = build_resnet50(num_classes=cfg.num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    all_image_paths = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # ImageFolder 会按固定顺序读取图片，可以用 dataset.samples 拿路径
            batch_indices = range(len(all_image_paths), len(all_image_paths) + len(images))
            for idx, pred in zip(batch_indices, preds):
                img_path, _ = test_dataset.samples[idx]
                all_image_paths.append(img_path)
                all_preds.append(class_names[pred])

    df = pd.DataFrame({
        "image_path": all_image_paths,
        "pred_label": all_preds,
    })

    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, "test_predictions.csv")
    df.to_csv(save_path, index=False)
    print("Saved test predictions to:", save_path)

if __name__ == "__main__":
    main()