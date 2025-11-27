import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.base_config import BaseConfig
from utils.dataset import create_dataloaders
from models.vit_model import build_vit
from trainers.train_utils import train_model

def main():
    cfg = BaseConfig
    cfg.experiment_name = "vit_tiny_aug"
    exp_dir = cfg.get_experiment_dir()

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_dir=cfg.train_dir,
        test_dir=cfg.test_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,   # 可改小一点，如16，避免爆显存
        num_workers=cfg.num_workers,
        val_ratio=0.2,
        use_augmentation=True,
    )

    model = build_vit(num_classes=cfg.num_classes, model_name="vit_tiny_patch16_224", pretrained=True)

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    history, best_model_path = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=scheduler,
        num_epochs=cfg.epochs,
        device=cfg.device,
        save_dir=exp_dir,
        save_best_only=cfg.save_best_only,
    )

    print("Best model saved at:", best_model_path)

if __name__ == "__main__":
    main()