import os
import torch
from tqdm import tqdm
from utils.metrics import accuracy_from_logits
from utils.visualization import plot_training_curves

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    num_epochs: int = 30,
    device: str = "cuda",
    save_dir: str = "./outputs/exp",
    save_best_only: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += acc * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_acc / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = accuracy_from_logits(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_acc += acc * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_acc / len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch}/{num_epochs} "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Scheduler step
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Save model
        if save_best_only:
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), best_model_path)
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pt"))

    # 保存最后一轮
    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pt"))

    # 绘制曲线
    plot_training_curves(history, os.path.join(save_dir, "training_curves.png"))

    return history, best_model_path