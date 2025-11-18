from Model import ResNet
from Data import get_dataloaders

import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, weight_decay=1e-4, criterion = nn.CrossEntropyLoss(), scheduler=None, device=None):
    
    model = model.to(device)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # --- Training Loop ---
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Optional LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = r"C:\Users\leona\fiftyone\coco-2017\dataset"

    model = ResNet(pretrained=True, finetune_all=True).to(device)
    train_dataloader, val_dataloader, _ = get_dataloaders(path=path, batch_size=32)

    model, history = train_model(model=model, train_loader=train_dataloader, val_loader=val_dataloader, 
                         num_epochs=10, lr=1e-4, weight_decay=1e-4, criterion= nn.CrossEntropyLoss())
    #Train Loss: 0.0124 | Train Acc: 99.50% | Val Loss: 0.1292 | Val Acc: 95.63%
    
    #Save model weights
    save_path = "resnet_weights.pth"
    torch.save(model.state_dict(), save_path)

    print(f"Model weights saved to {save_path}")