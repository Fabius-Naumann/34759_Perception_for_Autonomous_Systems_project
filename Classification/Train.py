import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

from Model import ResNet
from Data import get_dataloaders
from Visualization import plot_training_history, plot_misclassified_images, plot_saliency_maps

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
    Train = True

    model = ResNet(pretrained=True, finetune_all=True).to(device)
    train_dataloader, val_dataloader, classes = get_dataloaders(path=path, batch_size=32)

    if Train:
        model, history = train_model(model=model, train_loader=train_dataloader, val_loader=val_dataloader, 
                            num_epochs=10, lr=1e-4, weight_decay=1e-4, criterion= nn.CrossEntropyLoss())
        #Train Loss: 0.0032 | Train Acc: 99.95% | Val Loss: 0.1208 | Val Acc: 97.02%

        #Save model weights
        save_path = "resnet_weights.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    else:
        model.load_state_dict(torch.load("resnet_weights.pth", map_location=device))
        model.to(device)

    model.eval()

    #Visualize
    plot_training_history(history)
    plot_misclassified_images(model, val_dataloader, classes, device=device, max_images=10)
    plot_saliency_maps(model, val_dataloader, device=device, classes=classes)