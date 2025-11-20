import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history, save_path = "Loss.png"):
    """
    Plots training and validation loss from the training history.
    
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close
    print("Saved figure to: ", save_path)


def plot_misclassified_images(model, dataloader, classes, device, max_images=20, save_path = "Misclassified_images.png"):
    """
    Shows misclassified images from a dataloader.

    Args:
        model: Trained PyTorch model.
        dataloader: Validation or test DataLoader.
        classes: List of class names (e.g. train_dataset.classes).
        device: "cuda" or "cpu".
        max_images: Maximum number of images to display.
    """

    misclassified = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            # Indices of incorrect predictions
            incorrect = preds != labels

            for img, pred, true in zip(images[incorrect], preds[incorrect], labels[incorrect]):
                misclassified.append((img.cpu(), pred.item(), true.item()))
                if len(misclassified) >= max_images:
                    break
            if len(misclassified) >= max_images:
                break

    # --- Plotting ---
    num_images = len(misclassified)
    cols = 5
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    for i, (img, pred, true) in enumerate(misclassified):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize display

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_np)
        plt.title(f"Pred: {classes[pred]}\nTrue: {classes[true]}", color="red")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close
    print("Saved figure to: ", save_path)

#TODO: Saliency map is not working yet
def plot_saliency_maps(model, dataloader, device, classes, alpha=0.4):

    for images, labels in dataloader:
        # Take one image
        image = images[0:1].to(device)         # (1,3,H,W)
        img_np = images[0].permute(1,2,0).cpu().numpy()  # for plotting

        image.requires_grad = True

        # Forward pass
        output = model(image)
        pred_class = output.argmax(dim=1).item()

        # Backprop on target score
        score = output[0, pred_class]
        model.zero_grad()
        score.backward()

        # Saliency: (1, H, W)
        saliency = image.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()

        # Normalize saliency to [0,1]
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # Convert to heatmap with cv2
        saliency_uint8 = np.uint8(saliency_norm * 255)
        heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on original image
        img_uint8 = np.uint8(img_np * 255)
        overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)

        # Plot side-by-side
        plt.figure(figsize=(12,5))

        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title(f"Image\nPredicted: {classes[pred_class]}")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(saliency_norm, cmap='hot')
        plt.title("Raw Saliency Map")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(overlay)
        plt.title("Overlay (Heatmap + Image)")
        plt.axis("off")

        save_path = "Saliency_overlay.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved figure to:", save_path)

        break    # remove to process more images
