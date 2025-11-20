import torch
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
def plot_saliency_maps(model, dataloader, device, classes=None, max_images=5):
    """
    Computes and plots saliency maps for some images.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader to draw images from.
        device: "cuda" or "cpu".
        classes: Optional list of class names.
        max_images: How many images to visualize.
    """

    images_shown = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True  # ✨ allow gradients w.r.t. pixels

        outputs = model(images)
        _, preds = outputs.max(1)

        for idx in range(images.size(0)):
            if images_shown >= max_images:
                return

            img = images[idx:idx+1]          # keep batch dimension
            label = labels[idx]
            pred = preds[idx]

            model.zero_grad()

            # Loss for the predicted class (common choice)
            score = outputs[idx, pred]
            score.backward(retain_graph=True)

            # saliency = max gradient across channels
            saliency = img.grad.data.abs().max(dim=1)[0].squeeze().cpu()

            # Normalize saliency for display
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

            # Convert image for plotting
            img_np = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            # --- Plot ---
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            title = f"Pred: {pred.item()}"
            if classes is not None:
                title = f"Pred: {classes[pred]} | True: {classes[label]}"
            plt.title(title)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(saliency, cmap="hot")
            plt.title("Saliency Map")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"Saliency_Map_{idx}", dpi=200)
            plt.close
            print("Saved figure to: ", f"Saliency_Map_{idx}")

            images_shown += 1

        if images_shown >= max_images:
            break