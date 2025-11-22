import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import os
import joblib
from sklearn.decomposition import PCA

class PCATransform:
    """Apply PCA on flattened image tensors"""
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None

    def fit(self, dataset):
        # Flatten all images and stack them
        data = torch.stack([img.view(-1) for img, _ in dataset])
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(data.numpy())

    def __call__(self, img):
        # Flatten, apply PCA, and convert back to tensor
        img_flat = img.view(-1).numpy()
        img_pca = self.pca.transform([img_flat])[0]
        return torch.tensor(img_pca, dtype=torch.float32)


def get_dataloaders(path, batch_size=32, num_workers=4, train_split=0.8, shuffle=True, apply_pca=False, pca_components=0.95):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(path, transform=transform)
    class_names = dataset.classes

    # Split sizes of training and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Optionally fit PCA on training set
    if apply_pca:
        if os.path.exists("pca.pkl"):
            print("Load PCA")
            pca_transform = PCATransform(n_components=pca_components)
            pca_transform.pca = joblib.load("pca.pkl")
        else:
            print("PCA Transform")
            pca_transform = PCATransform(n_components=pca_components)
            pca_transform.fit(train_dataset)
            joblib.dump(pca_transform.pca, "pca.pkl")
        
        pca_pipeline = transforms.Compose([
            transforms.ToTensor(),
            pca_transform
        ])

        # Replace original transform with PCA transform
        train_dataset.dataset.transform = pca_pipeline
        val_dataset.dataset.transform = pca_pipeline

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names


if __name__ == "__main__":
    path = r"C:\Users\leona\fiftyone\coco-2017\dataset"
    train_dataloader, val_dataloader, classes = get_dataloaders(path)
    print("classes:", classes) # Check if correct folder was loaded
    for imgs, labels in train_dataloader:
        print(imgs.shape, labels)
        #imgs.shape: torch.Size([32, 3, 128, 128])
        break
