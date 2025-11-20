from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(path, batch_size=32, num_workers=4, train_split=0.8, shuffle=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset and transform images 
    dataset = datasets.ImageFolder(path, transform=transform)
    class_names = dataset.classes

    # split sizes of training and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloader for training and validation data
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
