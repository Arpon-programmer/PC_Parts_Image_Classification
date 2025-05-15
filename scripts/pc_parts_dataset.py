from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .image_processing import train_transform, val_transform

def train_dataloader(train_data_path, batch_size=32):
    train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def val_dataloader(val_data_path, batch_size=32):
    val_dataset = ImageFolder(root=val_data_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return val_loader