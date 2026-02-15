"""
dataset.py - MNIST Data Loading
Downloads and prepares MNIST dataset with proper transforms.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_data_loaders(data_dir='data/raw/mnist', batch_size=64, num_workers=2):
    """Load MNIST training and test datasets."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
