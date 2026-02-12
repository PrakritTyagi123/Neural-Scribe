"""
dataset.py - MNIST Dataset Loading and Preparation

This module handles loading the MNIST dataset from local storage or
downloading it automatically via PyTorch's torchvision.

The MNIST dataset contains:
    - 60,000 training images
    - 10,000 test images
    - Each image is 28×28 grayscale (0-255)
    - Labels are digits 0-9
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Path to store MNIST data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data', 'raw', 'mnist'
)


def get_transforms():
    """
    Get the image transformations for MNIST.
    
    Transforms:
        1. ToTensor: Convert PIL Image to tensor (0-1 range)
        2. Normalize: Standardize with MNIST mean/std
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])


def get_train_dataset():
    """
    Load the MNIST training dataset.
    
    Downloads the dataset if not present locally.
    
    Returns:
        torchvision.datasets.MNIST: Training dataset
    """
    return datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=get_transforms()
    )


def get_test_dataset():
    """
    Load the MNIST test dataset.
    
    Downloads the dataset if not present locally.
    
    Returns:
        torchvision.datasets.MNIST: Test dataset
    """
    return datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=get_transforms()
    )


def get_train_loader(batch_size=64, shuffle=True, num_workers=0):
    """
    Create a DataLoader for training data.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        
    Returns:
        DataLoader: Training data loader
    """
    dataset = get_train_dataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def get_test_loader(batch_size=64, shuffle=False, num_workers=0):
    """
    Create a DataLoader for test data.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        
    Returns:
        DataLoader: Test data loader
    """
    dataset = get_test_dataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def get_data_loaders(batch_size=64, num_workers=0):
    """
    Get both training and test data loaders.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = get_train_loader(batch_size, num_workers=num_workers)
    test_loader = get_test_loader(batch_size, num_workers=num_workers)
    
    return train_loader, test_loader


def get_sample_batch(loader):
    """
    Get a single batch from a data loader.
    
    Args:
        loader: DataLoader instance
        
    Returns:
        tuple: (images, labels) tensors
    """
    return next(iter(loader))


if __name__ == "__main__":
    # Test the dataset loading
    print("Loading MNIST dataset...")
    print(f"Data directory: {DATA_DIR}")
    
    train_loader, test_loader = get_data_loaders()
    
    print(f"\nTraining samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    images, labels = get_sample_batch(train_loader)
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Sample labels: {labels[:10].tolist()}")