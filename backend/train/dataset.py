"""
dataset.py - Stable EMNIST Data Loader (Windows + GPU optimized)

This version prioritizes reliable loading and fast training startup.
Heavy augmentations are removed temporarily to avoid Windows stalls.
You can re-add them later once training runs smoothly.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# EMNIST ByMerge: 47 classes
EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

NUM_CLASSES = 47

# EMNIST dataset normalization stats
EMNIST_MEAN = 0.1751
EMNIST_STD = 0.3332


class TransposeImage:
    """
    EMNIST images are rotated/transposed by default.
    This fixes orientation and is safe for multiprocessing.
    """
    def __call__(self, x):
        return x.transpose(1, 2)


def get_data_loaders(data_dir='data/raw/emnist', batch_size=128, num_workers=4):
    """
    Create EMNIST ByMerge train/test DataLoaders.

    Optimizations:
    - num_workers=4 for fast loading
    - pin_memory=True for faster GPU transfer
    - persistent_workers=True to avoid worker restart overhead
    - simple transforms to avoid Windows stalls
    """

    print("Loading EMNIST dataset...")

    # Stable transforms (no heavy augmentation)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeImage(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeImage(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,))
    ])

    os.makedirs(data_dir, exist_ok=True)

    # Download/load datasets
    train_dataset = datasets.EMNIST(
        root=data_dir,
        split='bymerge',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split='bymerge',
        train=False,
        download=True,
        transform=test_transform
    )

    print("Dataset loaded. Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print("DataLoaders ready. Training can begin.")

    return train_loader, test_loader