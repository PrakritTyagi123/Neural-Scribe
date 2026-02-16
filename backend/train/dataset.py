"""
dataset.py - EMNIST ByMerge Data Loading
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

NUM_CLASSES = 47


class TransposeImage:
    """EMNIST images are transposed vs MNIST — this fixes them."""
    def __call__(self, x):
        return x.transpose(1, 2)


def get_data_loaders(data_dir='data/raw/emnist', batch_size=64, num_workers=8):
    """Load EMNIST ByMerge training and test datasets."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeImage(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])

    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.EMNIST(
        root=data_dir, split='bymerge', train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        root=data_dir, split='bymerge', train=False, download=True, transform=transform
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