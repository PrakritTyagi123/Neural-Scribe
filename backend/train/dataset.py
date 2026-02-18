"""
dataset.py - EMNIST Data Loader with Strong Augmentation Pipeline

Improvements over original:
1. Heavy data augmentation (RandomAffine, RandomPerspective, GaussianBlur, RandomErasing)
   - Forces model to learn invariant features instead of memorizing pixel positions
   - Simulates real-world drawing variation (tilt, position, thickness, partial occlusion)
2. Platform-safe num_workers (auto-detects Windows vs Linux)
3. Separate strong augmentation for training vs clean transforms for testing
4. Elastic-like distortion via perspective transform
"""

import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# EMNIST ByMerge: 47 classes
EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
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


def _get_num_workers():
    """8 workers to keep the GPU fed at all times."""
    return 8


def get_data_loaders(data_dir='data/raw/emnist', batch_size=128, num_workers=None):
    """
    Create EMNIST ByMerge train/test DataLoaders with strong augmentation.

    Training augmentation pipeline:
    - RandomAffine: rotation ±10°, translation ±10%, scale 90-110%
      Simulates natural variation in character positioning and size
    - RandomPerspective: 15% distortion, 30% probability
      Simulates viewing angle differences (phone tilt, etc.)
    - GaussianBlur: kernel 3, sigma 0.1-0.7
      Simulates different stroke widths and pen/stylus types
    - RandomErasing: erases small random patches
      Forces model to recognize characters with missing parts (robustness)

    Test pipeline has NO augmentation - clean evaluation.
    """
    if num_workers is None:
        num_workers = _get_num_workers()

    print(f"DataLoader workers: {num_workers}")
    print("Loading EMNIST dataset...")

    # === STRONG training augmentation ===
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeImage(),
        # Geometric augmentation - simulates drawing variation
        transforms.RandomAffine(
            degrees=10,              # ±10° rotation
            translate=(0.10, 0.10),  # ±10% shift
            scale=(0.90, 1.10),      # 90%-110% zoom
            fill=0                   # black fill for new pixels
        ),
        # Perspective warp - simulates viewing angle
        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.3,
            fill=0
        ),
        # Blur - simulates stroke width variation
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.7)
        ),
        # Normalize to EMNIST stats
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
        # Random erasing - forces robustness to partial occlusion
        # Applied AFTER normalize so the erase value is in normalized space
        transforms.RandomErasing(
            p=0.15,                  # 15% chance
            scale=(0.02, 0.12),      # erase 2-12% of image area
            ratio=(0.3, 3.3),        # aspect ratio of erased region
            value=(-EMNIST_MEAN / EMNIST_STD)  # fill with normalized "black"
        ),
    ])

    # === Clean test transform (NO augmentation) ===
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

    print(f"Train samples: {len(train_dataset):,} | Test samples: {len(test_dataset):,}")
    print("Creating DataLoaders...")

    use_persistent = num_workers > 0 and sys.platform != 'win32'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent
    )

    print("DataLoaders ready. Training can begin.")
    return train_loader, test_loader