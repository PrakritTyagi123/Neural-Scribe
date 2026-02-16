"""
preprocess.py - Canvas Data → EMNIST Format
Converts raw pixel arrays from the browser canvas into normalized tensors.
"""
import torch
import numpy as np


def preprocess_pixels(pixel_data: list[float], device=None) -> torch.Tensor:
    """
    Convert 784-length pixel array from canvas to EMNIST-compatible tensor.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pixels = np.array(pixel_data, dtype=np.float32).reshape(28, 28)

    # Normalize to [0, 1] then apply EMNIST normalization
    pixels = pixels / 255.0
    pixels = (pixels - 0.1751) / 0.3332

    tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)