"""
preprocess.py - Canvas Data → MNIST Format
Converts raw pixel arrays from the browser canvas into normalized tensors.
"""
import torch
import numpy as np


def preprocess_pixels(pixel_data: list[float], device=None) -> torch.Tensor:
    """
    Convert 784-length pixel array from canvas to MNIST-compatible tensor.
    
    Args:
        pixel_data: List of 784 floats (0-255), already grayscale inverted
        device: torch device
    
    Returns:
        Tensor of shape (1, 1, 28, 28), normalized for MNIST
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to numpy array
    pixels = np.array(pixel_data, dtype=np.float32).reshape(28, 28)

    # Normalize to [0, 1] then apply MNIST normalization
    pixels = pixels / 255.0
    pixels = (pixels - 0.1307) / 0.3081

    # Convert to tensor: (1, 1, 28, 28)
    tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)
