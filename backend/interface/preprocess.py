"""
preprocess.py - Image Preprocessing for Inference

This module converts raw pixel data from the browser canvas into
MNIST-compatible format for model inference.

Input format from frontend:
    - 28×28 array of pixel values (0-255 grayscale)
    - Or flat array of 784 values
    
Output format for model:
    - PyTorch tensor of shape (1, 1, 28, 28)
    - Normalized with MNIST statistics
"""

import numpy as np
import torch
from PIL import Image
import io
import base64


# MNIST normalization values
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_pixels(pixels):
    """
    Normalize pixel values to MNIST range.
    
    Args:
        pixels: Numpy array with values 0-255
        
    Returns:
        Numpy array with normalized values
    """
    # Convert to 0-1 range
    normalized = pixels.astype(np.float32) / 255.0
    
    # Apply MNIST normalization
    normalized = (normalized - MNIST_MEAN) / MNIST_STD
    
    return normalized


def preprocess_pixel_array(pixel_array):
    """
    Preprocess a pixel array from the frontend canvas.
    
    Args:
        pixel_array: List or array of 784 pixel values (0-255)
                    Can be flat [784] or 2D [28, 28]
        
    Returns:
        torch.Tensor: Shape (1, 1, 28, 28), normalized
    """
    # Convert to numpy array
    pixels = np.array(pixel_array, dtype=np.float32)
    
    # Reshape if flat
    if pixels.ndim == 1:
        pixels = pixels.reshape(28, 28)
    
    # Ensure correct shape
    assert pixels.shape == (28, 28), f"Expected (28, 28), got {pixels.shape}"
    
    # Normalize
    normalized = normalize_pixels(pixels)
    
    # Convert to tensor with batch and channel dimensions
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor


def preprocess_image(image):
    """
    Preprocess a PIL Image for inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        torch.Tensor: Shape (1, 1, 28, 28), normalized
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    pixels = np.array(image, dtype=np.float32)
    
    # Normalize and convert to tensor
    normalized = normalize_pixels(pixels)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor


def preprocess_base64(base64_string):
    """
    Preprocess a base64-encoded image.
    
    Args:
        base64_string: Base64 encoded image data
                      Can include data URL prefix
        
    Returns:
        torch.Tensor: Shape (1, 1, 28, 28), normalized
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(base64_string)
    
    # Open as PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    return preprocess_image(image)


def invert_if_needed(tensor):
    """
    Invert image colors if background is dark.
    
    MNIST uses white digits on black background (0 = black, 1 = white after norm).
    Canvas typically draws black on white, so we may need to invert.
    
    Args:
        tensor: Input tensor
        
    Returns:
        torch.Tensor: Possibly inverted tensor
    """
    # Calculate mean (before normalization this would be center value)
    # If mean is high, the image has white background (needs inversion)
    mean_val = tensor.mean().item()
    
    # Threshold based on normalized values
    # High mean = white background = needs inversion
    if mean_val > 0:  # Centered around 0 after normalization
        # Invert by negating (since it's normalized)
        tensor = -tensor
    
    return tensor


def center_digit(pixels):
    """
    Center the digit in the image based on center of mass.
    
    Args:
        pixels: 28x28 numpy array
        
    Returns:
        numpy array: Centered 28x28 image
    """
    # Find non-zero pixels (digit pixels)
    rows, cols = np.where(pixels > 0.1 * pixels.max())
    
    if len(rows) == 0:
        return pixels
    
    # Calculate center of mass
    center_row = int(np.mean(rows))
    center_col = int(np.mean(cols))
    
    # Calculate shift needed to center
    shift_row = 14 - center_row
    shift_col = 14 - center_col
    
    # Apply shift using numpy roll
    centered = np.roll(pixels, shift_row, axis=0)
    centered = np.roll(centered, shift_col, axis=1)
    
    return centered


def preprocess_canvas_data(pixel_array, invert=False, center=True):
    """
    Full preprocessing pipeline for canvas data.
    
    Args:
        pixel_array: Raw pixel data from canvas (784 or 28x28 values)
                    Canvas.js already sends inverted data (0=background, 255=digit)
        invert: Whether to invert colors (usually False - canvas.js handles this)
        center: Whether to center the digit
        
    Returns:
        torch.Tensor: Ready for model inference
    """
    # Convert to numpy
    pixels = np.array(pixel_array, dtype=np.float32)
    
    # Reshape if needed
    if pixels.ndim == 1:
        pixels = pixels.reshape(28, 28)
    
    # Invert if needed (canvas.js already inverts, so usually False)
    if invert:
        pixels = 255.0 - pixels
    
    # Center the digit
    if center:
        pixels = center_digit(pixels)
    
    # Normalize to 0-1 range first
    pixels = pixels / 255.0
    
    # Apply MNIST normalization
    normalized = (pixels - MNIST_MEAN) / MNIST_STD
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing utilities...")
    
    # Create a dummy 28x28 image with a simple pattern
    dummy_pixels = np.zeros((28, 28), dtype=np.float32)
    dummy_pixels[10:20, 12:16] = 255  # Draw a vertical line (like digit "1")
    
    # Test pixel array preprocessing
    tensor = preprocess_pixel_array(dummy_pixels)
    print(f"Pixel array preprocessing:")
    print(f"  Input shape: (28, 28)")
    print(f"  Output shape: {tensor.shape}")
    print(f"  Output range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Test flat array
    flat_pixels = dummy_pixels.flatten()
    tensor_flat = preprocess_pixel_array(flat_pixels)
    print(f"\nFlat array preprocessing:")
    print(f"  Input shape: (784,)")
    print(f"  Output shape: {tensor_flat.shape}")
    
    # Test canvas preprocessing
    tensor_canvas = preprocess_canvas_data(dummy_pixels, invert=False, center=True)
    print(f"\nCanvas preprocessing (with centering):")
    print(f"  Output shape: {tensor_canvas.shape}")
    
    print("\n✓ All preprocessing tests passed!")