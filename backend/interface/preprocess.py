"""
preprocess.py - Canvas Data → Model Format

Updated for 76 classes:
- Wider center-of-mass shift clamp (±6px) for asymmetric symbols like ∫, ∑
- Adaptive smoothing based on stroke density
- Same pipeline otherwise (must match training exactly)
"""
import torch
import numpy as np


def _center_of_mass(pixels_2d: np.ndarray) -> tuple[float, float]:
    """Compute brightness-weighted centroid of the image."""
    total = pixels_2d.sum()
    if total < 1e-6:
        return 14.0, 14.0

    rows = np.arange(pixels_2d.shape[0])
    cols = np.arange(pixels_2d.shape[1])

    cy = float(np.sum(rows[:, None] * pixels_2d) / total)
    cx = float(np.sum(cols[None, :] * pixels_2d) / total)

    return cy, cx


def preprocess_pixels(pixel_data: list[float], device=None) -> torch.Tensor:
    """
    Convert 784-length pixel array from canvas to model-compatible tensor.

    Pipeline:
    1. Reshape to 28×28
    2. Normalize to [0, 1]
    3. Adaptive Gaussian smoothing
    4. Center-of-mass alignment (wider ±6px clamp for symbols)
    5. Transpose (matches EMNIST TransposeImage)
    6. EMNIST normalization
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MEAN = 0.1751
    STD = 0.3332

    # Step 1: Reshape
    pixels = np.array(pixel_data, dtype=np.float32).reshape(28, 28)

    # Step 2: Normalize to [0, 1]
    pixels = pixels / 255.0

    # Step 3: Adaptive Gaussian smoothing
    # Compute stroke density — thin strokes (symbols) get less smoothing
    stroke_density = (pixels > 0.1).sum() / (28 * 28)

    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0

    padded = np.pad(pixels, 1, mode='constant', constant_values=0)
    smoothed = np.zeros_like(pixels)
    for i in range(28):
        for j in range(28):
            smoothed[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kernel)

    # Less smoothing for thin strokes (math symbols), more for thick strokes (letters)
    blend = min(0.35, max(0.15, stroke_density * 1.5))
    pixels = (1.0 - blend) * pixels + blend * smoothed

    # Step 4: Center-of-mass alignment
    cy, cx = _center_of_mass(pixels)

    shift_y = 13.5 - cy
    shift_x = 13.5 - cx

    # Wider clamp for symbols like ∫, ∑ that have asymmetric mass distribution
    shift_y = np.clip(shift_y, -6, 6)
    shift_x = np.clip(shift_x, -6, 6)

    if abs(shift_y) > 0.5 or abs(shift_x) > 0.5:
        iy, ix = int(np.round(shift_y)), int(np.round(shift_x))
        shifted = np.zeros_like(pixels)

        src_y0 = max(0, -iy)
        src_y1 = min(28, 28 - iy)
        src_x0 = max(0, -ix)
        src_x1 = min(28, 28 - ix)
        dst_y0 = max(0, iy)
        dst_y1 = min(28, 28 + iy)
        dst_x0 = max(0, ix)
        dst_x1 = min(28, 28 + ix)

        h = min(src_y1 - src_y0, dst_y1 - dst_y0)
        w = min(src_x1 - src_x0, dst_x1 - dst_x0)

        if h > 0 and w > 0:
            shifted[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = \
                pixels[src_y0:src_y0 + h, src_x0:src_x0 + w]

        pixels = shifted

    # Step 5: Transpose
    pixels = pixels.T

    # Step 6: Normalize
    pixels = (pixels - MEAN) / STD

    # Convert to tensor: (1, 1, 28, 28)
    tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)
