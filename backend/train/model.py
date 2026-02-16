"""
model.py - CNN Neural Network with Live Activation Extraction
Supports EMNIST ByMerge (47 classes: digits + letters).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.train.dataset import NUM_CLASSES


class DigitCNN(nn.Module):
    """
    CNN Architecture:
      Conv1(1→32, 3x3) → ReLU → MaxPool
      Conv2(32→64, 3x3) → ReLU → MaxPool
      FC1(64*7*7 → 256) → ReLU → Dropout
      FC2(256 → 47)

    Wider than the MNIST-only version to handle 47 classes.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

        # Store activations for visualization
        self._activations = {}

    def forward(self, x):
        # Conv Block 1: (B, 1, 28, 28) → (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        self._activations['conv1'] = x.detach()

        # Conv Block 2: (B, 32, 14, 14) → (B, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        self._activations['conv2'] = x.detach()

        # Flatten: (B, 64, 7, 7) → (B, 3136)
        x = x.view(x.size(0), -1)

        # FC Block: (B, 3136) → (B, 256) → (B, 47)
        x = F.relu(self.fc1(x))
        self._activations['fc1'] = x.detach()
        x = self.dropout(x)

        x = self.fc2(x)
        self._activations['output'] = x.detach()

        return x

    def get_activations(self):
        """Return normalized activations for visualization."""
        result = {}
        for name, act in self._activations.items():
            a = act.squeeze(0)
            if a.dim() == 3:
                a = a.mean(dim=(1, 2))
            a_min, a_max = a.min(), a.max()
            if a_max - a_min > 1e-6:
                a = (a - a_min) / (a_max - a_min)
            else:
                a = torch.zeros_like(a)
            result[name] = a.cpu().tolist()
        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)