"""
model.py - CNN Neural Network with Live Activation Extraction
Optimized for MNIST with hooks to extract layer activations for real-time visualization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    CNN Architecture:
      Conv1(1→16, 3x3) → ReLU → MaxPool
      Conv2(16→32, 3x3) → ReLU → MaxPool
      FC1(32*5*5 → 128) → ReLU → Dropout
      FC2(128 → 10)
    
    ~99.2% accuracy on MNIST in ~10 epochs
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Store activations for visualization
        self._activations = {}

    def forward(self, x):
        # Conv Block 1: (B, 1, 28, 28) → (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        self._activations['conv1'] = x.detach()

        # Conv Block 2: (B, 16, 14, 14) → (B, 32, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        self._activations['conv2'] = x.detach()

        # Flatten: (B, 32, 7, 7) → (B, 1568)
        x = x.view(x.size(0), -1)

        # FC Block: (B, 1568) → (B, 128) → (B, 10)
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
                # Conv layers: average across spatial dims
                a = a.mean(dim=(1, 2))
            # Normalize to [0, 1]
            a_min, a_max = a.min(), a.max()
            if a_max - a_min > 1e-6:
                a = (a - a_min) / (a_max - a_min)
            else:
                a = torch.zeros_like(a)
            result[name] = a.cpu().tolist()
        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
