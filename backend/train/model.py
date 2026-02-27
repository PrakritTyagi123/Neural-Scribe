"""
model.py - CNN for EMNIST + Math/Greek Symbol Recognition (76 classes)

Expanded from 47 → 76 classes with wider channels to handle:
- Original EMNIST characters (digits, letters)
- Math operators (+, −, ×, ÷, =, √, etc.)
- Greek letters (α, β, γ, δ, θ, π, σ, ω, etc.)
- Special symbols (∞, ∑, ∫)

Architecture changes vs 47-class version:
- Wider block3: 160→384 (was 160→320) for more feature capacity
- Wider block4: 384→320 (was 320→256) to preserve more features
- FC: 320→76 (was 256→47)
- ~3.8M parameters (was ~3.3M)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.train.dataset import NUM_CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention mechanism."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DropPath(nn.Module):
    """Stochastic depth: randomly drops entire residual blocks during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class ResBlock(nn.Module):
    """Residual block with BatchNorm, SE attention, and optional stochastic depth."""
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 28x28 → 14x14
        self.block1 = ResBlock(32, 64, stride=2, drop_path=0.05)
        # 14x14 → 7x7
        self.block2 = ResBlock(64, 160, stride=2, drop_path=0.10)
        # 7x7 → 7x7 — wider for 76 classes
        self.block3 = ResBlock(160, 384, stride=1, drop_path=0.15)
        # 7x7 → 7x7 — refinement
        self.block4 = ResBlock(384, 320, stride=1, drop_path=0.20)

        # Global average pool → classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(320, NUM_CLASSES)

        self._activations = {}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stem(x)

        x = self.block1(x)
        self._activations['conv1'] = x.detach()

        x = self.block2(x)
        self._activations['conv2'] = x.detach()

        x = self.block3(x)
        self._activations['fc1'] = x.detach()

        x = self.block4(x)
        self._activations['fc2'] = x.detach()

        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
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
