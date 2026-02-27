"""
predictor.py - Inference Engine with Context Modes + TTA

New: Context mode masking
- 'all': All 76 classes active
- 'text': Only digits + Latin letters (classes 0-46)
- 'math': Only digits + math operators + Greek letters

Masking zeros out irrelevant class probabilities before argmax,
keeping accuracy high within each mode.
"""
import torch
import torch.nn.functional as F
import os
import time
import numpy as np
from backend.train.model import DigitCNN
from backend.train.dataset import ALL_LABELS, NUM_CLASSES, MODE_INDICES
from backend.interface.preprocess import preprocess_pixels


class Predictor:
    def __init__(self, model_path='backend/models/digit_model.pt', device=None, use_tta=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitCNN().to(self.device)
        self.model_loaded = False
        self.inference_times = []
        self.saved_history = {'train_loss': [], 'test_loss': [], 'accuracy': []}
        self.use_tta = use_tta

        # Precompute mode masks as tensors for fast GPU masking
        self._mode_masks = {}
        for mode_name, indices in MODE_INDICES.items():
            mask = torch.zeros(NUM_CLASSES, device=self.device)
            for idx in indices:
                mask[idx] = 1.0
            self._mode_masks[mode_name] = mask

        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, path='backend/models/digit_model.pt'):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            accuracy = checkpoint.get('accuracy', 0)
            epoch = checkpoint.get('epoch', 0)
            num_cls = checkpoint.get('num_classes', 47)
            self.saved_history = checkpoint.get('history', {'train_loss': [], 'test_loss': [], 'accuracy': []})
            print(f"Model loaded: epoch {epoch}, accuracy {accuracy:.2f}%, classes {num_cls}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def _create_tta_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create 5-variant TTA batch with milder rotation for symbol safety."""
        device = tensor.device
        variants = [tensor]

        def apply_affine(theta_2x3):
            theta = theta_2x3.unsqueeze(0).to(device)
            grid = F.affine_grid(theta, tensor.size(), align_corners=False)
            return F.grid_sample(tensor, grid, align_corners=False, padding_mode='zeros')

        # ±3° rotation (reduced from ±4° — safer for symbols like + vs ×)
        angle = 3.0 * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        theta_pos = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
        variants.append(apply_affine(theta_pos))

        theta_neg = torch.tensor([[cos_a, sin_a, 0], [-sin_a, cos_a, 0]], dtype=torch.float32)
        variants.append(apply_affine(theta_neg))

        shift = 2.0 / 28.0
        theta_right = torch.tensor([[1, 0, -shift], [0, 1, 0]], dtype=torch.float32)
        variants.append(apply_affine(theta_right))

        theta_down = torch.tensor([[1, 0, 0], [0, 1, -shift]], dtype=torch.float32)
        variants.append(apply_affine(theta_down))

        return torch.cat(variants, dim=0)

    def predict(self, pixel_data: list[float], mode: str = 'all') -> dict:
        """
        Run inference with optional context mode masking.

        Args:
            pixel_data: 784-length pixel array from canvas
            mode: 'all' | 'text' | 'math' — controls which classes are active

        Returns:
            Prediction dict with label, confidence, probabilities, activations
        """
        if not self.model_loaded:
            return {
                'label': '?',
                'class_index': -1,
                'digit': '?',
                'confidence': 0.0,
                'probabilities': [0.0] * NUM_CLASSES,
                'is_digit': False,
                'is_upper': False,
                'is_lower': False,
                'is_math': False,
                'is_greek': False,
                'activations': {},
                'inference_ms': 0.0,
                'mode': mode,
                'error': 'Model not loaded'
            }

        start = time.perf_counter()

        tensor = preprocess_pixels(pixel_data, self.device)

        with torch.no_grad():
            if self.use_tta:
                tta_batch = self._create_tta_batch(tensor)
                logits_batch = self.model(tta_batch)
                probs_batch = F.softmax(logits_batch, dim=1)
                probs = probs_batch.mean(dim=0)
                _ = self.model(tensor)
            else:
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)

        # === Context Mode Masking ===
        # Zero out classes not in the active mode, then renormalize
        if mode in self._mode_masks and mode != 'all':
            mask = self._mode_masks[mode]
            probs = probs * mask
            prob_sum = probs.sum()
            if prob_sum > 1e-6:
                probs = probs / prob_sum

        inference_ms = (time.perf_counter() - start) * 1000
        self.inference_times.append(inference_ms)

        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()
        label = ALL_LABELS[class_idx] if class_idx < len(ALL_LABELS) else '?'

        is_digit = class_idx < 10
        is_upper = 10 <= class_idx <= 35
        is_lower = 36 <= class_idx <= 46
        is_math = 47 <= class_idx <= 58
        is_greek = 59 <= class_idx <= 75

        activations = self.model.get_activations()

        return {
            'label': label,
            'class_index': class_idx,
            'digit': label,
            'confidence': round(confidence * 100, 1),
            'probabilities': [round(p.item() * 100, 1) for p in probs],
            'is_digit': is_digit,
            'is_upper': is_upper,
            'is_lower': is_lower,
            'is_math': is_math,
            'is_greek': is_greek,
            'activations': activations,
            'inference_ms': round(inference_ms, 2),
            'mode': mode,
        }

    def get_avg_inference_ms(self):
        if not self.inference_times:
            return 0.0
        recent = self.inference_times[-50:]
        return round(sum(recent) / len(recent), 2)

    @property
    def fps(self):
        avg = self.get_avg_inference_ms()
        return round(1000 / avg, 1) if avg > 0 else 0.0
