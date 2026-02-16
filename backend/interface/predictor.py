"""
predictor.py - Real-Time Inference Engine
Loads trained EMNIST model and performs prediction with character label mapping.
"""
import torch
import torch.nn.functional as F
import os
import time
from backend.train.model import DigitCNN
from backend.train.dataset import EMNIST_LABELS, NUM_CLASSES
from backend.interface.preprocess import preprocess_pixels


class Predictor:
    def __init__(self, model_path='backend/models/digit_model.pt', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitCNN().to(self.device)
        self.model_loaded = False
        self.inference_times = []
        self.saved_history = {'train_loss': [], 'test_loss': [], 'accuracy': []}

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
            self.saved_history = checkpoint.get('history', {'train_loss': [], 'test_loss': [], 'accuracy': []})
            print(f"Model loaded: epoch {epoch}, accuracy {accuracy:.2f}%")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(self, pixel_data: list[float]) -> dict:
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
                'activations': {},
                'inference_ms': 0.0,
                'error': 'Model not loaded'
            }

        start = time.perf_counter()

        tensor = preprocess_pixels(pixel_data, self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        inference_ms = (time.perf_counter() - start) * 1000
        self.inference_times.append(inference_ms)

        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()
        label = EMNIST_LABELS[class_idx] if class_idx < len(EMNIST_LABELS) else '?'

        is_digit = class_idx < 10
        is_upper = 10 <= class_idx <= 35
        is_lower = class_idx >= 36

        activations = self.model.get_activations()

        return {
            'label': label,
            'class_index': class_idx,
            'digit': label,  # backward compat
            'confidence': round(confidence * 100, 1),
            'probabilities': [round(p.item() * 100, 1) for p in probs],
            'is_digit': is_digit,
            'is_upper': is_upper,
            'is_lower': is_lower,
            'activations': activations,
            'inference_ms': round(inference_ms, 2),
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