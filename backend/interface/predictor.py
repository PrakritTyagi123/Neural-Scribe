"""
predictor.py - Model Inference for Digit Recognition

This module handles loading the trained model and performing
fast forward inference to predict digits from preprocessed images.

Usage:
    from predictor import DigitPredictor
    
    predictor = DigitPredictor()
    result = predictor.predict(pixel_array)
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.train.model import get_model
from backend.train.save_model import load_model, model_exists, DEFAULT_MODEL_PATH
from backend.interface.preprocess import preprocess_canvas_data, preprocess_pixel_array


class DigitPredictor:
    """
    Handles digit prediction using the trained model.
    
    Attributes:
        model: The loaded neural network
        device: CPU or CUDA device
        is_loaded: Whether the model is loaded successfully
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor and load the model.
        
        Args:
            model_path: Path to the saved model weights
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        if not model_exists(self.model_path):
            print(f"⚠️ Model not found at {self.model_path}")
            print("   Run 'python -m backend.train.train' to train the model first.")
            return
        
        try:
            self.model = get_model()
            load_model(self.model, self.model_path, self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.is_loaded = False
    
    def predict(self, pixel_data, preprocess=True):
        """
        Predict a digit from pixel data.
        
        Args:
            pixel_data: Raw pixel array (784 or 28x28 values, 0-255)
                       Or pre-processed tensor
            preprocess: Whether to apply preprocessing
            
        Returns:
            dict: {
                'success': bool,
                'digit': int (0-9),
                'confidence': float (0-1),
                'probabilities': list of 10 floats
            }
        """
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Model not loaded. Train the model first.',
                'digit': -1,
                'confidence': 0.0,
                'probabilities': [0.0] * 10
            }
        
        try:
            # Preprocess if needed
            # Canvas.js already inverts (black drawing -> white digit on black bg)
            if preprocess:
                tensor = preprocess_canvas_data(pixel_data, invert=False, center=True)
            else:
                tensor = pixel_data
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Get prediction
                confidence, predicted = probabilities.max(1)
                
            # Convert to Python types
            probs_list = probabilities.squeeze().cpu().numpy().tolist()
            
            return {
                'success': True,
                'digit': int(predicted.item()),
                'confidence': float(confidence.item()),
                'probabilities': probs_list
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'digit': -1,
                'confidence': 0.0,
                'probabilities': [0.0] * 10
            }
    
    def predict_batch(self, batch_data):
        """
        Predict multiple digits at once.
        
        Args:
            batch_data: List of pixel arrays
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(data) for data in batch_data]
    
    def get_top_k(self, pixel_data, k=3, preprocess=True):
        """
        Get top-k predictions with their probabilities.
        
        Args:
            pixel_data: Raw pixel array
            k: Number of top predictions to return
            preprocess: Whether to apply preprocessing
            
        Returns:
            dict: {
                'success': bool,
                'predictions': [(digit, probability), ...]
            }
        """
        result = self.predict(pixel_data, preprocess)
        
        if not result['success']:
            return result
        
        # Get top-k
        probs = result['probabilities']
        indexed_probs = [(i, p) for i, p in enumerate(probs)]
        top_k = sorted(indexed_probs, key=lambda x: x[1], reverse=True)[:k]
        
        return {
            'success': True,
            'predictions': top_k,
            'probabilities': result['probabilities']
        }


# Global predictor instance (lazy loading)
_predictor = None


def get_predictor():
    """
    Get the global predictor instance.
    
    Returns:
        DigitPredictor: Singleton predictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = DigitPredictor()
    return _predictor


def predict_digit(pixel_data):
    """
    Convenience function for quick predictions.
    
    Args:
        pixel_data: Raw pixel array
        
    Returns:
        dict: Prediction result
    """
    predictor = get_predictor()
    return predictor.predict(pixel_data)


if __name__ == "__main__":
    import numpy as np
    
    print("Testing Digit Predictor...")
    print("-" * 40)
    
    # Initialize predictor
    predictor = DigitPredictor()
    
    if not predictor.is_loaded:
        print("\n⚠️ Cannot test without trained model.")
        print("   Run: python -m backend.train.train")
        sys.exit(1)
    
    # Create a simple test pattern (vertical line like "1")
    test_image = np.zeros((28, 28), dtype=np.float32)
    test_image[5:23, 13:16] = 255  # Vertical line
    
    # Make prediction
    result = predictor.predict(test_image)
    
    print(f"\nTest prediction:")
    print(f"  Predicted digit: {result['digit']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\n  All probabilities:")
    for digit, prob in enumerate(result['probabilities']):
        bar = '█' * int(prob * 30)
        print(f"    {digit}: {prob:.4f} {bar}")
    
    # Test top-k
    top_k = predictor.get_top_k(test_image, k=3)
    print(f"\n  Top 3 predictions:")
    for digit, prob in top_k['predictions']:
        print(f"    {digit}: {prob:.2%}")
    
    print("\n✓ Predictor tests complete!")