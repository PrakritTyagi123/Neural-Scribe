"""
model.py - Neural Network Architecture for Digit Recognition

Defines a fully connected neural network that learns to recognize
handwritten digits from MNIST images.

Architecture:
    Input (784) → Hidden1 (128) → Hidden2 (64) → Output (10)
    
The network takes flattened 28×28 images (784 pixels) and outputs
probability scores for digits 0-9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitRecognizer(nn.Module):
    """
    A fully connected neural network for digit classification.
    
    Layers:
        - fc1: 784 → 128 (input to first hidden)
        - fc2: 128 → 64 (first to second hidden)
        - fc3: 64 → 10 (second hidden to output)
    
    Activation: ReLU between hidden layers
    Output: Raw logits (use softmax for probabilities)
    """
    
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(784, 128)   # Input layer
        self.fc2 = nn.Linear(128, 64)    # Hidden layer
        self.fc3 = nn.Linear(64, 10)     # Output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            Output tensor of shape (batch_size, 10) with raw logits
        """
        # Flatten the input if needed (batch_size, 1, 28, 28) → (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (raw logits)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Make a prediction with probability scores.
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (predicted_digit, probability_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
        return predicted, probabilities


def get_model():
    """
    Factory function to create a new model instance.
    
    Returns:
        DigitRecognizer: A new untrained model
    """
    return DigitRecognizer()


def count_parameters(model):
    """
    Count the total number of trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = get_model()
    print(f"Model Architecture:\n{model}")
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    pred, probs = model.predict(dummy_input)
    print(f"Predicted digit: {pred.item()}")
    print(f"Confidence: {probs.max().item():.2%}")