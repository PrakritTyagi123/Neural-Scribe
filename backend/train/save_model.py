"""
save_model.py - Model Saving and Loading Utilities

This module provides functions for saving and loading trained model weights.
Models are saved in PyTorch's standard .pt format.
"""

import os
import torch


# Default path for saving models
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models'
)

DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'digit_model.pt')


def ensure_models_dir():
    """
    Ensure the models directory exists.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, path=None, optimizer=None, epoch=None, loss=None, accuracy=None):
    """
    Save model weights and optional training state.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model (default: models/digit_model.pt)
        optimizer: Optional optimizer state to save
        epoch: Optional current epoch number
        loss: Optional final loss value
        accuracy: Optional accuracy value
        
    Returns:
        str: Path where the model was saved
    """
    if path is None:
        path = DEFAULT_MODEL_PATH
    
    ensure_models_dir()
    
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    # Add optional training info
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    return path


def save_model_weights_only(model, path=None):
    """
    Save only the model weights (lighter file).
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        
    Returns:
        str: Path where the model was saved
    """
    if path is None:
        path = DEFAULT_MODEL_PATH
    
    ensure_models_dir()
    torch.save(model.state_dict(), path)
    
    return path


def load_model(model, path=None, device=None):
    """
    Load model weights from a checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to the checkpoint file
        device: Device to load the model onto
        
    Returns:
        dict: Checkpoint dictionary (may contain training info)
    """
    if path is None:
        path = DEFAULT_MODEL_PATH
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    
    # Handle both full checkpoints and weights-only files
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        checkpoint = {'model_state_dict': checkpoint}
    
    model.to(device)
    
    return checkpoint


def model_exists(path=None):
    """
    Check if a saved model exists.
    
    Args:
        path: Path to check (default: models/digit_model.pt)
        
    Returns:
        bool: True if model file exists
    """
    if path is None:
        path = DEFAULT_MODEL_PATH
    
    return os.path.exists(path)


def get_model_info(path=None):
    """
    Get information about a saved model.
    
    Args:
        path: Path to the model file
        
    Returns:
        dict: Model information (file size, training info if available)
    """
    if path is None:
        path = DEFAULT_MODEL_PATH
    
    if not model_exists(path):
        return None
    
    info = {
        'path': path,
        'size_bytes': os.path.getsize(path),
        'size_mb': os.path.getsize(path) / (1024 * 1024)
    }
    
    # Try to load additional info from checkpoint
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
            if 'loss' in checkpoint:
                info['loss'] = checkpoint['loss']
            if 'accuracy' in checkpoint:
                info['accuracy'] = checkpoint['accuracy']
    except Exception:
        pass
    
    return info


if __name__ == "__main__":
    # Test the save/load functionality
    from model import get_model
    
    print("Testing model save/load utilities...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Default model path: {DEFAULT_MODEL_PATH}")
    
    # Create a test model
    model = get_model()
    
    # Save it
    test_path = os.path.join(MODELS_DIR, 'test_model.pt')
    save_model(model, test_path, epoch=5, loss=0.05, accuracy=0.98)
    print(f"\nSaved test model to: {test_path}")
    
    # Get info
    info = get_model_info(test_path)
    print(f"\nModel info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Load it back
    model2 = get_model()
    checkpoint = load_model(model2, test_path)
    print(f"\nLoaded model from checkpoint")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A')}")
    
    # Clean up
    os.remove(test_path)
    print(f"\nCleaned up test file")