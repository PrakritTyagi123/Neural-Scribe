"""
train.py - Training Script for Digit Recognition Model

This script handles the complete training pipeline:
    1. Load MNIST training and test data
    2. Initialize the neural network model
    3. Train using backpropagation with Adam optimizer
    4. Evaluate on test set
    5. Save the trained weights

Usage:
    python -m backend.train.train
    
    Or from project root:
    python backend/train/train.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.train.model import get_model, count_parameters
from backend.train.dataset import get_data_loaders
from backend.train.save_model import save_model, DEFAULT_MODEL_PATH


# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress indicator
        if (batch_idx + 1) % 200 == 0:
            print(f'    Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Complete training pipeline.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        tuple: (trained_model, final_accuracy)
    """
    print("=" * 60)
    print("  DIGIT RECOGNIZER - TRAINING")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load data
    print("\n📊 Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size)
    print(f"   Training samples: {len(train_loader.dataset):,}")
    print(f"   Test samples: {len(test_loader.dataset):,}")
    print(f"   Batch size: {batch_size}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = get_model().to(device)
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    print(f"\n🏋️ Training for {epochs} epochs...")
    print("-" * 60)
    
    best_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch:2d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Track best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
    
    total_time = time.time() - start_time
    
    print("-" * 60)
    print(f"\n✅ Training complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Best test accuracy: {best_accuracy:.2f}%")
    
    # Save model
    print(f"\n💾 Saving model to: {DEFAULT_MODEL_PATH}")
    save_model(
        model, 
        epoch=epochs, 
        loss=test_loss, 
        accuracy=test_acc,
        optimizer=optimizer
    )
    print("   Model saved successfully!")
    
    return model, best_accuracy


def quick_test(model, test_loader, device):
    """
    Quick test with sample predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:5].to(device), labels[:5]
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = outputs.max(1)
    
    print("\n🔍 Sample predictions:")
    print("   " + "-" * 40)
    for i in range(5):
        status = "✓" if predictions[i] == labels[i] else "✗"
        print(f"   {status} Predicted: {predictions[i].item()}, "
              f"Actual: {labels[i].item()}")


if __name__ == "__main__":
    # Run training
    model, accuracy = train_model()
    
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_data_loaders()
    quick_test(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("  Training complete! Run 'python run_backend.py' to start the server.")
    print("=" * 60)