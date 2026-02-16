"""
train.py - Training Loop with Live Progress
Trains the CNN model and puts epoch updates into a thread-safe queue
so the async server can broadcast them in real time.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from backend.train.model import DigitCNN
from backend.train.dataset import get_data_loaders


def train_model(epochs=15, lr=0.001, batch_size=64, save_path='backend/models/digit_model.pt',
                progress_queue=None, device=None):
    """
    Train the CNN model on MNIST.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        save_path: Path to save the trained model
        progress_queue: thread-safe queue.Queue — push epoch dicts into it
        device: torch device (auto-detected if None)
    
    Returns:
        (model, history_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on: {device}")

    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Create model
    model = DigitCNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        'train_loss': [],
        'test_loss': [],
        'accuracy': [],
        'epoch_times': []
    }

    best_accuracy = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        # === Training Phase ===
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # === Evaluation Phase ===
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / total
        epoch_time = time.time() - epoch_start

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['accuracy'].append(accuracy)
        history['epoch_times'].append(epoch_time)

        # Save best model + history
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch + 1,
                'history': {
                    'train_loss': history['train_loss'][:],
                    'test_loss': history['test_loss'][:],
                    'accuracy': history['accuracy'][:],
                },
            }, save_path)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Push update into queue (thread-safe, non-blocking)
        if progress_queue is not None:
            update = {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'train_loss': round(train_loss, 4),
                'test_loss': round(test_loss, 4),
                'accuracy': round(accuracy, 2),
                'train_accuracy': round(train_acc, 2),
                'epoch_time': round(epoch_time, 1),
                'best_accuracy': round(best_accuracy, 2),
                'lr': optimizer.param_groups[0]['lr'],
            }
            progress_queue.put(update)

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    return model, history


if __name__ == '__main__':
    model, history = train_model(epochs=15)
    print(f"\nFinal accuracy: {history['accuracy'][-1]:.2f}%")
