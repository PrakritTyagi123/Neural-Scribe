"""
train.py - Training Loop for 76-Class Recognition

Supports combined EMNIST + synthetic symbol training.
Key additions:
- include_symbols flag to control whether math/Greek symbols are trained
- Higher mixup_alpha (0.3) for better boundary smoothing with more confusable pairs
- Focal Loss gamma=2.0 remains — even more important with Greek/Latin confusions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time
import os
import numpy as np
from backend.train.model import DigitCNN
from backend.train.dataset import get_data_loaders, NUM_CLASSES


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing for hard example mining."""
    def __init__(self, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        smooth = self.label_smoothing / num_classes
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1.0)
        smooth_target = one_hot * (1 - self.label_smoothing) + smooth

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        p_t = (probs * one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        loss = -(smooth_target * log_probs).sum(dim=1)
        loss = focal_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def mixup_data(x, y, alpha=0.2):
    """Mixup regularization: blends pairs of training images and labels."""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def train_model(epochs=35, lr=0.003, batch_size=128, save_path='backend/models/digit_model.pt',
                progress_queue=None, device=None, include_symbols=True):
    """
    Train on EMNIST + optional math/Greek symbols (76 classes total).

    Args:
        epochs: Number of training epochs
        lr: Peak learning rate
        batch_size: Batch size
        save_path: Where to save best model
        progress_queue: Queue for sending updates to UI
        device: torch device
        include_symbols: If True, include synthetic math/Greek symbol data
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_type = device.type
    use_amp = device_type == 'cuda'
    print(f"Training on: {device} | Mixed precision: {use_amp}")
    print(f"Include symbols: {include_symbols} | Total classes: {NUM_CLASSES}")

    # Load data
    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        include_symbols=include_symbols
    )
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Create model
    model = DigitCNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    scaler = GradScaler(enabled=use_amp)

    history = {'train_loss': [], 'test_loss': [], 'accuracy': [], 'epoch_times': []}
    best_accuracy = 0.0
    # Slightly stronger mixup for 76 classes (more confusable pairs)
    mixup_alpha = 0.3 if include_symbols else 0.2

    for epoch in range(epochs):
        epoch_start = time.time()

        # === TRAINING ===
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            mixed_data, targets_a, targets_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type, enabled=use_amp):
                output = model(mixed_data)
                loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(targets_a).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # === EVALUATION ===
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        eval_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with autocast(device_type, enabled=use_amp):
                    output = model(data)
                    test_loss += eval_criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / total
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['accuracy'].append(accuracy)
        history['epoch_times'].append(epoch_time)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch + 1,
                'num_classes': NUM_CLASSES,
                'include_symbols': include_symbols,
                'history': {
                    'train_loss': history['train_loss'][:],
                    'test_loss': history['test_loss'][:],
                    'accuracy': history['accuracy'][:],
                },
            }, save_path)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Loss: {train_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:.1f}%/{accuracy:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"Best: {best_accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        if progress_queue is not None:
            progress_queue.put({
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'train_loss': round(train_loss, 4),
                'test_loss': round(test_loss, 4),
                'accuracy': round(accuracy, 2),
                'train_accuracy': round(train_acc, 2),
                'epoch_time': round(epoch_time, 1),
                'best_accuracy': round(best_accuracy, 2),
                'lr': current_lr,
            })

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    return model, history


if __name__ == '__main__':
    model, history = train_model(epochs=35, include_symbols=True)
    print(f"\nFinal accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"Best accuracy: {max(history['accuracy']):.2f}%")
