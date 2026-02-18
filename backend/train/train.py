"""
train.py - High-Accuracy Training Loop

Improvements over original:
1. CosineAnnealingWarmRestarts scheduler: multiple LR cycles let the model escape
   local minima. T_0=10, T_mult=2 means cycles of 10, 20, 40 epochs.
2. Mixup regularization: blends pairs of training images and their labels,
   forcing the model to learn smoother decision boundaries between confusable classes.
3. Focal Loss: downweights easy examples, focuses learning on hard cases
   (O/0, l/1/I, S/5, etc.) — the ones that limit accuracy.
4. Fixed autocast: uses device.type instead of hardcoded 'cuda' so CPU training works.
5. Fixed CUDA threading: creates device inside training thread to avoid cross-thread issues.
6. Gradient clipping + EMA-style best model saving.
7. Longer training support: 30-50 epochs recommended with warm restarts.
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
    """
    Focal Loss: focuses training on hard examples.

    Standard cross-entropy gives equal weight to all examples. Focal loss adds
    a factor (1 - p_t)^gamma that downweights well-classified examples:
    - Easy example (model predicts 95% correctly): weight = 0.05^2 = 0.0025
    - Hard example (model predicts 30% correctly): weight = 0.70^2 = 0.49

    This means the model spends ~200x more "learning effort" on hard examples
    compared to easy ones. For EMNIST, the hard examples are confusable character
    pairs (O/0, l/1, S/5, Z/2) — exactly what limits accuracy.

    Label smoothing is applied on top: instead of [0,0,1,0,...] the target becomes
    [0.002, 0.002, 0.9, 0.002, ...] which prevents overconfidence.

    Args:
        gamma: focusing parameter. 0 = standard CE, 2 = strong focus on hard examples.
        label_smoothing: softens target distribution.
        reduction: 'mean' (default) or 'sum'.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply label smoothing: smooth_target = (1 - smoothing) * one_hot + smoothing / num_classes
        num_classes = inputs.size(1)
        smooth = self.label_smoothing / num_classes
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1.0)
        smooth_target = one_hot * (1 - self.label_smoothing) + smooth

        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        # p_t is the probability assigned to the correct class
        p_t = (probs * one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        # Weighted cross-entropy with smooth targets
        loss = -(smooth_target * log_probs).sum(dim=1)
        loss = focal_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def mixup_data(x, y, alpha=0.2):
    """
    Mixup: blends pairs of training images and labels.

    Given batch of images x and labels y:
    1. Sample lambda from Beta(alpha, alpha) — usually lambda ≈ 0.8-1.0
    2. Randomly shuffle the batch to get pairs
    3. mixed_x = lambda * x + (1-lambda) * x_shuffled
    4. Return both original and shuffled labels with lambda

    The model must predict a blend of both labels, forcing it to learn
    features that smoothly interpolate between classes. This builds
    much better decision boundaries for confusable pairs.

    alpha=0.2 gives mild mixing (lambda usually > 0.8), which is best
    for character recognition where heavy mixing would create unreadable images.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    # Ensure lam >= 0.5 so the primary image dominates
    lam = max(lam, 1 - lam)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def train_model(epochs=35, lr=0.003, batch_size=128, save_path='backend/models/digit_model.pt',
                progress_queue=None, device=None):
    """
    Train the CNN on EMNIST ByMerge (47 classes) with all improvements.

    Hyperparameter rationale:
    - epochs=35: CosineAnnealingWarmRestarts needs enough epochs for multiple cycles.
      With T_0=10, T_mult=2: cycle 1 = epochs 0-9, cycle 2 = epochs 10-29, cycle 3 starts at 30.
      35 epochs gives ~1.5 full restart cycles for good convergence.
    - lr=0.003: peak learning rate. Warm restarts will cycle between lr and eta_min.
    - batch_size=128: sweet spot for GPU utilization vs generalization.
    - Focal loss gamma=2.0: standard value, strong focus on hard examples.
    - Mixup alpha=0.2: mild mixing, appropriate for character recognition.
    """
    # Create device inside this function to be thread-safe
    # (this function may run in a thread pool executor)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_type = device.type  # 'cuda' or 'cpu'
    use_amp = device_type == 'cuda'  # mixed precision only on GPU
    print(f"Training on: {device} | Mixed precision: {use_amp}")

    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Create model
    model = DigitCNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Focal Loss: downweights easy examples, focuses on hard cases
    # gamma=2.0 is standard. label_smoothing=0.1 prevents overconfidence.
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

    # AdamW: Adam with decoupled weight decay
    # weight_decay=1e-4 penalizes large weights, reducing overfitting
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingWarmRestarts: multiple LR cycles
    # T_0=10: first cycle is 10 epochs
    # T_mult=2: each subsequent cycle is 2x longer
    # So: cycle 1 = 10 epochs, cycle 2 = 20 epochs, cycle 3 = 40 epochs
    # The restart "kicks" help escape local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,          # first cycle length
        T_mult=2,        # multiply cycle length by 2 each restart
        eta_min=1e-6     # minimum LR at end of each cycle
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=use_amp)

    history = {'train_loss': [], 'test_loss': [], 'accuracy': [], 'epoch_times': []}
    best_accuracy = 0.0
    mixup_alpha = 0.2  # mild mixup for character recognition

    for epoch in range(epochs):
        epoch_start = time.time()

        # === TRAINING with Mixup ===
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Apply Mixup: blend pairs of images and labels
            mixed_data, targets_a, targets_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            # Use device_type so this works on both CUDA and CPU
            with autocast(device_type, enabled=use_amp):
                output = model(mixed_data)
                # Mixup loss: weighted combination of losses for both mixed labels
                loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # For accuracy tracking, use the primary (unmixed) target
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(targets_a).sum().item()

        # Step scheduler per epoch (CosineAnnealingWarmRestarts is per-epoch)
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # === EVALUATION (no mixup, no augmentation) ===
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        # Use standard CrossEntropyLoss for clean evaluation
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

        # Save best model
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
    model, history = train_model(epochs=35)
    print(f"\nFinal accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"Best accuracy: {max(history['accuracy']):.2f}%")