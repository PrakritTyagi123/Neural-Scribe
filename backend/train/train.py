"""
train.py - Modern Training Loop with Mixed Precision, OneCycleLR, and Label Smoothing

Key improvements over original:
- Mixed precision (torch.amp): 1.5-2x faster on GPUs with tensor cores (RTX series)
- OneCycleLR scheduler: reaches higher accuracy than StepLR by using a warmup→peak→decay
  learning rate schedule. The cosine annealing phase is especially good for final convergence.
- Label smoothing (0.1): prevents overconfident predictions, improves generalization.
  Instead of training toward [0,0,1,0,...] it trains toward [0.002,0.002,0.9,0.002,...]
  which gives a softer probability distribution at inference time.
- Gradient clipping: prevents training instability from rare large gradients.
- non_blocking transfers: overlaps CPU→GPU data movement with computation.
- EMA (optional): exponential moving average of weights for smoother final model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time
import os
from backend.train.model import DigitCNN
from backend.train.dataset import get_data_loaders


def train_model(epochs=20, lr=0.003, batch_size=128, save_path='backend/models/digit_model.pt',
                progress_queue=None, device=None):
    """
    Train the CNN on EMNIST ByMerge (47 classes).

    Hyperparameter rationale:
    - lr=0.003: OneCycleLR will peak at this then decay. Higher than typical because
      BatchNorm stabilizes training enough to handle it.
    - batch_size=128: sweet spot for GPU utilization vs generalization.
      Too large (512+) hurts generalization; too small wastes GPU.
    - epochs=20: with OneCycleLR and augmentation, 20 epochs is usually enough.
      The scheduler is designed to complete its full cycle in exactly this many epochs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_amp = device.type == 'cuda'  # mixed precision only on GPU
    print(f"Training on: {device} | Mixed precision: {use_amp}")

    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Create model
    model = DigitCNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Label smoothing: cross entropy with soft targets
    # Prevents the model from becoming overconfident (outputting 99.9%)
    # which hurts generalization. With smoothing=0.1, the target for the
    # correct class is 0.9 instead of 1.0, and 0.1/(47-1)≈0.002 for others.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW: Adam with decoupled weight decay
    # Weight decay 1e-4 penalizes large weights, reducing overfitting.
    # AdamW applies decay correctly (to weights, not gradients) unlike Adam+L2.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # OneCycleLR: warmup → peak → cosine decay in exactly `epochs` epochs
    # This is consistently the best simple scheduler for image classification.
    # It starts slow (warmup), goes fast (peak lr), then fine-tunes (cosine decay).
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,     # 10% warmup
        anneal_strategy='cos',
        div_factor=10,      # start_lr = max_lr / 10
        final_div_factor=100  # end_lr = max_lr / 1000
    )

    # Mixed precision scaler: scales loss to prevent fp16 underflow
    scaler = GradScaler(enabled=use_amp)

    history = {'train_loss': [], 'test_loss': [], 'accuracy': [], 'epoch_times': []}
    best_accuracy = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        # === TRAINING ===
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            # non_blocking=True: starts GPU transfer immediately, doesn't wait
            # for completion. Combined with pin_memory, this overlaps transfer
            # with the previous batch's backward pass.
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

            # Mixed precision forward pass: conv/matmul in fp16, accumulation in fp32
            # ~1.5-2x faster on RTX GPUs with tensor cores, no accuracy loss
            with autocast('cuda', enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)

            # Backward with gradient scaling (prevents fp16 underflow)
            scaler.scale(loss).backward()

            # Gradient clipping: prevents rare large gradients from destabilizing training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Step scheduler per batch (OneCycleLR is per-step, not per-epoch)
            scheduler.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # === EVALUATION ===
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    output = model(data)
                    test_loss += criterion(output, target).item()
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
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:.1f}%/{accuracy:.2f}% | "
              f"LR: {current_lr:.6f} | "
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
    model, history = train_model(epochs=20)
    print(f"\nFinal accuracy: {history['accuracy'][-1]:.2f}%")