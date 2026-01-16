"""
Training script for Video Tokenizer.

This script trains the video tokenizer to encode video frames into discrete
tokens and decode them back with minimal reconstruction loss.

The training objective is simple: minimize the MSE between input frames
and reconstructed frames. The FSQ quantization is handled via straight-through
estimator (STE), so gradients flow through the quantization step.

Usage:
    # Train with dummy data (for testing)
    uv run python train.py --use-dummy-data

    # Train with real video data
    uv run python train.py --data-path /path/to/videos --data-type folder

    # Train with custom settings
    uv run python train.py --batch-size 16 --learning-rate 5e-4 --num-epochs 50
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import Config, get_config
from data_utils import get_dataloader, split_dataset, DummyVideoDataset
from models import VideoTokenizer


def get_lr(step: int, config: Config) -> float:
    """
    Compute learning rate with warmup and cosine decay.

    Learning rate schedule:
    1. Linear warmup: 0 -> lr over warmup_steps
    2. Cosine decay: lr -> min_lr over remaining steps

    Args:
        step: Current training step
        config: Configuration object

    Returns:
        Current learning rate
    """
    warmup_steps = config.training.warmup_steps
    lr = config.training.learning_rate
    min_lr = config.training.min_lr

    # Warmup phase: linear increase
    if step < warmup_steps:
        return lr * step / warmup_steps

    # After warmup: cosine decay
    # We don't have total steps, so decay based on progress
    # For simplicity, use epoch-based decay in the training loop
    return lr


def cosine_decay(lr: float, min_lr: float, progress: float) -> float:
    """
    Compute cosine-decayed learning rate.

    Args:
        lr: Peak learning rate
        min_lr: Minimum learning rate
        progress: Training progress in [0, 1]

    Returns:
        Decayed learning rate
    """
    # Cosine decay from lr to min_lr
    # progress=0 -> lr, progress=1 -> min_lr
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * decay


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    epoch: int,
    global_step: int,
) -> tuple:
    """
    Train for one epoch.

    Args:
        model: VideoTokenizer model
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        config: Configuration
        epoch: Current epoch number
        global_step: Global step counter

    Returns:
        (average_loss, global_step, codebook_usage)
    """
    model.train()
    device = config.device

    total_loss = 0.0
    num_batches = 0
    all_indices = []

    # Timing
    last_log_time = time.time()
    last_log_step = global_step
    steps_per_epoch = len(dataloader)
    total_steps = config.training.num_epochs * steps_per_epoch

    # Determine device type for autocast
    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = config.use_amp and device_type == "cuda"  # AMP only on CUDA

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        # batch shape: (B, T, C, H, W)
        x = batch.to(device)

        # Forward pass with mixed precision (only on CUDA)
        with autocast(device_type=device_type, enabled=use_amp):
            loss, x_hat, indices = model(x)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.max_grad_norm,
        )

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        all_indices.append(indices.detach().cpu())
        global_step += 1

        # Update learning rate per step (warmup + cosine decay)
        if global_step < config.training.warmup_steps:
            # Linear warmup: 0 -> lr
            lr = config.training.learning_rate * global_step / config.training.warmup_steps
        else:
            # Cosine decay after warmup
            progress = (global_step - config.training.warmup_steps) / max(1, total_steps - config.training.warmup_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            lr = config.training.min_lr + (config.training.learning_rate - config.training.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Logging
        if global_step % config.training.log_interval == 0:
            avg_loss = total_loss / num_batches

            # Calculate timing
            now = time.time()
            steps_since_log = global_step - last_log_step
            time_per_interval = now - last_log_time
            steps_per_sec = steps_since_log / time_per_interval if time_per_interval > 0 else 0

            # Estimated time remaining
            remaining_steps = total_steps - global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_hours = eta_seconds / 3600

            if eta_hours >= 1:
                eta_str = f"{eta_hours:.1f}h"
            else:
                eta_str = f"{eta_minutes:.1f}m"

            print(
                f"  Step {global_step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"{time_per_interval:.1f}s/100 | "
                f"ETA: {eta_str}"
            )

            # Update timing trackers
            last_log_time = now
            last_log_step = global_step

    # Compute codebook usage
    all_indices = torch.cat(all_indices, dim=0).flatten()
    usage = torch.bincount(all_indices, minlength=model.codebook_size)
    num_used = (usage > 0).sum().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step, (num_used, model.codebook_size)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Config,
) -> float:
    """
    Validate the model.

    Args:
        model: VideoTokenizer model
        dataloader: Validation data loader
        config: Configuration

    Returns:
        Average validation loss
    """
    model.eval()
    device = config.device

    # Determine device type for autocast
    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = config.use_amp and device_type == "cuda"

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)

            with autocast(device_type=device_type, enabled=use_amp):
                loss, _, _ = model(x)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    config: Config,
    path: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["epoch"],
        checkpoint["global_step"],
        checkpoint.get("loss", 0.0),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Video Tokenizer")

    # Data arguments
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data for testing",
    )
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument(
        "--data-type",
        type=str,
        default="folder",
        choices=["folder", "images"],
        help="Type of data to load",
    )

    # Model arguments
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4)

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)

    # Other arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create configuration
    config = get_config(
        model_frame_size=args.frame_size,
        model_num_frames=args.num_frames,
        model_patch_size=args.patch_size,
        model_embed_dim=args.embed_dim,
        model_num_heads=args.num_heads,
        model_num_blocks=args.num_blocks,
        training_batch_size=args.batch_size,
        training_learning_rate=args.learning_rate,
        training_num_epochs=args.num_epochs,
    )

    # Setup data
    if args.use_dummy_data:
        data_type = "dummy"
        data_path = None
        print("Using dummy data for training")
    else:
        data_type = args.data_type
        data_path = args.data_path
        if data_path is None:
            print("Error: --data-path is required when not using --use-dummy-data")
            sys.exit(1)
        print(f"Using {data_type} data from {data_path}")

    # Create data loaders
    print("\nSetting up data...")

    if data_type == "dummy":
        # Create train and val dummy datasets
        train_dataset = DummyVideoDataset(
            num_samples=800,
            num_frames=config.model.num_frames,
            frame_size=config.model.frame_size,
            seed=args.seed,
        )
        val_dataset = DummyVideoDataset(
            num_samples=200,
            num_frames=config.model.num_frames,
            frame_size=config.model.frame_size,
            seed=args.seed + 1,
        )
    else:
        # Load real data and split
        full_loader = get_dataloader(
            data_type=data_type,
            data_path=data_path,
            num_frames=config.model.num_frames,
            frame_size=config.model.frame_size,
            batch_size=1,  # Will create new loaders after split
        )
        train_dataset, val_dataset = split_dataset(
            full_loader.dataset,
            train_ratio=0.9,
            seed=args.seed,
        )

    # Create dataloaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = VideoTokenizer(
        in_channels=config.model.in_channels,
        frame_size=config.model.frame_size,
        num_frames=config.model.num_frames,
        patch_size=config.model.patch_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_blocks=config.model.num_blocks,
        latent_dim=config.model.latent_dim,
        num_bins=config.model.num_bins,
        dropout=config.model.dropout,
    )
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Codebook size: {model.codebook_size}")
    print(f"Device: {config.device}")

    # Create optimizer
    # Separate weight decay for different parameter types
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or "scale" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=config.training.betas,
    )

    # Gradient scaler for mixed precision (only effective on CUDA)
    use_amp = config.use_amp and config.device == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, global_step, _ = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
        )
        start_epoch += 1  # Start from next epoch

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start = time.time()

        # Compute learning rate with cosine decay
        progress = epoch / config.training.num_epochs
        lr = cosine_decay(
            config.training.learning_rate,
            config.training.min_lr,
            progress,
        )

        # Apply warmup if early in training
        if global_step < config.training.warmup_steps:
            lr = config.training.learning_rate * global_step / config.training.warmup_steps

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs} (lr={lr:.6f})")
        print("-" * 40)

        # Train
        train_loss, global_step, codebook_usage = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )

        # Validate
        val_loss = validate(model, val_loader, config)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        used, total = codebook_usage
        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Codebook: {used}/{total} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or val_loss < best_val_loss:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=val_loss,
                config=config,
                path=checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt",
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=val_loss,
                config=config,
                path=checkpoint_dir / "best_model.pt",
            )
            print("  New best model!")

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.num_epochs - 1,
        global_step=global_step,
        loss=val_loss,
        config=config,
        path=checkpoint_dir / "final_model.pt",
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
