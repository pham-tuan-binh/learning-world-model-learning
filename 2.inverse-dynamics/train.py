"""
Training script for Latent Action Model (Inverse Dynamics).

This script trains the model to:
1. Infer latent actions from consecutive frame pairs
2. Use those actions to reconstruct the next frame

The training objective is reconstruction loss + variance penalty.

Usage:
    # Train with dummy data (for testing)
    uv run python train.py --use-dummy-data

    # Train with real video data
    uv run python train.py --data-path /path/to/videos --data-type folder

    # Train with custom settings
    uv run python train.py --n-actions 16 --embed-dim 256 --num-epochs 100
"""

import argparse
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config, get_config
from video_tokenizer import get_dataloader, split_dataset, DummyVideoDataset
from inverse_dynamics import LatentActionModel


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    epoch: int,
    global_step: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    device = config.device

    total_loss = 0.0
    num_batches = 0

    last_log_time = time.time()
    last_log_step = global_step
    steps_per_epoch = len(dataloader)
    total_steps = config.training.num_epochs * steps_per_epoch

    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = config.use_amp and device_type == "cuda"

    for batch_idx, batch in enumerate(dataloader):
        x = batch.to(device)

        with autocast(device_type=device_type, enabled=use_amp):
            loss, pred_frames = model(x)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.max_grad_norm,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Learning rate schedule
        if global_step < config.training.warmup_steps:
            lr = config.training.learning_rate * global_step / config.training.warmup_steps
        else:
            progress = (global_step - config.training.warmup_steps) / max(1, total_steps - config.training.warmup_steps)
            progress = min(1.0, progress)
            lr = config.training.min_lr + (config.training.learning_rate - config.training.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Logging
        if global_step % config.training.log_interval == 0:
            avg_loss = total_loss / num_batches
            now = time.time()
            steps_since_log = global_step - last_log_step
            time_per_interval = now - last_log_time
            steps_per_sec = steps_since_log / time_per_interval if time_per_interval > 0 else 0

            remaining_steps = total_steps - global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds >= 3600 else f"{eta_seconds/60:.1f}m"

            print(
                f"  Step {global_step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg: {avg_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"ETA: {eta_str}"
            )

            last_log_time = now
            last_log_step = global_step

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def validate(model: nn.Module, dataloader: DataLoader, config: Config) -> float:
    """Validate the model."""
    model.eval()
    device = config.device

    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = config.use_amp and device_type == "cuda"

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            with autocast(device_type=device_type, enabled=use_amp):
                loss, _ = model(x)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, global_step, loss, config, path):
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


def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint.get("loss", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train Latent Action Model")

    # Data arguments
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--data-type", type=str, default="folder", choices=["folder", "images"])

    # Model arguments
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--n-actions", type=int, default=8)
    parser.add_argument("--no-adaptive-conditioning", action="store_true",
                       help="Disable adaptive layer norm conditioning")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)

    # Other
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config(
        model_frame_size=args.frame_size,
        model_num_frames=args.num_frames,
        model_patch_size=args.patch_size,
        model_embed_dim=args.embed_dim,
        model_num_heads=args.num_heads,
        model_num_blocks=args.num_blocks,
        model_n_actions=args.n_actions,
        training_batch_size=args.batch_size,
        training_learning_rate=args.learning_rate,
        training_num_epochs=args.num_epochs,
    )

    # Setup data
    if args.use_dummy_data:
        print("Using dummy data for training")
        train_dataset = DummyVideoDataset(
            num_samples=800, num_frames=config.model.num_frames,
            frame_size=config.model.frame_size, seed=args.seed,
        )
        val_dataset = DummyVideoDataset(
            num_samples=200, num_frames=config.model.num_frames,
            frame_size=config.model.frame_size, seed=args.seed + 1,
        )
    else:
        if args.data_path is None:
            print("Error: --data-path required when not using --use-dummy-data")
            sys.exit(1)
        print(f"Using {args.data_type} data from {args.data_path}")
        full_loader = get_dataloader(
            data_type=args.data_type, data_path=args.data_path,
            num_frames=config.model.num_frames, frame_size=config.model.frame_size,
            batch_size=1,
        )
        train_dataset, val_dataset = split_dataset(full_loader.dataset, train_ratio=0.9, seed=args.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = LatentActionModel(
        frame_size=config.model.frame_size,
        n_actions=config.model.n_actions,
        patch_size=config.model.patch_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_blocks=config.model.num_blocks,
        use_adaptive_conditioning=not args.no_adaptive_conditioning,
    )
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Action vocabulary: {config.model.n_actions}")
    print(f"Action dimension: {config.model.action_dim}")
    print(f"Adaptive conditioning: {not args.no_adaptive_conditioning}")
    print(f"Device: {config.device}")

    # Optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or "scale" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.training.learning_rate, betas=config.training.betas)

    use_amp = config.use_amp and config.device == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0

    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, global_step, _ = load_checkpoint(Path(args.resume), model, optimizer)
        start_epoch += 1

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print("-" * 40)

        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, config, epoch, global_step
        )

        val_loss = validate(model, val_loader, config)

        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")

        if (epoch + 1) % 10 == 0 or val_loss < best_val_loss:
            save_checkpoint(model, optimizer, epoch, global_step, val_loss, config,
                          checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, global_step, val_loss, config,
                          checkpoint_dir / "best_model.pt")
            print("  New best model!")

    save_checkpoint(model, optimizer, config.training.num_epochs - 1, global_step, val_loss, config,
                  checkpoint_dir / "final_model.pt")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
