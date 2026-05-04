#!/usr/bin/env python3
"""
Training script for the Token Dynamics Model.

``dynamics/data_utils.py`` turns raw video clips into cached token/action pairs
using the trained video tokenizer from folder 1 and inverse dynamics model from
folder 2.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import Config, get_config
from dynamics import (
    DEFAULT_DATA_PATH,
    DEFAULT_INVERSE_DYNAMICS_CHECKPOINT,
    DEFAULT_TOKENIZER_CHECKPOINT,
    TokenDynamicsModel,
    get_dataloader,
    split_dataset,
)


def build_model(config: Config) -> TokenDynamicsModel:
    """Create a token dynamics model from config."""
    return TokenDynamicsModel(
        vocab_size=config.model.vocab_size,
        num_patches=config.model.num_patches,
        action_dim=config.model.action_dim,
        n_actions=config.model.n_actions,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_blocks=config.model.num_blocks,
        grid_size=config.model.grid_size,
        dropout=config.model.dropout,
        use_adaptive_conditioning=config.model.use_adaptive_conditioning,
    )


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    epoch: int,
    global_step: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = config.use_amp and device_type == "cuda"

    total_loss = 0.0
    num_batches = 0
    last_log_time = time.time()
    last_log_step = global_step
    total_steps = config.training.num_epochs * len(dataloader)

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        actions = batch["actions"].to(device)

        with autocast(device_type=device_type, enabled=use_amp):
            loss, _ = model.training_step(tokens, actions)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        total_loss += loss.item()
        num_batches += 1

        if global_step < config.training.warmup_steps:
            lr = config.training.learning_rate * global_step / config.training.warmup_steps
        else:
            progress = (global_step - config.training.warmup_steps) / max(
                1,
                total_steps - config.training.warmup_steps,
            )
            progress = min(1.0, progress)
            lr = config.training.min_lr + (
                config.training.learning_rate - config.training.min_lr
            ) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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

    return total_loss / max(num_batches, 1), global_step


def validate(model: nn.Module, dataloader, config: Config) -> float:
    """Validate the dynamics model."""
    model.eval()
    device = config.device
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            actions = batch["actions"].to(device)
            loss, _ = model.training_step(tokens, actions)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, epoch, global_step, loss, config, path):
    """Save a training checkpoint."""
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
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint.get("loss", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train Token Dynamics Model")
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--data-type",
        type=str,
        default="folder",
        choices=["dummy", "folder"],
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default=DEFAULT_TOKENIZER_CHECKPOINT,
    )
    parser.add_argument(
        "--inverse-dynamics-checkpoint",
        type=str,
        default=DEFAULT_INVERSE_DYNAMICS_CHECKPOINT,
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--precompute-batch-size", type=int, default=16)
    parser.add_argument("--precompute-device", type=str)

    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="3.dynamics/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    data_type = "dummy" if args.use_dummy_data else args.data_type
    config = get_config(
        model_embed_dim=args.embed_dim,
        model_num_heads=args.num_heads,
        model_num_blocks=args.num_blocks,
        model_frame_size=args.frame_size,
        model_patch_size=args.patch_size,
        training_num_epochs=args.num_epochs,
        training_batch_size=args.batch_size,
        training_learning_rate=args.learning_rate,
        data_data_type=data_type,
        data_data_path=args.data_path,
        data_tokenizer_checkpoint=args.tokenizer_checkpoint,
        data_inverse_dynamics_checkpoint=args.inverse_dynamics_checkpoint,
        data_frame_skip=args.frame_skip,
        data_num_workers=args.num_workers,
        data_refresh_cache=args.refresh_cache,
        data_precompute_batch_size=args.precompute_batch_size,
        data_precompute_device=args.precompute_device,
        checkpoint_dir=args.checkpoint_dir,
    )

    torch.manual_seed(config.training.seed)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("Setting up data...")
    full_loader = get_dataloader(
        data_type=config.data.data_type,
        data_path=config.data.data_path,
        tokenizer_checkpoint=config.data.tokenizer_checkpoint,
        inverse_dynamics_checkpoint=config.data.inverse_dynamics_checkpoint,
        batch_size=config.training.batch_size,
        num_frames=config.model.num_frames,
        frame_size=config.model.frame_size,
        num_patches=config.model.num_patches,
        vocab_size=config.model.vocab_size,
        action_dim=config.model.action_dim,
        frame_skip=config.data.frame_skip,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        dummy_num_samples=config.data.dummy_num_samples,
        refresh_cache=config.data.refresh_cache,
        precompute_batch_size=config.data.precompute_batch_size,
        precompute_device=config.data.precompute_device,
    )

    train_dataset, val_dataset = split_dataset(full_loader.dataset, config.data.train_split)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    print("Creating model...")
    model = build_model(config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {config.device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )
    scaler = GradScaler(enabled=config.use_amp and "cuda" in config.device)

    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, global_step, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1  # start from next epoch

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    checkpoint_dir = Path(config.checkpoint_dir)
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print("-" * 40)
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            config,
            epoch,
            global_step,
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
