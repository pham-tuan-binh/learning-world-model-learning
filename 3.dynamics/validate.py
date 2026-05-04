#!/usr/bin/env python3
"""
Validation script for the Token Dynamics Model.

The dataset loader in ``dynamics/data_utils.py`` caches token/action pairs
derived from raw videos before validation starts.
"""

import argparse
from pathlib import Path

import torch

from config import get_config
from dynamics import (
    DEFAULT_DATA_PATH,
    DEFAULT_INVERSE_DYNAMICS_CHECKPOINT,
    DEFAULT_TOKENIZER_CHECKPOINT,
    TokenDynamicsModel,
    get_dataloader,
)


def build_model_from_config(config) -> TokenDynamicsModel:
    """Create a dynamics model from a stored config."""
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


def validate(model, dataloader, device) -> float:
    """Compute average token cross-entropy loss."""
    model.eval()
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


def main():
    parser = argparse.ArgumentParser(description="Validate Token Dynamics Model")
    parser.add_argument("--checkpoint", type=str, required=True)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", get_config())
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.training.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.data_type = "dummy" if args.use_dummy_data else args.data_type
    config.data.data_path = args.data_path
    config.data.tokenizer_checkpoint = args.tokenizer_checkpoint
    config.data.inverse_dynamics_checkpoint = args.inverse_dynamics_checkpoint
    config.data.frame_skip = args.frame_skip
    config.data.refresh_cache = args.refresh_cache
    config.data.precompute_batch_size = args.precompute_batch_size
    config.data.precompute_device = args.precompute_device

    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)

    dataloader = get_dataloader(
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

    loss = validate(model, dataloader, config.device)
    print(f"Validation loss: {loss:.4f}")

    output_dir = Path("3.dynamics/checkpoints")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint directory: {output_dir}")


if __name__ == "__main__":
    main()
