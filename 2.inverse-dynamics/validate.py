"""
Validation script for Latent Action Model.

Evaluates the model and optionally saves visualizations of:
- Original frames
- Predicted next frames
- Action distributions

Usage:
    uv run python validate.py --checkpoint checkpoints/best_model.pt --data-path ./data
    uv run python validate.py --checkpoint checkpoints/best_model.pt --use-dummy-data --save-images
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_config
from video_tokenizer import get_dataloader, DummyVideoDataset
from inverse_dynamics import LatentActionModel


def validate(model, dataloader, device):
    """Compute validation metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            loss, _ = model(x)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def analyze_actions(model, dataloader, device, num_batches=10):
    """Analyze the distribution of inferred actions."""
    model.eval()
    all_actions = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = batch.to(device)
            actions = model.encode(x)  # (B, T-1, A)
            all_actions.append(actions.cpu())

    all_actions = torch.cat(all_actions, dim=0)  # (N, T-1, A)

    print("\n=== Action Analysis ===")
    print(f"Total samples: {all_actions.shape[0]}")
    print(f"Actions per sample: {all_actions.shape[1]}")
    print(f"Action dimension: {all_actions.shape[2]}")

    # Flatten for statistics
    flat_actions = all_actions.view(-1, all_actions.shape[-1])

    print(f"\nAction statistics:")
    print(f"  Mean: {flat_actions.mean(dim=0).tolist()}")
    print(f"  Std:  {flat_actions.std(dim=0).tolist()}")
    print(f"  Min:  {flat_actions.min(dim=0).values.tolist()}")
    print(f"  Max:  {flat_actions.max(dim=0).values.tolist()}")

    # Count unique actions (after quantization, values are discrete)
    # Convert to integer indices
    action_indices = ((flat_actions + 1) / 2).round().long()
    basis = torch.tensor([2 ** i for i in range(action_indices.shape[-1])])
    flat_indices = (action_indices * basis).sum(dim=-1)

    unique_actions = flat_indices.unique()
    print(f"\nUnique actions used: {len(unique_actions)} / {model.n_actions}")
    print(f"Action indices: {unique_actions.tolist()}")


def save_predictions(model, dataloader, device, output_dir, num_samples=5):
    """Save prediction visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(dataloader))
        x = batch[:num_samples].to(device)
        _, pred_frames = model(x)

        # x: (B, T, C, H, W), pred: (B, T-1, C, H, W)
        x = x.cpu()
        pred_frames = pred_frames.cpu()

        for i in range(min(num_samples, x.shape[0])):
            fig, axes = plt.subplots(2, x.shape[1], figsize=(3 * x.shape[1], 6))

            # Original frames
            for t in range(x.shape[1]):
                img = x[i, t].permute(1, 2, 0).numpy()
                axes[0, t].imshow(img.clip(0, 1))
                axes[0, t].set_title(f"Frame {t+1}")
                axes[0, t].axis("off")

            # Predicted frames (offset by 1)
            axes[1, 0].axis("off")
            axes[1, 0].set_title("(input)")
            for t in range(pred_frames.shape[1]):
                img = pred_frames[i, t].permute(1, 2, 0).numpy()
                axes[1, t + 1].imshow(img.clip(0, 1))
                axes[1, t + 1].set_title(f"Pred {t+2}")
                axes[1, t + 1].axis("off")

            axes[0, 0].set_ylabel("Original", fontsize=12)
            axes[1, 0].set_ylabel("Predicted", fontsize=12)

            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{i+1}.png", dpi=150)
            plt.close()

    print(f"\nSaved {num_samples} prediction visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Validate Latent Action Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--data-type", type=str, default="folder")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Create model
    model = LatentActionModel(
        frame_size=config.model.frame_size,
        n_actions=config.model.n_actions,
        patch_size=config.model.patch_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_blocks=config.model.num_blocks,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Action vocabulary: {config.model.n_actions}")

    # Setup data
    if args.use_dummy_data:
        dataset = DummyVideoDataset(
            num_samples=200, num_frames=config.model.num_frames,
            frame_size=config.model.frame_size,
        )
    else:
        if args.data_path is None:
            print("Error: --data-path required when not using --use-dummy-data")
            return
        loader = get_dataloader(
            data_type=args.data_type, data_path=args.data_path,
            num_frames=config.model.num_frames, frame_size=config.model.frame_size,
            batch_size=1,
        )
        dataset = loader.dataset

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Validate
    val_loss = validate(model, dataloader, config.device)
    print(f"\nValidation Loss: {val_loss:.4f}")

    # Analyze actions
    analyze_actions(model, dataloader, config.device)

    # Save visualizations
    if args.save_images:
        save_predictions(model, dataloader, config.device, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
