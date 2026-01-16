"""
Validation script for Video Tokenizer.

This script evaluates a trained video tokenizer by:
1. Computing reconstruction metrics (MSE, PSNR)
2. Visualizing original vs reconstructed frames
3. Analyzing codebook usage

Usage:
    # Validate with dummy data
    uv run python validate.py --checkpoint checkpoints/best_model.pt --use-dummy-data

    # Validate with real data
    uv run python validate.py --checkpoint checkpoints/best_model.pt --data-path /path/to/videos

    # Save visualizations
    uv run python validate.py --checkpoint checkpoints/best_model.pt --save-images
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from config import Config
from data_utils import DummyVideoDataset, get_dataloader
from models import VideoTokenizer


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio from MSE.

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        mse: Mean squared error
        max_val: Maximum pixel value (1.0 for normalized images)

    Returns:
        PSNR in decibels (dB)
        Higher is better, >30 dB is generally good quality
    """
    if mse < 1e-10:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def load_model(checkpoint_path: Path, device: str = "cpu") -> tuple:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config")
    if config is None:
        # Use default config if not saved in checkpoint
        from config import get_config
        config = get_config()

    # Create model
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
        dropout=0.0,  # No dropout during evaluation
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate(
    model: VideoTokenizer,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_batches: Optional[int] = None,
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: Trained VideoTokenizer
        dataloader: Data loader
        device: Device
        num_batches: Maximum batches to evaluate (None = all)

    Returns:
        Dictionary with metrics
    """
    model.eval()

    total_mse = 0.0
    total_samples = 0
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            x = batch.to(device)
            B = x.shape[0]

            # Forward pass
            loss, x_hat, indices = model(x)

            # Accumulate metrics
            total_mse += loss.item() * B
            total_samples += B
            all_indices.append(indices.cpu())

    # Compute final metrics
    avg_mse = total_mse / total_samples
    avg_psnr = compute_psnr(avg_mse)

    # Analyze codebook usage
    all_indices = torch.cat(all_indices, dim=0).flatten()
    usage = torch.bincount(all_indices, minlength=model.codebook_size)
    num_used = (usage > 0).sum().item()
    usage_ratio = num_used / model.codebook_size

    return {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "codebook_usage": num_used,
        "codebook_size": model.codebook_size,
        "usage_ratio": usage_ratio,
        "usage_histogram": usage,
        "num_samples": total_samples,
    }


def save_comparison_images(
    model: VideoTokenizer,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    output_dir: Path,
    num_samples: int = 5,
):
    """
    Save side-by-side comparison images (original vs reconstructed).

    Args:
        model: Trained VideoTokenizer
        dataloader: Data loader
        device: Device
        output_dir: Directory to save images
        num_samples: Number of samples to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping image saving.")
        print("Install with: uv add matplotlib")
        return

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            x = batch.to(device)

            # Get reconstruction
            _, x_hat, indices = model(x)

            # Move to CPU and convert to numpy
            x = x[0].cpu()  # Take first sample: (T, C, H, W)
            x_hat = x_hat[0].cpu()
            indices = indices[0].cpu()  # (T, N)

            T = x.shape[0]

            # Create figure with 2 rows (original, reconstructed) x T columns
            fig, axes = plt.subplots(2, T, figsize=(4 * T, 8))

            for t in range(T):
                # Original frame
                orig = x[t].permute(1, 2, 0).numpy()  # (H, W, C)
                orig = orig.clip(0, 1)

                # Reconstructed frame
                recon = x_hat[t].permute(1, 2, 0).numpy()
                recon = recon.clip(0, 1)

                # Plot
                if T == 1:
                    axes[0].imshow(orig)
                    axes[0].set_title(f"Original")
                    axes[0].axis("off")

                    axes[1].imshow(recon)
                    axes[1].set_title(f"Reconstructed")
                    axes[1].axis("off")
                else:
                    axes[0, t].imshow(orig)
                    axes[0, t].set_title(f"Original t={t}")
                    axes[0, t].axis("off")

                    axes[1, t].imshow(recon)
                    axes[1, t].set_title(f"Recon t={t}")
                    axes[1, t].axis("off")

            # Compute per-sample metrics
            sample_mse = F.mse_loss(x_hat, x).item()
            sample_psnr = compute_psnr(sample_mse)

            plt.suptitle(
                f"Sample {batch_idx + 1} | MSE: {sample_mse:.4f} | PSNR: {sample_psnr:.2f} dB"
            )
            plt.tight_layout()

            # Save
            save_path = output_dir / f"sample_{batch_idx + 1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Saved {save_path}")


def print_codebook_analysis(usage: torch.Tensor, top_k: int = 20):
    """
    Print analysis of codebook usage.

    Args:
        usage: Token usage histogram, shape (codebook_size,)
        top_k: Number of top/bottom tokens to show
    """
    total_tokens = usage.sum().item()
    num_used = (usage > 0).sum().item()
    codebook_size = len(usage)

    print("\n=== Codebook Analysis ===")
    print(f"Total tokens generated: {total_tokens:,}")
    print(f"Unique tokens used: {num_used}/{codebook_size} ({100*num_used/codebook_size:.1f}%)")

    # Usage statistics
    used_tokens = usage[usage > 0]
    if len(used_tokens) > 0:
        print(f"\nUsage statistics (among used tokens):")
        print(f"  Min: {used_tokens.min().item():.0f}")
        print(f"  Max: {used_tokens.max().item():.0f}")
        print(f"  Mean: {used_tokens.float().mean().item():.1f}")
        print(f"  Std: {used_tokens.float().std().item():.1f}")

    # Top tokens
    top_indices = usage.argsort(descending=True)[:top_k]
    print(f"\nTop {top_k} most used tokens:")
    for i, idx in enumerate(top_indices):
        count = usage[idx].item()
        pct = 100 * count / total_tokens if total_tokens > 0 else 0
        print(f"  {i+1}. Token {idx}: {count:,} ({pct:.2f}%)")

    # Entropy
    probs = usage.float() / total_tokens
    probs = probs[probs > 0]  # Remove zeros
    entropy = -(probs * probs.log()).sum().item()
    max_entropy = math.log(codebook_size)
    print(f"\nCodebook entropy: {entropy:.2f} / {max_entropy:.2f} ({100*entropy/max_entropy:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Validate Video Tokenizer")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data for validation",
    )
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument(
        "--data-type",
        type=str,
        default="folder",
        choices=["folder", "images"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, help="Max batches to evaluate")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save comparison images",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to visualize")

    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(Path(args.checkpoint), device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Codebook size: {model.codebook_size}")

    # Setup data
    if args.use_dummy_data:
        print("\nUsing dummy data for validation")
        dataset = DummyVideoDataset(
            num_samples=100,
            num_frames=config.model.num_frames,
            frame_size=config.model.frame_size,
            seed=123,  # Different from training
        )
    else:
        if args.data_path is None:
            print("Error: --data-path required when not using --use-dummy-data")
            return

        print(f"\nLoading data from {args.data_path}")
        dataloader = get_dataloader(
            data_type=args.data_type,
            data_path=args.data_path,
            num_frames=config.model.num_frames,
            frame_size=config.model.frame_size,
            batch_size=args.batch_size,
            shuffle=False,
        )
        dataset = dataloader.dataset

    # Create dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(model, dataloader, device, args.num_batches)

    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"MSE Loss: {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(
        f"Codebook usage: {metrics['codebook_usage']}/{metrics['codebook_size']} "
        f"({100*metrics['usage_ratio']:.1f}%)"
    )

    # Detailed codebook analysis
    print_codebook_analysis(metrics["usage_histogram"])

    # Save comparison images
    if args.save_images:
        print(f"\nSaving comparison images to {args.output_dir}...")
        save_comparison_images(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples,
        )

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
