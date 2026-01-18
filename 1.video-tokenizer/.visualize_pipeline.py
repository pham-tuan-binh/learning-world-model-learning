"""
Visualization script for video tokenizer pipeline components.

This script generates educational visualizations for:
1. Patch Embedding: How Conv2D extracts and projects patches
2. Positional Encoding: Sinusoidal patterns for x, y, and time
3. FSQ Quantization: How continuous latents become discrete tokens

Usage:
    uv run python 1.video-tokenizer/visualize_pipeline.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.patch_embed import PatchEmbedding
from models.positional_encoding import SpatioTemporalPositionalEncoding
from models.fsq import FiniteScalarQuantizer


def create_sample_image(size: int = 128) -> torch.Tensor:
    """
    Create a sample image with distinct visual features for demonstration.

    Creates a gradient background with colored rectangles to show
    how patches capture different regions.

    Returns:
        image: Tensor of shape (3, H, W) with values in [0, 1]
    """
    # Create gradient background
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    # RGB channels with different gradients
    r = xx * 0.3 + 0.2  # Horizontal gradient (red)
    g = yy * 0.3 + 0.3  # Vertical gradient (green)
    b = (xx + yy) / 2 * 0.3 + 0.2  # Diagonal gradient (blue)

    image = torch.stack([r, g, b], dim=0)

    # Add colored rectangles in different regions
    # Top-left: red rectangle
    image[0, 10:40, 10:40] = 0.9
    image[1, 10:40, 10:40] = 0.2
    image[2, 10:40, 10:40] = 0.2

    # Top-right: green rectangle
    image[0, 10:40, 88:118] = 0.2
    image[1, 10:40, 88:118] = 0.8
    image[2, 10:40, 88:118] = 0.2

    # Bottom-left: blue rectangle
    image[0, 88:118, 10:40] = 0.2
    image[1, 88:118, 10:40] = 0.2
    image[2, 88:118, 10:40] = 0.9

    # Bottom-right: yellow rectangle
    image[0, 88:118, 88:118] = 0.9
    image[1, 88:118, 88:118] = 0.9
    image[2, 88:118, 88:118] = 0.2

    # Center: white circle
    center = size // 2
    radius = 15
    for i in range(size):
        for j in range(size):
            if (i - center) ** 2 + (j - center) ** 2 < radius ** 2:
                image[:, i, j] = 0.95

    return image.clamp(0, 1)


def visualize_patch_embedding(save_path: str):
    """
    Visualize how patch embedding works.

    Shows:
    - Left: Original input image (128x128 RGB)
    - Middle: Grid overlay showing 8x8 patch divisions
    - Right: Several channels of patch embedding output
    """
    print("Generating patch embedding visualization...")

    # Create sample image and model
    image = create_sample_image(128)
    patch_embed = PatchEmbedding(
        in_channels=3,
        embed_dim=128,
        patch_size=8,
        frame_size=128,
    )

    # Process image through patch embedding
    # Add batch and time dimensions: (1, 1, 3, 128, 128)
    x = image.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        patches = patch_embed(x)  # (1, 1, 256, 128)

    # Reshape patches to grid for visualization
    # patches: (1, 1, 256, 128) -> (16, 16, 128)
    patch_grid = patches[0, 0].view(16, 16, 128)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Left: Original image
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title('Input Image\n(128×128×3)', fontsize=12)
    axes[0].axis('off')

    # Middle: Image with grid overlay
    axes[1].imshow(image.permute(1, 2, 0).numpy())
    axes[1].set_title('Patch Division\n(16×16 patches, each 8×8 pixels)', fontsize=12)
    # Draw grid lines
    for i in range(17):
        axes[1].axhline(y=i * 8 - 0.5, color='white', linewidth=0.5, alpha=0.8)
        axes[1].axvline(x=i * 8 - 0.5, color='white', linewidth=0.5, alpha=0.8)
    # Highlight one patch
    rect = mpatches.Rectangle((24 - 0.5, 24 - 0.5), 8, 8,
                                linewidth=2, edgecolor='yellow', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].axis('off')

    # Right: Embedding channels visualization
    # Show 4 channels of the patch embedding as a 2x2 grid
    channels_to_show = [0, 32, 64, 96]  # Sample different channels
    embedding_vis = np.zeros((32, 32))

    # Create a 2x2 grid of embedding channels
    for idx, ch in enumerate(channels_to_show):
        row, col = idx // 2, idx % 2
        channel_data = patch_grid[:, :, ch].numpy()
        # Normalize for visualization
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
        embedding_vis[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = channel_data

    im = axes[2].imshow(embedding_vis, cmap='viridis')
    axes[2].set_title('Patch Embeddings\n(4 of 128 channels, 16×16 grid)', fontsize=12)
    # Add channel labels
    for idx, ch in enumerate(channels_to_show):
        row, col = idx // 2, idx % 2
        axes[2].text(col * 16 + 8, row * 16 + 8, f'Ch {ch}',
                     ha='center', va='center', color='white', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_positional_encoding(save_path: str):
    """
    Visualize sinusoidal positional encoding patterns.

    Shows heatmaps for:
    - X position encoding (horizontal patterns)
    - Y position encoding (vertical patterns)
    - Temporal encoding (frame-based patterns)
    """
    print("Generating positional encoding visualization...")

    # Create positional encoding module
    pos_enc = SpatioTemporalPositionalEncoding(
        embed_dim=128,
        grid_size=16,
        max_frames=8,
    )

    # Get the encoding patterns
    # X encoding: (16, x_dim)
    x_positions = torch.arange(16)
    pe_x = pos_enc.pe_x(x_positions).numpy()  # (16, x_dim)

    # Y encoding: (16, y_dim)
    y_positions = torch.arange(16)
    pe_y = pos_enc.pe_y(y_positions).numpy()  # (16, y_dim)

    # Temporal encoding: (8, temporal_dim)
    t_positions = torch.arange(8)
    pe_t = pos_enc.pe_t(t_positions).numpy()  # (8, temporal_dim)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # X position encoding
    im0 = axes[0].imshow(pe_x.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title(f'X Position Encoding\n(dim={pos_enc.x_dim})', fontsize=12)
    axes[0].set_xlabel('X Position (0-15)', fontsize=10)
    axes[0].set_ylabel('Encoding Dimension', fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Y position encoding
    im1 = axes[1].imshow(pe_y.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f'Y Position Encoding\n(dim={pos_enc.y_dim})', fontsize=12)
    axes[1].set_xlabel('Y Position (0-15)', fontsize=10)
    axes[1].set_ylabel('Encoding Dimension', fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Temporal encoding
    im2 = axes[2].imshow(pe_t.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title(f'Temporal Encoding\n(dim={pos_enc.temporal_dim})', fontsize=12)
    axes[2].set_xlabel('Frame Index (0-7)', fontsize=10)
    axes[2].set_ylabel('Encoding Dimension', fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Add explanation text
    fig.text(0.5, 0.02,
             'Low frequencies (top rows) distinguish far positions; High frequencies (bottom rows) distinguish nearby positions',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_fsq(save_path: str):
    """
    Visualize FSQ (Finite Scalar Quantization) process.

    Shows:
    - Left: Distribution of continuous latent values (before quantization)
    - Middle: The quantization grid (how values map to bins)
    - Right: Token index distribution (codebook usage)
    """
    print("Generating FSQ quantization visualization...")

    # Create FSQ module
    fsq = FiniteScalarQuantizer(latent_dim=5, num_bins=4)

    # Generate random latent values (simulating encoder output)
    # Using a realistic distribution from a neural network
    torch.manual_seed(42)
    z = torch.randn(1000, 5)  # 1000 samples, 5 latent dims

    # Quantize
    with torch.no_grad():
        z_q, indices = fsq(z)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Left: Continuous latent distribution (before quantization)
    # Show distribution for all dimensions combined
    z_bounded = torch.tanh(z).numpy().flatten()
    axes[0].hist(z_bounded, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].set_title('Continuous Latent Values\n(after tanh, before quantization)', fontsize=12)
    axes[0].set_xlabel('Value', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].axvline(x=-1, color='red', linestyle='--', alpha=0.5, label='Bounds')
    axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.5)
    # Add quantization levels
    quant_levels = np.linspace(-1, 1, 4)
    for ql in quant_levels:
        axes[0].axvline(x=ql, color='green', linestyle=':', alpha=0.7)
    axes[0].set_xlim(-1.2, 1.2)

    # Middle: Quantization mapping visualization
    # Show how continuous values map to discrete bins
    x_cont = np.linspace(-2, 2, 200)
    x_bounded = np.tanh(x_cont)
    x_scaled = (x_bounded + 1) / 2 * 3  # Scale to [0, 3]
    x_rounded = np.round(x_scaled)
    x_quant = x_rounded / 3 * 2 - 1  # Back to [-1, 1]

    axes[1].plot(x_cont, x_bounded, 'b-', linewidth=2, label='tanh(z)', alpha=0.7)
    axes[1].plot(x_cont, x_quant, 'r-', linewidth=2, label='Quantized', alpha=0.7)
    axes[1].set_title('Quantization Function\n(num_bins=4)', fontsize=12)
    axes[1].set_xlabel('Input z', fontsize=10)
    axes[1].set_ylabel('Output', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    # Add horizontal lines for quantization levels
    for ql in quant_levels:
        axes[1].axhline(y=ql, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylim(-1.3, 1.3)

    # Right: Token distribution (codebook usage)
    indices_np = indices.numpy()
    unique, counts = np.unique(indices_np, return_counts=True)

    # Create histogram of token usage
    axes[2].bar(range(len(counts)), counts, color='purple', alpha=0.7, edgecolor='white')
    axes[2].set_title(f'Token Distribution\n(codebook size = {fsq.codebook_size})', fontsize=12)
    axes[2].set_xlabel('Token Index (sorted by frequency)', fontsize=10)
    axes[2].set_ylabel('Count', fontsize=10)

    # Add statistics
    usage_pct = len(unique) / fsq.codebook_size * 100
    axes[2].text(0.95, 0.95, f'Tokens used: {len(unique)}/{fsq.codebook_size}\n({usage_pct:.1f}%)',
                 transform=axes[2].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def main():
    """Generate all visualizations."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "assets" / "1.video-tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating video tokenizer pipeline visualizations...\n")

    # Generate each visualization
    visualize_patch_embedding(str(output_dir / "patch_embedding.png"))
    visualize_positional_encoding(str(output_dir / "positional_encoding.png"))
    visualize_fsq(str(output_dir / "fsq_quantization.png"))

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
