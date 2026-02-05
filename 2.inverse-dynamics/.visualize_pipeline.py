"""
Visualization script for inverse dynamics pipeline components.

This script generates educational visualizations for:
1. Binary Action Quantization: How continuous action vectors become discrete tokens
2. Frame Pair to Action: How adjacent frames combine to infer actions
3. Token Masking: How aggressive masking forces reliance on actions
4. Adaptive Conditioning: How actions modulate transformer layers (FiLM)

Usage:
    uv run python 2.inverse-dynamics/.visualize_pipeline.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Add parent directory and video tokenizer to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "1.video-tokenizer"))

from video_tokenizer import FiniteScalarQuantizer


def create_sample_frame_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create two sample frames showing a simple motion (object moving right).

    Returns:
        frame1, frame2: Tensors of shape (3, H, W) with values in [0, 1]
    """
    size = 128

    # Create gradient background
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    # Background for both frames
    def create_background():
        r = xx * 0.2 + 0.3
        g = yy * 0.2 + 0.4
        b = torch.ones_like(xx) * 0.5
        return torch.stack([r, g, b], dim=0)

    frame1 = create_background()
    frame2 = create_background()

    # Add a moving square (red object)
    # Frame 1: square at x=30
    frame1[0, 50:80, 25:55] = 0.9
    frame1[1, 50:80, 25:55] = 0.2
    frame1[2, 50:80, 25:55] = 0.2

    # Frame 2: square at x=50 (moved right by 20 pixels)
    frame2[0, 50:80, 45:75] = 0.9
    frame2[1, 50:80, 45:75] = 0.2
    frame2[2, 50:80, 45:75] = 0.2

    return frame1.clamp(0, 1), frame2.clamp(0, 1)


def visualize_action_quantization(save_path: str):
    """
    Visualize binary action quantization (FSQ with 2 bins).

    Shows:
    - Left: 3D scatter of continuous action vectors colored by discrete action
    - Middle: How tanh + rounding maps continuous to discrete
    - Right: The 8 discrete action codes in binary form
    """
    print("Generating action quantization visualization...")

    # Create binary FSQ quantizer (same as in LatentActionModel)
    # action_dim=3, num_bins=2 -> 2^3 = 8 discrete actions
    fsq = FiniteScalarQuantizer(latent_dim=3, num_bins=2)

    # Generate random continuous action vectors
    torch.manual_seed(42)
    z = torch.randn(500, 3) * 1.5  # 500 samples, 3 dimensions

    # Quantize
    with torch.no_grad():
        z_q, indices = fsq(z)

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # Left: 3D scatter plot of continuous actions colored by discrete action
    ax1 = fig.add_subplot(131, projection='3d')

    # Apply tanh to bound values for visualization
    z_bounded = torch.tanh(z).numpy()
    indices_np = indices.numpy()

    # Color map for 8 actions
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for action_idx in range(8):
        mask = indices_np == action_idx
        if mask.any():
            ax1.scatter(
                z_bounded[mask, 0],
                z_bounded[mask, 1],
                z_bounded[mask, 2],
                c=[colors[action_idx]],
                label=f'Action {action_idx}',
                alpha=0.6,
                s=20,
            )

    ax1.set_xlabel('Dim 0', fontsize=10)
    ax1.set_ylabel('Dim 1', fontsize=10)
    ax1.set_zlabel('Dim 2', fontsize=10)
    ax1.set_title('Continuous Actions → Discrete\n(colored by action index)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=7, ncol=2)

    # Middle: Quantization function for 1D
    ax2 = fig.add_subplot(132)

    x_cont = np.linspace(-3, 3, 200)
    x_bounded = np.tanh(x_cont)
    # Binary quantization: [-1, 1] -> {0, 1} -> {-1, 1}
    x_scaled = (x_bounded + 1) / 2  # Scale to [0, 1]
    x_rounded = np.round(x_scaled)  # Round to {0, 1}
    x_quant = x_rounded * 2 - 1  # Back to {-1, 1}

    ax2.plot(x_cont, x_bounded, 'b-', linewidth=2, label='tanh(z)', alpha=0.7)
    ax2.plot(x_cont, x_quant, 'r-', linewidth=2, label='Quantized', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(x_cont, -1.5, 0, where=(x_quant < 0), alpha=0.1, color='blue', label='Bin 0')
    ax2.fill_between(x_cont, 0, 1.5, where=(x_quant > 0), alpha=0.1, color='red', label='Bin 1')
    ax2.set_xlabel('Input z', fontsize=10)
    ax2.set_ylabel('Output', fontsize=10)
    ax2.set_title('Binary Quantization (per dimension)\n2 bins: {-1, +1}', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # Right: Action codebook visualization
    ax3 = fig.add_subplot(133)

    # Create 8 action codes as binary representations
    action_codes = np.array([
        [0, 0, 0],  # Action 0
        [0, 0, 1],  # Action 1
        [0, 1, 0],  # Action 2
        [0, 1, 1],  # Action 3
        [1, 0, 0],  # Action 4
        [1, 0, 1],  # Action 5
        [1, 1, 0],  # Action 6
        [1, 1, 1],  # Action 7
    ])

    # Plot as heatmap
    im = ax3.imshow(action_codes.T, cmap='RdYlBu_r', aspect='auto', vmin=-0.5, vmax=1.5)
    ax3.set_xticks(range(8))
    ax3.set_xticklabels([f'{i}' for i in range(8)])
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(['Dim 0', 'Dim 1', 'Dim 2'])
    ax3.set_xlabel('Action Index', fontsize=10)
    ax3.set_title('Action Codebook\n(8 discrete actions = 2³)', fontsize=11)

    # Add text annotations
    for i in range(8):
        for j in range(3):
            ax3.text(i, j, str(action_codes[i, j]),
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     color='white' if action_codes[i, j] == 1 else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_frame_pair_to_action(save_path: str):
    """
    Visualize how two frames combine to produce an action.

    Shows:
    - Frame t and Frame t+1 side by side
    - The difference highlighting motion
    - How pooled features combine (conceptual diagram)
    """
    print("Generating frame pair to action visualization...")

    frame1, frame2 = create_sample_frame_pair()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Frame t
    axes[0].imshow(frame1.permute(1, 2, 0).numpy())
    axes[0].set_title('Frame t\n(current state)', fontsize=11)
    axes[0].axis('off')

    # Frame t+1
    axes[1].imshow(frame2.permute(1, 2, 0).numpy())
    axes[1].set_title('Frame t+1\n(next state)', fontsize=11)
    axes[1].axis('off')

    # Difference map (highlights motion)
    diff = (frame2 - frame1).abs().mean(dim=0)
    im = axes[2].imshow(diff.numpy(), cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Motion Detection\n|frame t+1 - frame t|', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Conceptual diagram of action inference
    axes[3].set_xlim(0, 10)
    axes[3].set_ylim(0, 10)
    axes[3].axis('off')
    axes[3].set_title('Action Inference\n(encoder architecture)', fontsize=11)

    # Draw diagram
    # Frame t box
    rect1 = mpatches.FancyBboxPatch((0.5, 6), 2, 1.5, boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black', linewidth=1.5)
    axes[3].add_patch(rect1)
    axes[3].text(1.5, 6.75, 'Pool(t)\n(B, E)', ha='center', va='center', fontsize=9)

    # Frame t+1 box
    rect2 = mpatches.FancyBboxPatch((0.5, 4), 2, 1.5, boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black', linewidth=1.5)
    axes[3].add_patch(rect2)
    axes[3].text(1.5, 4.75, 'Pool(t+1)\n(B, E)', ha='center', va='center', fontsize=9)

    # Concat operation
    axes[3].annotate('', xy=(3.5, 5.5), xytext=(2.5, 6.5),
                     arrowprops=dict(arrowstyle='->', color='black'))
    axes[3].annotate('', xy=(3.5, 5.5), xytext=(2.5, 4.75),
                     arrowprops=dict(arrowstyle='->', color='black'))

    rect3 = mpatches.FancyBboxPatch((3.5, 5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    axes[3].add_patch(rect3)
    axes[3].text(4.75, 5.6, 'Concat\n(B, E×2)', ha='center', va='center', fontsize=9)

    # Action head
    axes[3].annotate('', xy=(7, 5.6), xytext=(6, 5.6),
                     arrowprops=dict(arrowstyle='->', color='black'))

    rect4 = mpatches.FancyBboxPatch((7, 5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='salmon', edgecolor='black', linewidth=1.5)
    axes[3].add_patch(rect4)
    axes[3].text(8.25, 5.6, 'Action Head\nMLP', ha='center', va='center', fontsize=9)

    # Output action
    axes[3].annotate('', xy=(8.25, 4.5), xytext=(8.25, 5),
                     arrowprops=dict(arrowstyle='->', color='black'))

    rect5 = mpatches.FancyBboxPatch((7, 3.2), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='plum', edgecolor='black', linewidth=1.5)
    axes[3].add_patch(rect5)
    axes[3].text(8.25, 3.8, 'Action\n(B, A)', ha='center', va='center', fontsize=9)

    # Add description
    axes[3].text(5, 1.5, 'concat([frame_t, frame_t+1]) → action',
                 ha='center', va='center', fontsize=10, style='italic',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_token_masking(save_path: str):
    """
    Visualize how aggressive token masking forces reliance on actions.

    Shows:
    - Left: Original patch grid (what encoder sees)
    - Middle: Masked tokens during training (decoder input)
    - Right: The role of action in reconstruction
    """
    print("Generating token masking visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create a simple 4x4 grid representing patches
    grid_size = 4
    num_patches = grid_size * grid_size

    # Left: Original tokens (frame 1)
    original = np.random.rand(grid_size, grid_size)
    im1 = axes[0].imshow(original, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Frame t Patches\n(anchor, never masked)', fontsize=11)
    axes[0].set_xticks(range(grid_size))
    axes[0].set_yticks(range(grid_size))
    axes[0].grid(True, color='white', linewidth=2)
    for i in range(grid_size):
        for j in range(grid_size):
            axes[0].text(j, i, f'{i*grid_size+j}', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Middle: Masked tokens (frames 2, 3, ... during training)
    masked = np.ones((grid_size, grid_size)) * 0.5  # All masked (gray)
    im2 = axes[1].imshow(masked, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Frames t+1, t+2, ... (training)\n100% masked → [MASK] token', fontsize=11)
    axes[1].set_xticks(range(grid_size))
    axes[1].set_yticks(range(grid_size))
    axes[1].grid(True, color='white', linewidth=2)
    for i in range(grid_size):
        for j in range(grid_size):
            axes[1].text(j, i, 'M', ha='center', va='center',
                        fontsize=12, color='red', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Right: Conceptual diagram showing action's role
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 10)
    axes[2].axis('off')
    axes[2].set_title('Why Masking Forces Action Reliance', fontsize=11)

    # Draw explanation
    # Without action
    axes[2].text(5, 9, 'Without masking:', fontsize=10, ha='center', fontweight='bold')
    axes[2].text(5, 8, 'Decoder can copy input → ignores action', fontsize=9, ha='center', style='italic')

    # With action
    axes[2].text(5, 6.5, 'With 100% masking:', fontsize=10, ha='center', fontweight='bold')
    axes[2].text(5, 5.5, 'Decoder sees: Frame 1 + [MASK] + [MASK] + ...', fontsize=9, ha='center')
    axes[2].text(5, 4.5, 'Must use ACTION to reconstruct frames 2, 3, ...', fontsize=9, ha='center', style='italic')

    # Visual diagram
    boxes_y = 2
    # Frame 1 (visible)
    rect1 = mpatches.FancyBboxPatch((1, boxes_y), 1.5, 1, boxstyle="round,pad=0.05",
                                     facecolor='lightgreen', edgecolor='black')
    axes[2].add_patch(rect1)
    axes[2].text(1.75, boxes_y + 0.5, 'F1\n✓', ha='center', va='center', fontsize=9)

    # Frame 2 (masked)
    rect2 = mpatches.FancyBboxPatch((3, boxes_y), 1.5, 1, boxstyle="round,pad=0.05",
                                     facecolor='lightgray', edgecolor='black')
    axes[2].add_patch(rect2)
    axes[2].text(3.75, boxes_y + 0.5, '[M]\n?', ha='center', va='center', fontsize=9, color='red')

    # Frame 3 (masked)
    rect3 = mpatches.FancyBboxPatch((5, boxes_y), 1.5, 1, boxstyle="round,pad=0.05",
                                     facecolor='lightgray', edgecolor='black')
    axes[2].add_patch(rect3)
    axes[2].text(5.75, boxes_y + 0.5, '[M]\n?', ha='center', va='center', fontsize=9, color='red')

    # Action (key)
    rect4 = mpatches.FancyBboxPatch((7.5, boxes_y), 1.5, 1, boxstyle="round,pad=0.05",
                                     facecolor='salmon', edgecolor='black', linewidth=2)
    axes[2].add_patch(rect4)
    axes[2].text(8.25, boxes_y + 0.5, 'Action\n(KEY)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow from action
    axes[2].annotate('', xy=(5.75, boxes_y + 1.2), xytext=(8, boxes_y + 1.1),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    axes[2].text(5, 0.5, 'Action carries ALL information about motion!',
                 ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_adaptive_conditioning(save_path: str):
    """
    Visualize FiLM-style adaptive layer normalization conditioning.

    Shows:
    - Left: Standard layer normalization
    - Middle: How action generates scale/shift via MLP
    - Right: Effect of different actions on features
    """
    print("Generating adaptive conditioning visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Standard LayerNorm
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Standard Layer Normalization', fontsize=11)

    # Input features
    rect1 = mpatches.FancyBboxPatch((1, 7), 3, 1.5, boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black')
    ax1.add_patch(rect1)
    ax1.text(2.5, 7.75, 'Input x\n(B, N, E)', ha='center', va='center', fontsize=9)

    # LayerNorm
    ax1.annotate('', xy=(5, 7.75), xytext=(4, 7.75),
                 arrowprops=dict(arrowstyle='->', color='black'))
    rect2 = mpatches.FancyBboxPatch((5, 7), 3, 1.5, boxstyle="round,pad=0.1",
                                     facecolor='lightyellow', edgecolor='black')
    ax1.add_patch(rect2)
    ax1.text(6.5, 7.75, 'LayerNorm\nx̂ = (x-μ)/σ', ha='center', va='center', fontsize=9)

    # Output
    ax1.annotate('', xy=(6.5, 6.5), xytext=(6.5, 7),
                 arrowprops=dict(arrowstyle='->', color='black'))
    rect3 = mpatches.FancyBboxPatch((5, 4.5), 3, 1.5, boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black')
    ax1.add_patch(rect3)
    ax1.text(6.5, 5.25, 'Output\nγ·x̂ + β', ha='center', va='center', fontsize=9)

    # Note
    ax1.text(5, 2.5, 'γ, β are learnable\nbut FIXED after training',
             ha='center', va='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Middle: Adaptive LayerNorm (FiLM)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Adaptive LayerNorm (FiLM)\nx · (1 + scale) + shift', fontsize=11)

    # Input features
    rect1 = mpatches.FancyBboxPatch((0.5, 7), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black')
    ax2.add_patch(rect1)
    ax2.text(1.75, 7.6, 'Input x', ha='center', va='center', fontsize=9)

    # Action
    rect2 = mpatches.FancyBboxPatch((0.5, 4.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='salmon', edgecolor='black', linewidth=2)
    ax2.add_patch(rect2)
    ax2.text(1.75, 5.1, 'Action\n(B, A)', ha='center', va='center', fontsize=9, fontweight='bold')

    # MLP
    ax2.annotate('', xy=(4, 5.1), xytext=(3, 5.1),
                 arrowprops=dict(arrowstyle='->', color='black'))
    rect3 = mpatches.FancyBboxPatch((4, 4.5), 2, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='plum', edgecolor='black')
    ax2.add_patch(rect3)
    ax2.text(5, 5.1, 'MLP\n→ 2E', ha='center', va='center', fontsize=9)

    # Split to scale and shift
    ax2.annotate('', xy=(7, 6), xytext=(6, 5.3),
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax2.annotate('', xy=(7, 4.2), xytext=(6, 4.9),
                 arrowprops=dict(arrowstyle='->', color='red'))

    rect4 = mpatches.FancyBboxPatch((7, 5.5), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor='lightskyblue', edgecolor='blue')
    ax2.add_patch(rect4)
    ax2.text(8, 6, 'scale', ha='center', va='center', fontsize=9)

    rect5 = mpatches.FancyBboxPatch((7, 3.7), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor='lightcoral', edgecolor='red')
    ax2.add_patch(rect5)
    ax2.text(8, 4.2, 'shift', ha='center', va='center', fontsize=9)

    # Modulation
    ax2.annotate('', xy=(5, 7), xytext=(3, 7.4),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax2.annotate('', xy=(5, 7), xytext=(8, 6.5),
                 arrowprops=dict(arrowstyle='->', color='blue', linestyle='dashed'))
    ax2.annotate('', xy=(5, 7), xytext=(8, 4.7),
                 arrowprops=dict(arrowstyle='->', color='red', linestyle='dashed'))

    # Output
    rect6 = mpatches.FancyBboxPatch((4, 7.5), 3, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black')
    ax2.add_patch(rect6)
    ax2.text(5.5, 8.1, 'x·(1+scale)+shift', ha='center', va='center', fontsize=9)

    # Note
    ax2.text(5, 2, 'scale, shift DEPEND on action\n→ different actions = different outputs',
             ha='center', va='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Right: Effect visualization
    ax3 = axes[2]
    ax3.set_title('Effect of Different Actions on Features', fontsize=11)

    # Generate sample data showing how different "actions" modulate features
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    base_signal = np.sin(x) + 0.5

    # Different action effects (scale, shift)
    action_effects = [
        (1.0, 0.0, 'Action 0 (identity)'),
        (1.5, 0.3, 'Action 1 (scale up, shift up)'),
        (0.5, -0.3, 'Action 2 (scale down, shift down)'),
        (1.2, -0.5, 'Action 3 (scale up, shift down)'),
    ]

    colors = ['black', 'blue', 'red', 'green']
    for (scale, shift, label), color in zip(action_effects, colors):
        modulated = base_signal * scale + shift
        linestyle = '-' if scale == 1.0 else '--'
        linewidth = 2 if scale == 1.0 else 1.5
        ax3.plot(x, modulated, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    ax3.set_xlabel('Feature dimension', fontsize=10)
    ax3.set_ylabel('Feature value', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_variance_penalty(save_path: str):
    """
    Visualize the variance penalty that prevents action collapse.

    Shows:
    - Left: Action collapse (all same action)
    - Middle: Diverse actions (healthy variance)
    - Right: Variance penalty function
    """
    print("Generating variance penalty visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Action collapse
    ax1 = axes[0]
    np.random.seed(42)

    # Collapsed actions (all similar)
    collapsed_actions = np.random.randn(100, 2) * 0.05 + np.array([0.5, 0.5])

    ax1.scatter(collapsed_actions[:, 0], collapsed_actions[:, 1], alpha=0.5, c='red', s=30)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Action Dim 0', fontsize=10)
    ax1.set_ylabel('Action Dim 1', fontsize=10)
    ax1.set_title('Action Collapse (BAD)\nVariance ≈ 0.0025', fontsize=11)
    ax1.set_aspect('equal')

    var_collapsed = collapsed_actions.var()
    ax1.text(0, -1.2, f'var = {var_collapsed:.4f} < target',
             ha='center', fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Middle: Diverse actions
    ax2 = axes[1]
    diverse_actions = np.random.randn(100, 2) * 0.5

    ax2.scatter(diverse_actions[:, 0], diverse_actions[:, 1], alpha=0.5, c='green', s=30)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Action Dim 0', fontsize=10)
    ax2.set_ylabel('Action Dim 1', fontsize=10)
    ax2.set_title('Diverse Actions (GOOD)\nVariance ≈ 0.25', fontsize=11)
    ax2.set_aspect('equal')

    var_diverse = diverse_actions.var()
    ax2.text(0, -1.2, f'var = {var_diverse:.4f} > target',
             ha='center', fontsize=10, color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right: Variance penalty function
    ax3 = axes[2]

    var_target = 0.01  # From config
    var_values = np.linspace(0, 0.05, 100)
    penalty = np.maximum(var_target - var_values, 0)

    ax3.plot(var_values, penalty, 'b-', linewidth=2)
    ax3.fill_between(var_values, 0, penalty, alpha=0.3)
    ax3.axvline(x=var_target, color='red', linestyle='--', label=f'Target = {var_target}')
    ax3.set_xlabel('Action Variance', fontsize=10)
    ax3.set_ylabel('Penalty', fontsize=10)
    ax3.set_title('Variance Penalty Function\nReLU(target - variance)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add annotation
    ax3.annotate('Penalty kicks in\nwhen variance drops', xy=(0.005, 0.005),
                 xytext=(0.025, 0.006),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {save_path}")


def main():
    """Generate all visualizations."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "assets" / "2.inverse-dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating inverse dynamics pipeline visualizations...\n")

    # Generate each visualization
    visualize_action_quantization(str(output_dir / "action_quantization.png"))
    visualize_frame_pair_to_action(str(output_dir / "frame_pair_to_action.png"))
    visualize_token_masking(str(output_dir / "token_masking.png"))
    visualize_adaptive_conditioning(str(output_dir / "adaptive_conditioning.png"))
    visualize_variance_penalty(str(output_dir / "variance_penalty.png"))

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
