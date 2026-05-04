"""
Visualization script for token dynamics pipeline components.

This script generates educational visualizations for:
1. Token Vocabulary: How discrete visual token IDs map into an embedding space
2. Action Conditioning: How action embeddings are broadcast and added to patch tokens
3. Causal Temporal Attention: The attention mask that ensures tokens only attend to the past
4. Teacher-Forced Training: How the model is trained to predict next-frame tokens
5. Rollout Pipeline: Autoregressive generation of future frames from a checkpoint

Usage:
    uv run python 3.dynamics/.visualize_pipeline.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "1.video-tokenizer"))
sys.path.insert(0, str(Path(__file__).parent.parent / "2.inverse-dynamics"))

from dynamics import TokenDynamicsModel


def visualize_token_vocabulary(save_path: str):
    """
    Visualize the discrete visual token vocabulary.

    Shows:
    - Left: How raw patch pixel values map to discrete token IDs via FSQ
    - Middle: 2D PCA projection of the token embedding matrix
    - Right: Token ID distribution over a dummy clip
    """
    print("Generating token vocabulary visualization...")

    vocab_size = 1024
    embed_dim = 128
    num_patches = 256
    num_frames = 4

    model = TokenDynamicsModel(
        vocab_size=vocab_size,
        num_patches=num_patches,
        embed_dim=embed_dim,
    )

    # Sample random token ids to represent a batch of clips
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (2, num_frames, num_patches))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Quantization pipeline schematic (pixels → patches → FSQ tokens)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Pixels → Discrete Token IDs\n(via Video Tokenizer + FSQ)", fontsize=11)

    stages = [
        (5, 8.5, "Raw Frames\n(B, T, C, H, W)", "lightblue"),
        (5, 6.5, "Patch Embeddings\n(B, T, N, E)", "lightyellow"),
        (5, 4.5, "FSQ Latents\n(B, T, N, 5)", "lightcoral"),
        (5, 2.5, "Token IDs\n(B, T, N)  ∈ [0, 1023]", "lightgreen"),
    ]
    for x, y, label, color in stages:
        rect = mpatches.FancyBboxPatch(
            (x - 3, y - 0.6), 6, 1.2,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="black",
        )
        ax1.add_patch(rect)
        ax1.text(x, y, label, ha="center", va="center", fontsize=9)

    for i in range(len(stages) - 1):
        ax1.annotate(
            "", xy=(stages[i + 1][0], stages[i + 1][1] + 0.65),
            xytext=(stages[i][0], stages[i][1] - 0.65),
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    ax1.text(5, 1.0, f"vocab_size = {vocab_size}  (= 4⁵ bins)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # Middle: 2D PCA of embedding matrix
    ax2 = axes[1]
    with torch.no_grad():
        W = model.token_embed.weight.detach().numpy()  # (V, E)

    # PCA projection via SVD (float64; suppress spurious BLAS overflow warnings)
    W_centered = (W - W.mean(axis=0)).astype(np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, _, Vt = np.linalg.svd(W_centered, full_matrices=False)
        proj = W_centered @ Vt[:2].T  # (V, 2)

    scatter = ax2.scatter(proj[:, 0], proj[:, 1], c=np.arange(vocab_size),
                          cmap="viridis", s=3, alpha=0.6)
    ax2.set_title(f"Token Embedding Matrix (PCA)\n(vocab_size={vocab_size}, embed_dim={embed_dim})",
                  fontsize=11)
    ax2.set_xlabel("PC 1", fontsize=10)
    ax2.set_ylabel("PC 2", fontsize=10)
    plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04, label="Token ID")

    # Right: Token ID distribution over the sampled clip
    ax3 = axes[2]
    ids = tokens.flatten().numpy()
    ax3.hist(ids, bins=64, color="steelblue", edgecolor="white", alpha=0.8)
    ax3.set_title(f"Token ID Distribution\n(B=2, T={num_frames}, N={num_patches} patches)",
                  fontsize=11)
    ax3.set_xlabel("Token ID", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.text(0.97, 0.97, f"unique: {np.unique(ids).size}/{vocab_size}",
             transform=ax3.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_action_conditioning(save_path: str):
    """
    Visualize both action conditioning paths in the dynamics model.

    Shows:
    - Left:   Path 1 — additive conditioning (always active)
              action → MLP → embed broadcast-added to token embeds before transformer
    - Middle: Path 2 — FiLM / adaptive conditioning (use_adaptive_conditioning=True)
              mean(actions) per-block: gate scalars + AdaptiveRMSNorm scale/shift
    - Right:  Before/after L2 norm heatmap for the additive path
    """
    print("Generating action conditioning visualization...")

    torch.manual_seed(7)
    num_patches = 16
    grid_size = 4
    embed_dim = 32
    action_dim = 3

    model = TokenDynamicsModel(
        vocab_size=256,
        num_patches=num_patches,
        action_dim=action_dim,
        embed_dim=embed_dim,
        grid_size=grid_size,
    )

    tokens  = torch.randint(0, 256, (1, 1, num_patches))
    actions = torch.tensor([[[1.0, -1.0, 1.0]]])

    with torch.no_grad():
        x_before = model.token_embed(tokens.long())[0, 0]  # (N, E)
        act_embed = model._embed_actions(actions)[0, 0]    # (E,)
        x_after  = x_before + act_embed.unsqueeze(0)       # (N, E)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Left: Path 1 — additive input conditioning ──────────────────────────
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Path 1: Additive Conditioning\n(always active)", fontsize=11)

    def _box(ax, x, y, label, color, w=7, h=1.2):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="black"))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5)

    def _arrow(ax, x, y_from, y_to):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="->", color="black"))

    steps1 = [
        (5, 8.6, "action vector  (B, T, A)", "salmon"),
        (5, 6.8, "MLP: Linear → GELU → Linear\n→ action embed  (B, T, E)", "plum"),
        (5, 4.9, "broadcast to (B, T, N, E)\n(same embed for every patch)", "lightyellow"),
        (5, 3.1, "token embeds  +  action embed\n→ input to transformer", "lightgreen"),
    ]
    for x, y, label, color in steps1:
        _box(ax1, x, y, label, color)
    for i in range(len(steps1) - 1):
        _arrow(ax1, steps1[i][0], steps1[i][1] - 0.65, steps1[i + 1][1] + 0.65)

    ax1.text(5, 1.5, "Discrete id path: action_id_embed(id)",
             ha="center", fontsize=8, style="italic",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    # ── Middle: Path 2 — FiLM adaptive conditioning ─────────────────────────
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Path 2: FiLM Adaptive Conditioning\n(use_adaptive_conditioning=True)", fontsize=11)

    # Top flow  (y: 9.2 → 7.2)
    _box(ax2, 5, 9.0, "mean(actions, dim=T)  →  (B, A)\n(clip-level action summary)", "salmon",
         w=8.5, h=1.1)
    _arrow(ax2, 5, 9.0 - 0.6, 7.8 + 0.6)
    _box(ax2, 5, 7.8, "passed to every transformer block\nas conditioning  (B, A)", "plum",
         w=8.5, h=1.1)

    # Arrow from second box into the dashed container
    _arrow(ax2, 5, 7.8 - 0.6, 6.55)

    # Dashed container  (y: 1.4 → 6.4  height 5.0)
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.5, 1.4), 9.0, 5.0,
        boxstyle="round,pad=0.15",
        facecolor="#f7f7f7", edgecolor="#555555", linewidth=1.5, linestyle="--"))
    # Label sits *inside* the container with clear breathing room below the top border
    ax2.text(5, 6.2, "inside each SpatioTemporalBlock", ha="center", fontsize=8,
             color="#333333", style="italic")

    # Two sub-mechanisms with generous spacing
    _box(ax2, 5, 5.2, "gate_mlp  →  3 gate scalars\n× spatial / temporal / FFN residuals",
         "lightskyblue", w=8.0, h=1.1)
    _arrow(ax2, 5, 5.2 - 0.6, 3.6 + 0.6)
    _box(ax2, 5, 3.6, "AdaptiveRMSNorm (FiLM)\nx̂ · (1 + scale) + shift\n[attn + FFN outputs]",
         "lightcoral", w=8.0, h=1.2)

    ax2.text(5, 2.15, "zero-init  →  identity at start",
             ha="center", fontsize=7.5, style="italic", color="#555555")

    # ── Right: FiLM modulation effect ────────────────────────────────────────
    ax3 = axes[2]
    x_feat = np.linspace(0, 4 * np.pi, 128)
    base = np.sin(x_feat) * 0.7 + np.sin(2 * x_feat) * 0.3   # base normalized signal

    variants = [
        (0.0,  0.0,  "no conditioning\n(scale=0, shift=0)", "black",    "-",  2.0),
        (0.6,  0.4,  "scale=+0.6, shift=+0.4",              "#2196F3",  "--", 1.5),
        (-0.4, -0.3, "scale=−0.4, shift=−0.3",              "#F44336",  "--", 1.5),
        (0.8, -0.5,  "scale=+0.8, shift=−0.5",              "#4CAF50",  "--", 1.5),
    ]
    for scale, shift, label, color, ls, lw in variants:
        ax3.plot(x_feat, base * (1 + scale) + shift,
                 color=color, linestyle=ls, linewidth=lw, label=label)

    ax3.set_title("FiLM Effect: x̂·(1+scale)+shift\n(different conditioning → different activations)",
                  fontsize=10)
    ax3.set_xlabel("Feature dimension", fontsize=9)
    ax3.set_ylabel("Activation value", fontsize=9)
    ax3.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax3.legend(fontsize=7.5, loc="lower right")
    ax3.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_causal_temporal_attention(save_path: str):
    """
    Visualize the causal temporal attention mask.

    Shows:
    - Left: Conceptual token layout — T frames × N patches in the sequence
    - Middle: Causal mask matrix (which tokens can attend to which)
    - Right: Why causality matters for autoregressive rollout
    """
    print("Generating causal temporal attention visualization...")

    T = 4    # frames
    N = 4    # patches per frame (small for clarity)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Token layout diagram
    ax1 = axes[0]
    ax1.set_xlim(-0.5, N + 0.5)
    ax1.set_ylim(-0.5, T + 0.2)
    ax1.set_title(f"Token Layout: T={T} frames × N={N} patches\n(total sequence length = T×N)",
                  fontsize=11)
    ax1.set_xlabel("Patch index within frame", fontsize=10)
    ax1.set_ylabel("Frame index (t)", fontsize=10)
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(T))

    colors = plt.cm.tab10(np.linspace(0, 0.6, T))
    for t in range(T):
        for n in range(N):
            rect = mpatches.FancyBboxPatch(
                (n - 0.4, t - 0.35), 0.8, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=colors[t], edgecolor="black", linewidth=1,
            )
            ax1.add_patch(rect)
            ax1.text(n, t, f"t{t}p{n}", ha="center", va="center", fontsize=7, color="white",
                     fontweight="bold")

    ax1.text(N / 2 - 0.5, T + 0.05,
             "Each frame adds N tokens; temporal order is left-to-right, bottom-to-top",
             ha="center", fontsize=8, style="italic")

    # Middle: Causal attention mask
    ax2 = axes[1]
    seq_len = T * N
    mask = np.zeros((seq_len, seq_len))

    # Causal temporal: token at position (t, n) can attend to
    # all patches in frames t' < t AND all patches in frame t (spatial self-attention)
    for i in range(seq_len):
        t_i = i // N
        for j in range(seq_len):
            t_j = j // N
            if t_j <= t_i:   # causal on frame index
                mask[i, j] = 1.0

    im = ax2.imshow(mask, cmap="Blues", vmin=0, vmax=1, origin="upper")
    ax2.set_title("Causal Temporal Attention Mask\n(blue = can attend, white = masked)",
                  fontsize=11)
    ax2.set_xlabel("Key (attended-to) frame", fontsize=10)
    ax2.set_ylabel("Query (attending) frame", fontsize=10)

    # Frame boundary lines
    for k in range(1, T):
        ax2.axhline(y=k * N - 0.5, color="red", linewidth=1.5, alpha=0.7)
        ax2.axvline(x=k * N - 0.5, color="red", linewidth=1.5, alpha=0.7)

    # Frame block tick labels — placed at the centre of each N-wide block
    tick_positions = [t * N + N / 2 - 0.5 for t in range(T)]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"frame {t}" for t in range(T)], color="darkred", fontsize=8)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels([f"frame {t}" for t in range(T)], color="darkred", fontsize=8)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Right: Why causality matters
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("Why Causal Masking Enables Rollout", fontsize=11)

    frames_info = [
        (5, 8.5, "Frame 0 tokens\n(context)", "lightblue"),
        (5, 6.5, "Frame 1 tokens\n(predict)", "lightgreen"),
        (5, 4.5, "Frame 2 tokens\n(predict)", "lightgreen"),
        (5, 2.5, "Frame 3 tokens\n(predict)", "lightgreen"),
    ]
    for x, y, label, color in frames_info:
        rect = mpatches.FancyBboxPatch(
            (x - 3.5, y - 0.65), 7, 1.3,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="black",
        )
        ax3.add_patch(rect)
        ax3.text(x, y, label, ha="center", va="center", fontsize=9)

    # Arrows showing which frames can be used
    for i in range(len(frames_info) - 1):
        ax3.annotate(
            "", xy=(frames_info[i + 1][0], frames_info[i + 1][1] + 0.7),
            xytext=(frames_info[i][0], frames_info[i][1] - 0.7),
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    ax3.text(5, 1.0, "Frame k can only see frames 0 … k\n→ safe to generate autoregressively",
             ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_teacher_forcing(save_path: str):
    """
    Visualize the teacher-forced next-token prediction training objective.

    Shows:
    - Left: Input/target token alignment (shift-by-one)
    - Middle: Per-patch cross-entropy loss heatmap across prediction steps
    - Right: Loss distribution vs. random-baseline
    """
    print("Generating teacher-forcing visualization...")

    torch.manual_seed(0)
    vocab_size = 256
    grid_size = 5
    num_patches = grid_size * grid_size  # 25
    T = 5  # frames → 4 prediction steps, 100 data points for histogram

    model = TokenDynamicsModel(
        vocab_size=vocab_size,
        num_patches=num_patches,
        embed_dim=64,
        num_heads=4,
        num_blocks=2,
        grid_size=grid_size,
        action_dim=3,
    )
    model.eval()

    tokens  = torch.randint(0, vocab_size, (1, T, num_patches))
    actions = torch.randn(1, T - 1, 3) * 0.5

    import torch.nn.functional as F
    with torch.no_grad():
        loss, logits = model.training_step(tokens, actions)

    target = tokens[:, 1:].long()  # (1, T-1, N)
    per_patch_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1),
        reduction="none",
    ).reshape(T - 1, num_patches)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Shift-by-one diagram — compact 4-pair layout
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Teacher-Forced Training\n(shift-by-one prediction)", fontsize=11)

    pairs = [
        ("Frame 0 + Action 0", "Frame 1"),
        ("Frame 1 + Action 1", "Frame 2"),
        ("Frame 2 + Action 2", "Frame 3"),
        ("Frame 3 + Action 3", "Frame 4"),
    ]
    row_h = 0.8
    gap = 0.45
    n_rows = len(pairs)
    total_h = n_rows * row_h + (n_rows - 1) * gap
    y_start = 5 + total_h / 2 + 1.0  # shift up to fill space

    for i, (inp, tgt) in enumerate(pairs):
        y = y_start - i * (row_h + gap)
        # Input box
        ax1.add_patch(mpatches.FancyBboxPatch(
            (0.3, y - row_h / 2), 5.8, row_h,
            boxstyle="round,pad=0.08", facecolor="lightblue", edgecolor="black"))
        ax1.text(3.2, y, inp, ha="center", va="center", fontsize=8.5)
        # Arrow
        ax1.annotate("", xy=(8.5, y), xytext=(6.1, y),
                     arrowprops=dict(arrowstyle="->", color="black"))
        # Target box
        ax1.add_patch(mpatches.FancyBboxPatch(
            (8.5, y - row_h / 2), 1.2, row_h,
            boxstyle="round,pad=0.06", facecolor="lightgreen", edgecolor="black"))
        ax1.text(9.1, y, tgt, ha="center", va="center", fontsize=7.5)

    ax1.text(5, 1.0, "Loss = CrossEntropy(logits_t, tokens_t+1)\naveraged over all patches and timesteps",
             ha="center", fontsize=8.5,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # Middle: Per-patch loss heatmap — T-1 grids side by side
    ax2 = axes[1]
    grid_data = per_patch_loss.numpy().reshape(T - 1, grid_size, grid_size)
    combined = np.concatenate([grid_data[t] for t in range(T - 1)], axis=1)
    im = ax2.imshow(combined, cmap="RdYlGn_r", aspect="auto")
    for t in range(1, T - 1):
        ax2.axvline(x=t * grid_size - 0.5, color="white", linewidth=1.5)
    ax2.set_xticks([grid_size // 2 + t * grid_size for t in range(T - 1)])
    ax2.set_xticklabels([f"→ f{t+1}" for t in range(T - 1)], fontsize=8)
    ax2.set_yticks([])
    ax2.set_title(f"Per-Patch CE Loss  ({grid_size}×{grid_size} patch grid per step)\n"
                  "red = high loss, green = low loss", fontsize=11)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="CE Loss")

    # Right: Loss distribution vs. random baseline
    ax3 = axes[2]
    all_losses = per_patch_loss.numpy().flatten()
    ax3.hist(all_losses, bins=25, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    baseline = np.log(vocab_size)
    ax3.axvline(x=baseline, color="red", linestyle="--", linewidth=1.5,
                label=f"Random baseline  log({vocab_size}) = {baseline:.2f}")
    ax3.axvline(x=all_losses.mean(), color="green", linestyle="--", linewidth=1.5,
                label=f"Mean loss = {all_losses.mean():.2f}")
    ax3.set_title("Per-Patch Loss Distribution\n(untrained model)", fontsize=11)
    ax3.set_xlabel("Cross-Entropy Loss", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {save_path}")


def visualize_rollout_pipeline(
    checkpoint_path: str,
    tokenizer_checkpoint: str,
    inverse_dynamics_checkpoint: str,
    data_path: str,
    save_path: str,
    num_steps: int = 3,
):
    """
    Visualize autoregressive rollout: start frame + actions → generated token sequence.

    Shows:
    - Row 1: Start frame (decoded from tokens)
    - Row 2: Action used at each step
    - Row 3: Generated frames (decoded from predicted tokens)
    """
    print("Generating rollout pipeline visualization...")

    try:
        import cv2
    except ImportError:
        print("  OpenCV not available, skipping rollout visualization")
        return

    from dynamics import get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = TokenDynamicsModel(
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load a single batch of token/action data
    dataloader = get_dataloader(
        data_type="folder",
        data_path=data_path,
        tokenizer_checkpoint=tokenizer_checkpoint,
        inverse_dynamics_checkpoint=inverse_dynamics_checkpoint,
        batch_size=1,
        num_frames=config.model.num_frames,
        frame_size=config.model.frame_size,
        num_patches=config.model.num_patches,
        vocab_size=config.model.vocab_size,
        action_dim=config.model.action_dim,
    )
    batch = next(iter(dataloader))
    tokens  = batch["tokens"].to(device)   # (1, T, N)
    actions = batch["actions"].to(device)  # (1, T-1, A)

    # Rollout: give model the first frame + num_steps actions
    start_tokens  = tokens[:, 0]                         # (1, N)
    plan_actions  = actions[:, :num_steps]               # (1, S, A)

    with torch.no_grad():
        generated = model.rollout(start_tokens, plan_actions, sample=False)  # (1, S+1, N)

    generated = generated[0].cpu()   # (S+1, N)
    gt_tokens = tokens[0].cpu()      # (T, N)
    act_np    = actions[0].cpu().numpy()  # (T-1, A)

    # Load tokenizer for decoding
    sys.path.insert(0, str(Path(__file__).parent.parent / "1.video-tokenizer"))
    from video_tokenizer import VideoTokenizer

    tok_checkpoint = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=False)
    tok_config = tok_checkpoint["config"]
    tokenizer = VideoTokenizer(
        in_channels=3,
        embed_dim=tok_config.model.embed_dim,
        patch_size=tok_config.model.patch_size,
        frame_size=tok_config.model.frame_size,
        latent_dim=tok_config.model.latent_dim,
        num_bins=tok_config.model.num_bins,
        num_heads=tok_config.model.num_heads,
        num_blocks=tok_config.model.num_blocks,
    )
    tokenizer.load_state_dict(tok_checkpoint["model_state_dict"])
    tokenizer.eval()

    def decode_tokens(tok_ids):
        """Decode (N,) integer token ids to an (H, W, 3) numpy image."""
        with torch.no_grad():
            return tokenizer.decode_tokens(tok_ids.unsqueeze(0).unsqueeze(0))[0, 0].permute(1, 2, 0).numpy().clip(0, 1)

    n_cols = num_steps + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))

    # Row 1: ground-truth frames
    for t in range(n_cols):
        ax = axes[0, t]
        img = decode_tokens(gt_tokens[t])
        ax.imshow(img)
        ax.set_title(f"GT Frame {t}", fontsize=11)
        ax.axis("off")
    axes[0, 0].set_ylabel("Ground Truth", fontsize=11, fontweight="bold", rotation=0,
                           ha="right", va="center")

    # Row 2: actions
    for t in range(n_cols):
        ax = axes[1, t]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        if t < num_steps:
            av = act_np[t]
            rect = mpatches.FancyBboxPatch(
                (1, 2), 8, 6, boxstyle="round,pad=0.3",
                facecolor="salmon", edgecolor="black", linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(5, 6.5, f"Action {t}", ha="center", va="center", fontsize=13, fontweight="bold")
            ax.text(5, 4.5, f"[{av[0]:+.2f}, {av[1]:+.2f}, {av[2]:+.2f}]",
                    ha="center", va="center", fontsize=10)
        else:
            ax.text(5, 5, "(end)", ha="center", va="center", fontsize=10, color="gray")
    axes[1, 0].set_ylabel("Actions", fontsize=11, fontweight="bold", rotation=0,
                           ha="right", va="center")

    # Row 3: rollout predictions
    for t in range(n_cols):
        ax = axes[2, t]
        img = decode_tokens(generated[t])
        ax.imshow(img)
        lbl = "Start" if t == 0 else f"Gen {t}"
        ax.set_title(lbl, fontsize=11)
        ax.axis("off")
    axes[2, 0].set_ylabel("Rollout", fontsize=11, fontweight="bold", rotation=0,
                           ha="right", va="center")

    plt.suptitle("Token Dynamics Rollout: tokens_t + action_t → tokens_t+1",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved to {save_path}")


def main():
    """Generate all visualizations."""
    output_dir = Path(__file__).parent.parent / "assets" / "3.dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating token dynamics pipeline visualizations...\n")

    visualize_token_vocabulary(str(output_dir / "token_vocabulary.png"))
    visualize_action_conditioning(str(output_dir / "action_conditioning.png"))
    visualize_causal_temporal_attention(str(output_dir / "causal_attention.png"))
    visualize_teacher_forcing(str(output_dir / "teacher_forcing.png"))

    # Optional: rollout visualization if checkpoints + data are available
    checkpoint_path     = Path(__file__).parent / "checkpoints" / "best_model.pt"
    tokenizer_ckpt      = Path(__file__).parent.parent / "1.video-tokenizer" / "checkpoints" / "best_model.pt"
    inv_dyn_ckpt        = Path(__file__).parent.parent / "2.inverse-dynamics" / "checkpoints" / "best_model.pt"
    data_path           = Path(__file__).parent / "data"

    if checkpoint_path.exists() and tokenizer_ckpt.exists() and inv_dyn_ckpt.exists() and data_path.exists():
        visualize_rollout_pipeline(
            checkpoint_path=str(checkpoint_path),
            tokenizer_checkpoint=str(tokenizer_ckpt),
            inverse_dynamics_checkpoint=str(inv_dyn_ckpt),
            data_path=str(data_path),
            save_path=str(output_dir / "rollout_pipeline.png"),
            num_steps=3,
        )
    else:
        print("  Checkpoints or data not found, skipping rollout visualization")

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
