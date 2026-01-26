"""
Latent Action Model (Inverse Dynamics) implementation from first principles.

The core insight: if we can't get action labels, we can LEARN them.

Given two consecutive frames, what "action" transformed frame 1 into frame 2?
This is inverse dynamics - instead of: state + action -> next_state
We solve: state + next_state -> action

The model has three parts:
1. Encoder: Look at frame pairs and infer what action happened
2. Quantizer: Discretize the continuous action into a finite set of actions
3. Decoder: Given a frame and an action, predict the next frame

The training signal is reconstruction: if our inferred actions are good,
we should be able to use them to reconstruct the next frame.

Reference: Genie (Google DeepMind) - "Genie: Generative Interactive Environments"
https://arxiv.org/abs/2402.15391
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import components from video tokenizer package (folder 1)
from video_tokenizer import (
    PatchEmbedding,
    SpatioTemporalTransformer,
    FiniteScalarQuantizer,
    SpatioTemporalPositionalEncoding,
)


class LatentActionsEncoder(nn.Module):
    """
    Encoder that infers latent actions from pairs of consecutive frames.

    Given frames at time t and t+1, what action caused this transition?

    Architecture:
    1. Patch embedding: Convert frames to patch tokens
    2. Positional encoding: Add spatial and temporal position info
    3. ST-Transformer: Learn spatial-temporal relationships
    4. Pooling: Mean pool over patches (one action per frame transition)
    5. Action head: Combine adjacent frame features to predict action

    Args:
        frame_size: Height/width of input frames (H=W)
        patch_size: Size of each patch (P)
        embed_dim: Embedding dimension (E)
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        action_dim: Dimension of the latent action vector (A)
    """

    def __init__(
        self,
        frame_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        action_dim: int = 3,
    ):
        super().__init__()

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.grid_size = frame_size // patch_size

        # Patch embedding from video tokenizer
        # (B, T, C, H, W) -> (B, T, N, E)
        self.patch_embed = PatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size,
        )

        # Positional encoding from video tokenizer
        self.pos_encoding = SpatioTemporalPositionalEncoding(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            max_frames=32,
        )

        # Spatio-temporal transformer (causal for temporal)
        self.transformer = SpatioTemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            causal_temporal=True,
        )

        # Action prediction head
        # Takes concatenated features from frame t and t+1
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 4 * action_dim),
            nn.GELU(),
            nn.Linear(4 * action_dim, action_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Infer latent actions from frame sequences.

        Args:
            frames: Video frames, shape (B, T, C, H, W)

        Returns:
            actions: Inferred latent actions, shape (B, T-1, A)
        """
        batch_size, seq_len, C, H, W = frames.shape

        # Convert to patch embeddings
        embeddings = self.patch_embed(frames)  # (B, T, N, E)

        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)

        # Apply spatio-temporal transformer
        transformed = self.transformer(embeddings)  # (B, T, N, E)

        # Mean pool over patches
        pooled = transformed.mean(dim=2)  # (B, T, E)

        # Infer action from each adjacent pair of frames
        actions = []
        for t in range(seq_len - 1):
            combined = torch.cat([pooled[:, t], pooled[:, t + 1]], dim=1)
            action = self.action_head(combined)
            actions.append(action)

        actions = torch.stack(actions, dim=1)  # (B, T-1, A)
        return actions


class LatentActionsDecoder(nn.Module):
    """
    Decoder that predicts the next frame given current frame and action.

    This is the forward dynamics model:
    state(t) + action(t) -> state(t+1)

    Architecture:
    1. Patch embedding: Convert current frame to patches
    2. Action conditioning: Project action and add to embeddings
    3. Token masking: Mask tokens to force reliance on actions
    4. ST-Transformer: Process conditioned embeddings
    5. Frame head: Convert back to pixels

    Args:
        frame_size: Height/width of frames
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        action_dim: Dimension of action conditioning (A)
    """

    def __init__(
        self,
        frame_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        action_dim: int = 3,
    ):
        super().__init__()

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = frame_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size,
        )

        # Positional encoding
        self.pos_encoding = SpatioTemporalPositionalEncoding(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            max_frames=32,
        )

        # Action conditioning projection
        # Projects action from A -> E dimensions
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Transformer (non-causal since we're reconstructing)
        self.transformer = SpatioTemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            causal_temporal=False,  # Can see all frames during decoding
        )

        # Frame reconstruction head
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
            nn.Tanh(),
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Predict next frames given current frames and actions.

        Args:
            frames: Input video frames, shape (B, T, C, H, W)
            actions: Latent actions, shape (B, T-1, A)
            training: Whether in training mode (enables masking)

        Returns:
            pred_frames: Predicted next frames, shape (B, T-1, C, H, W)
        """
        B, T, C, H, W = frames.shape

        # Use frames 1 to T-1 to predict frames 2 to T
        input_frames = frames[:, :-1].contiguous()  # (B, T-1, C, H, W)
        seq_len = T - 1

        # Convert to patch embeddings
        video_embeddings = self.patch_embed(input_frames)  # (B, T-1, N, E)

        # Add positional encoding
        video_embeddings = self.pos_encoding(video_embeddings)

        # Project and add action conditioning
        # actions: (B, T-1, A) -> (B, T-1, E)
        action_embed = self.action_proj(actions)
        # Expand to patches: (B, T-1, E) -> (B, T-1, 1, E) -> (B, T-1, N, E)
        action_embed = action_embed.unsqueeze(2).expand(-1, -1, self.num_patches, -1)
        video_embeddings = video_embeddings + action_embed

        # Mask tokens during training to force reliance on actions
        if training and self.training:
            keep_rate = 0.0  # Mask all tokens except first frame
            keep = torch.rand(B, seq_len, self.num_patches, 1, device=frames.device) < keep_rate
            keep[:, 0] = True  # Never mask first frame
            video_embeddings = torch.where(
                keep,
                video_embeddings,
                self.mask_token.expand_as(video_embeddings),
            )

        # Apply transformer
        transformed = self.transformer(video_embeddings)  # (B, T-1, N, E)

        # Reconstruct frames
        patches = self.frame_head(transformed)  # (B, T-1, N, 3*P*P)

        # Reshape patches to images
        P = self.patch_size
        patches = patches.view(B, seq_len, self.grid_size, self.grid_size, 3, P, P)
        pred_frames = patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        pred_frames = pred_frames.view(B, seq_len, 3, H, W)

        # Scale from [-1, 1] to [0, 1]
        pred_frames = (pred_frames + 1) / 2

        return pred_frames


class LatentActionModel(nn.Module):
    """
    Complete Latent Action Model combining encoder, quantizer, and decoder.

    The model learns to:
    1. Infer latent actions from frame pairs (encoder)
    2. Discretize actions to a finite vocabulary (quantizer)
    3. Use actions to predict next frames (decoder)

    Args:
        frame_size: Height/width of frames
        n_actions: Size of discrete action vocabulary (must be power of 2)
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
    """

    NUM_LATENT_ACTIONS_BINS = 2  # Binary quantization

    def __init__(
        self,
        frame_size: int = 128,
        n_actions: int = 8,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
    ):
        super().__init__()

        # Compute action dimension from vocabulary size
        assert math.log(n_actions, self.NUM_LATENT_ACTIONS_BINS).is_integer(), (
            f"n_actions must be a power of {self.NUM_LATENT_ACTIONS_BINS}"
        )
        self.action_dim = int(math.log(n_actions, self.NUM_LATENT_ACTIONS_BINS))
        self.n_actions = n_actions

        # Encoder: frames -> continuous action latents
        self.encoder = LatentActionsEncoder(
            frame_size=frame_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            action_dim=self.action_dim,
        )

        # Quantizer: continuous -> discrete actions (from video tokenizer)
        self.quantizer = FiniteScalarQuantizer(
            latent_dim=self.action_dim,
            num_bins=self.NUM_LATENT_ACTIONS_BINS,
        )

        # Decoder: frame + action -> next frame
        self.decoder = LatentActionsDecoder(
            frame_size=frame_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            action_dim=self.action_dim,
        )

        # Variance regularization (prevents action collapse)
        self.var_target = 0.01
        self.var_lambda = 100.0

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: infer actions and predict next frames.

        Args:
            frames: Video frames, shape (B, T, C, H, W)

        Returns:
            total_loss: Combined reconstruction + variance loss
            pred_frames: Predicted next frames, shape (B, T-1, C, H, W)
        """
        # Infer continuous action latents
        action_latents = self.encoder(frames)  # (B, T-1, A)

        # Quantize to discrete actions
        action_latents_quantized, _ = self.quantizer(action_latents)

        # Predict next frames
        pred_frames = self.decoder(frames, action_latents_quantized, training=True)

        # Reconstruction loss
        target_frames = frames[:, 1:]
        recon_loss = F.smooth_l1_loss(pred_frames, target_frames)

        # Variance loss (prevent collapse)
        z_var = action_latents.var(dim=0, unbiased=False).mean()
        var_penalty = F.relu(self.var_target - z_var)

        total_loss = recon_loss + self.var_lambda * var_penalty

        return total_loss, pred_frames

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to discrete actions."""
        action_latents = self.encoder(frames)
        action_latents_quantized, _ = self.quantizer(action_latents)
        return action_latents_quantized

    def decode(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode frame + action to next frame."""
        return self.decoder(frames, actions, training=False)


if __name__ == "__main__":
    print("Testing Latent Action Model...")

    model = LatentActionModel(
        frame_size=128,
        n_actions=8,
        patch_size=8,
        embed_dim=128,
        num_heads=8,
        num_blocks=4,
    )

    print(f"Action dim: {model.action_dim}")
    print(f"Number of actions: {model.n_actions}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    frames = torch.randn(2, 4, 3, 128, 128)
    loss, pred = model(frames)
    print(f"\nInput: {frames.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Predicted: {pred.shape}")

    print("\nAll tests passed!")
