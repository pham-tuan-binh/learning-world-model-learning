"""
Video Tokenizer implementation from first principles.

The Video Tokenizer is the main model that combines all components:
1. Patch Embedding: Convert frames to patch sequences
2. Positional Encoding: Add spatial and temporal position information
3. Spatio-Temporal Transformer: Process with attention
4. FSQ Quantizer: Discretize to tokens
5. Decoder: Reconstruct frames from tokens

This is based on the Genie architecture and implements an autoencoder that
compresses video frames into discrete tokens, which can later be used for
training a dynamics model (world model).

The key insight is that we're creating a "vocabulary" of visual tokens that
can represent any video frame, similar to how text tokens represent language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .patch_embed import PatchEmbedding, PatchUnembedding
from .positional_encoding import SpatioTemporalPositionalEncoding
from .st_transformer import SpatioTemporalTransformer
from .fsq import FiniteScalarQuantizer


class VideoTokenizerEncoder(nn.Module):
    """
    Encoder part of the Video Tokenizer.

    Takes video frames and produces discrete token indices.

    Pipeline:
        frames -> patches -> spatial_pe -> st_transformer -> linear -> fsq -> tokens

    Args:
        in_channels: Number of input channels (C), typically 3 for RGB
        frame_size: Height/width of input frames (H=W)
        num_frames: Number of frames in temporal context (T)
        patch_size: Size of each square patch (P)
        embed_dim: Dimension of patch embeddings (E)
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        latent_dim: Number of dimensions for FSQ (D)
        num_bins: Number of quantization levels per dimension (L)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        frame_size: int = 128,
        num_frames: int = 4,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        latent_dim: int = 5,
        num_bins: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store configuration
        self.in_channels = in_channels
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_bins = num_bins

        # Derived dimensions
        self.grid_size = frame_size // patch_size  # G = H/P (e.g., 128/8 = 16)
        self.num_patches = self.grid_size ** 2     # N = G^2 (e.g., 256)

        # Patch embedding: (B, T, C, H, W) -> (B, T, N, E)
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size,
        )

        # Positional encoding
        self.pos_encoding = SpatioTemporalPositionalEncoding(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            max_frames=32,  # Support up to 32 frames
        )

        # Spatio-temporal transformer (with causal temporal attention)
        self.transformer = SpatioTemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
            causal_temporal=True,  # Can't look at future frames
        )

        # Linear projection to latent space
        # (B, T, N, E) -> (B, T, N, D)
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        # FSQ quantizer
        self.quantizer = FiniteScalarQuantizer(
            latent_dim=latent_dim,
            num_bins=num_bins,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode video frames to quantized latents and token indices.

        Args:
            x: Video frames
               Shape: (B, T, C, H, W)
               - B: batch size
               - T: num_frames (temporal dimension)
               - C: channels (3 for RGB)
               - H: height (frame_size)
               - W: width (frame_size)

        Returns:
            z_q: Quantized latent vectors
                 Shape: (B, T, N, D)
                 - N: num_patches
                 - D: latent_dim
            indices: Token indices (discrete)
                     Shape: (B, T, N)
                     Values in [0, codebook_size)
            z: Pre-quantization latents (for analysis)
               Shape: (B, T, N, D)
        """
        # Step 1: Convert to patch embeddings
        # (B, T, C, H, W) -> (B, T, N, E)
        patches = self.patch_embed(x)

        # Step 2: Add positional encoding
        # (B, T, N, E) -> (B, T, N, E)
        patches = self.pos_encoding(patches, add_temporal=True)

        # Step 3: Apply spatio-temporal transformer
        # (B, T, N, E) -> (B, T, N, E)
        features = self.transformer(patches)

        # Step 4: Project to latent space
        # (B, T, N, E) -> (B, T, N, D)
        z = self.to_latent(features)

        # Step 5: Quantize with FSQ
        # z: (B, T, N, D) -> z_q: (B, T, N, D), indices: (B, T, N)
        z_q, indices = self.quantizer(z)

        return z_q, indices, z


class VideoTokenizerDecoder(nn.Module):
    """
    Decoder part of the Video Tokenizer.

    Takes quantized latents and reconstructs video frames.

    Pipeline:
        z_q -> embed -> spatial_pe -> st_transformer -> patch_unembed -> frames

    Args:
        out_channels: Number of output channels (C), typically 3 for RGB
        frame_size: Height/width of output frames (H=W)
        num_frames: Number of frames in temporal context (T)
        patch_size: Size of each square patch (P)
        embed_dim: Dimension of patch embeddings (E)
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        latent_dim: Number of dimensions from FSQ (D)
        dropout: Dropout probability
    """

    def __init__(
        self,
        out_channels: int = 3,
        frame_size: int = 128,
        num_frames: int = 4,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        latent_dim: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store configuration
        self.out_channels = out_channels
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Derived dimensions
        self.grid_size = frame_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Embed quantized latents back to embedding dimension
        # (B, T, N, D) -> (B, T, N, E)
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        # Positional encoding (same structure as encoder)
        self.pos_encoding = SpatioTemporalPositionalEncoding(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            max_frames=32,
        )

        # Spatio-temporal transformer (non-causal for decoding)
        # The decoder can see all positions since we're reconstructing
        self.transformer = SpatioTemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
            causal_temporal=False,  # Decoder sees all frames
        )

        # Patch unembedding: (B, T, N, E) -> (B, T, C, H, W)
        self.patch_unembed = PatchUnembedding(
            embed_dim=embed_dim,
            out_channels=out_channels,
            patch_size=patch_size,
            frame_size=frame_size,
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to video frames.

        Args:
            z_q: Quantized latent vectors
                 Shape: (B, T, N, D)
                 - B: batch size
                 - T: num_frames
                 - N: num_patches
                 - D: latent_dim

        Returns:
            x_hat: Reconstructed video frames
                   Shape: (B, T, C, H, W)
                   - C: channels (3 for RGB)
                   - H: height (frame_size)
                   - W: width (frame_size)
        """
        # Step 1: Project from latent to embedding dimension
        # (B, T, N, D) -> (B, T, N, E)
        embeddings = self.from_latent(z_q)

        # Step 2: Add positional encoding
        # (B, T, N, E) -> (B, T, N, E)
        embeddings = self.pos_encoding(embeddings, add_temporal=True)

        # Step 3: Apply spatio-temporal transformer
        # (B, T, N, E) -> (B, T, N, E)
        features = self.transformer(embeddings)

        # Step 4: Convert patches back to frames
        # (B, T, N, E) -> (B, T, C, H, W)
        x_hat = self.patch_unembed(features)

        return x_hat


class VideoTokenizer(nn.Module):
    """
    Complete Video Tokenizer (Encoder + Decoder).

    This is the main model that:
    1. Encodes video frames into discrete tokens
    2. Decodes tokens back to video frames
    3. Computes reconstruction loss for training

    The discrete tokens can be used for:
    - Training a dynamics/world model (predict future tokens)
    - Video compression
    - Video generation

    Args:
        in_channels: Number of channels (C), typically 3 for RGB
        frame_size: Height/width of frames (H=W)
        num_frames: Number of frames in temporal context (T)
        patch_size: Size of each square patch (P)
        embed_dim: Dimension of patch embeddings (E)
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        latent_dim: Number of dimensions for FSQ (D)
        num_bins: Number of quantization levels per dimension (L)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        frame_size: int = 128,
        num_frames: int = 4,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        latent_dim: int = 5,
        num_bins: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store configuration for easy access
        self.config = {
            "in_channels": in_channels,
            "frame_size": frame_size,
            "num_frames": num_frames,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "latent_dim": latent_dim,
            "num_bins": num_bins,
            "dropout": dropout,
        }

        # Derived values
        self.grid_size = frame_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.codebook_size = num_bins ** latent_dim

        # Encoder
        self.encoder = VideoTokenizerEncoder(
            in_channels=in_channels,
            frame_size=frame_size,
            num_frames=num_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            latent_dim=latent_dim,
            num_bins=num_bins,
            dropout=dropout,
        )

        # Decoder
        self.decoder = VideoTokenizerDecoder(
            out_channels=in_channels,
            frame_size=frame_size,
            num_frames=num_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            x: Video frames
               Shape: (B, T, C, H, W)
               Values should be normalized to [0, 1] or [-1, 1]

        Returns:
            loss: Reconstruction loss (MSE between input and output)
                  Scalar tensor
            x_hat: Reconstructed video frames
                   Shape: (B, T, C, H, W)
            indices: Token indices
                     Shape: (B, T, N)
                     Values in [0, codebook_size)
        """
        # Encode to quantized latents
        # z_q: (B, T, N, D), indices: (B, T, N)
        z_q, indices, _ = self.encoder(x)

        # Decode back to frames
        # x_hat: (B, T, C, H, W)
        x_hat = self.decoder(z_q)

        # Compute reconstruction loss (MSE)
        loss = F.mse_loss(x_hat, x)

        return loss, x_hat, indices

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video frames to token indices (for inference).

        Args:
            x: Video frames, shape (B, T, C, H, W)

        Returns:
            indices: Token indices, shape (B, T, N)
            z_q: Quantized latents, shape (B, T, N, D)
        """
        z_q, indices, _ = self.encoder(x)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to video frames.

        Args:
            z_q: Quantized latents, shape (B, T, N, D)

        Returns:
            x_hat: Reconstructed frames, shape (B, T, C, H, W)
        """
        return self.decoder(z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from token indices directly.

        Args:
            indices: Token indices, shape (B, T, N)

        Returns:
            x_hat: Reconstructed frames, shape (B, T, C, H, W)
        """
        # Convert indices to quantized latents
        z_q = self.encoder.quantizer.indices_to_latent(indices)
        return self.decoder(z_q)

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute histogram of codebook usage.

        Useful for monitoring if all tokens are being used (no codebook collapse).

        Args:
            indices: Token indices, shape (*)

        Returns:
            usage: Histogram of token usage, shape (codebook_size,)
        """
        flat_indices = indices.flatten()
        usage = torch.bincount(flat_indices, minlength=self.codebook_size)
        return usage.float()


if __name__ == "__main__":
    # Quick test to verify the implementation
    print("Testing Video Tokenizer...")

    # Create model with default settings
    model = VideoTokenizer(
        in_channels=3,
        frame_size=128,
        num_frames=4,
        patch_size=8,
        embed_dim=128,
        num_heads=8,
        num_blocks=4,
        latent_dim=5,
        num_bins=4,
        dropout=0.0,
    )

    print(f"\nModel configuration:")
    print(f"  Grid size: {model.grid_size}x{model.grid_size}")
    print(f"  Num patches per frame: {model.num_patches}")
    print(f"  Codebook size: {model.codebook_size}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Test forward pass
    print("\n--- Forward Pass ---")
    x = torch.randn(2, 4, 3, 128, 128)  # (B=2, T=4, C=3, H=128, W=128)
    print(f"Input shape: {x.shape}")

    loss, x_hat, indices = model(x)
    print(f"Loss: {loss.item():.4f}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Index range: [{indices.min().item()}, {indices.max().item()}]")

    # Test encode/decode
    print("\n--- Encode/Decode ---")
    indices, z_q = model.encode(x)
    print(f"Encoded indices shape: {indices.shape}")
    print(f"Quantized latents shape: {z_q.shape}")

    x_reconstructed = model.decode(z_q)
    print(f"Decoded shape: {x_reconstructed.shape}")

    # Test decode from indices
    x_from_indices = model.decode_indices(indices)
    print(f"Decoded from indices shape: {x_from_indices.shape}")

    # Check reconstruction consistency
    diff = (x_reconstructed - x_from_indices).abs().max().item()
    print(f"Difference between decode methods: {diff}")

    # Check codebook usage
    print("\n--- Codebook Usage ---")
    usage = model.get_codebook_usage(indices)
    num_used = (usage > 0).sum().item()
    print(f"Tokens used: {num_used}/{model.codebook_size}")
    print(f"Usage distribution: min={usage.min().item():.0f}, "
          f"max={usage.max().item():.0f}, mean={usage.mean().item():.1f}")

    print("\nAll tests passed!")
