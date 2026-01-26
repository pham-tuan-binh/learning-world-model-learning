"""
Patch Embedding implementation from first principles.

The patch embedding converts an image into a sequence of patch embeddings,
similar to how ViT (Vision Transformer) works. Instead of processing pixels
directly, we divide the image into non-overlapping patches and project each
patch to a vector representation.

Why patches?
1. Reduces sequence length: 128x128 image with 8x8 patches = 256 tokens (not 16384)
2. Each patch captures local spatial structure
3. Efficient parallel processing with Conv2D

For a video, we apply patch embedding to each frame independently, then combine
them with temporal attention later.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Convert video frames into patch embeddings.

    Uses a Conv2D with kernel_size=stride=patch_size to extract non-overlapping
    patches and project them to embed_dim dimensions.

    Args:
        in_channels: Number of input channels (C), typically 3 for RGB
        embed_dim: Dimension of patch embeddings (E)
        patch_size: Size of each square patch (P), e.g., 8 means 8x8 patches
        frame_size: Height/width of input frames (H=W), e.g., 128

    Input/Output dimensions:
        Input:  (B, T, C, H, W) - batch, time, channels, height, width
        Output: (B, T, N, E) - batch, time, num_patches, embed_dim
                where N = (H/P) * (W/P) = number of patches per frame
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: int = 8,
        frame_size: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels  # C - input channels (3 for RGB)
        self.embed_dim = embed_dim      # E - embedding dimension
        self.patch_size = patch_size    # P - patch height/width
        self.frame_size = frame_size    # H = W - frame height/width

        # Number of patches per row/column: H/P (e.g., 128/8 = 16)
        self.grid_size = frame_size // patch_size

        # Total number of patches per frame: (H/P)^2 (e.g., 16^2 = 256)
        self.num_patches = self.grid_size ** 2

        # Projection layer: Conv2D with kernel=stride=patch_size
        # This extracts non-overlapping patches and projects them in one step
        # Input:  (*, C, H, W)
        # Output: (*, E, H/P, W/P) = (*, E, grid_size, grid_size)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert video frames to patch embeddings.

        Args:
            x: Video frames
               Shape: (B, T, C, H, W)
               - B: batch size
               - T: number of frames (temporal dimension)
               - C: channels (3 for RGB)
               - H: height (frame_size)
               - W: width (frame_size)

        Returns:
            patches: Patch embeddings
                     Shape: (B, T, N, E)
                     - N: num_patches = (H/P)^2
                     - E: embed_dim

        Example dimensions with default settings:
            Input:  (B, 4, 3, 128, 128)
            Output: (B, 4, 256, 128)
            where 256 = (128/8)^2 patches, 128 = embed_dim
        """
        B, T, C, H, W = x.shape

        # Reshape to merge batch and time dimensions for Conv2D
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Apply patch projection
        # (B*T, C, H, W) -> (B*T, E, H/P, W/P) = (B*T, E, grid_size, grid_size)
        x = self.proj(x)

        # Reshape to (B*T, E, N) where N = grid_size^2
        # First flatten the spatial dimensions
        # (B*T, E, grid_size, grid_size) -> (B*T, E, N)
        x = x.flatten(start_dim=2)

        # Transpose to (B*T, N, E) - patches as sequence, embed as features
        x = x.transpose(1, 2)

        # Reshape back to separate batch and time
        # (B*T, N, E) -> (B, T, N, E)
        x = x.view(B, T, self.num_patches, self.embed_dim)

        return x


class PatchUnembedding(nn.Module):
    """
    Convert patch embeddings back to image pixels.

    This is the inverse of PatchEmbedding. Uses a linear projection followed
    by reshaping to reconstruct the original image dimensions.

    Note: We use pixel shuffle for more efficient upsampling, which produces
    better quality reconstructions than simple reshaping.

    Args:
        embed_dim: Dimension of patch embeddings (E)
        out_channels: Number of output channels (C), typically 3 for RGB
        patch_size: Size of each square patch (P)
        frame_size: Height/width of output frames (H=W)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        out_channels: int = 3,
        patch_size: int = 8,
        frame_size: int = 128,
    ):
        super().__init__()

        self.embed_dim = embed_dim       # E - embedding dimension
        self.out_channels = out_channels  # C - output channels (3 for RGB)
        self.patch_size = patch_size     # P - patch height/width
        self.frame_size = frame_size     # H = W - frame height/width

        self.grid_size = frame_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Project embeddings to pixel values
        # Each patch embedding (E) is projected to (C * P * P) values
        # Then reshaped to (C, P, P) to form the patch image
        # We use pixel shuffle which requires (C * r^2) channels, where r = patch_size
        # Output channels of projection: C * P^2 (e.g., 3 * 64 = 192 for 8x8 patches)
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        # PixelShuffle rearranges (*, C*r^2, H, W) -> (*, C, H*r, W*r)
        # We use it to go from (grid_size, grid_size) -> (frame_size, frame_size)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch embeddings back to video frames.

        Args:
            x: Patch embeddings
               Shape: (B, T, N, E)
               - B: batch size
               - T: number of frames
               - N: num_patches = (H/P)^2
               - E: embed_dim

        Returns:
            frames: Reconstructed video frames
                    Shape: (B, T, C, H, W)
                    - C: channels (3 for RGB)
                    - H: height (frame_size)
                    - W: width (frame_size)
        """
        B, T, N, E = x.shape

        # Project to pixel values
        # (B, T, N, E) -> (B, T, N, C * P^2)
        x = self.proj(x)

        # Reshape to (B*T, N, C * P^2)
        x = x.view(B * T, N, -1)

        # Reshape to spatial grid for pixel shuffle
        # (B*T, N, C * P^2) -> (B*T, C * P^2, grid_size, grid_size)
        # N = grid_size^2, so we can reshape to (grid_size, grid_size)
        x = x.transpose(1, 2)  # (B*T, C * P^2, N)
        x = x.view(B * T, -1, self.grid_size, self.grid_size)

        # Apply pixel shuffle to upsample
        # (B*T, C * P^2, grid_size, grid_size) -> (B*T, C, H, W)
        x = self.pixel_shuffle(x)

        # Reshape back to separate batch and time
        # (B*T, C, H, W) -> (B, T, C, H, W)
        x = x.view(B, T, self.out_channels, self.frame_size, self.frame_size)

        return x


if __name__ == "__main__":
    # Quick test to verify the implementation
    print("Testing Patch Embedding/Unembedding...")

    # Create modules with default settings
    embed = PatchEmbedding(
        in_channels=3,
        embed_dim=128,
        patch_size=8,
        frame_size=128,
    )
    unembed = PatchUnembedding(
        embed_dim=128,
        out_channels=3,
        patch_size=8,
        frame_size=128,
    )

    print(f"Grid size: {embed.grid_size}x{embed.grid_size}")
    print(f"Num patches: {embed.num_patches}")

    # Test with random video
    # (B=2, T=4, C=3, H=128, W=128)
    x = torch.randn(2, 4, 3, 128, 128)
    print(f"Input shape: {x.shape}")

    # Embed to patches
    patches = embed(x)
    print(f"Patches shape: {patches.shape}")

    # Unembed back to frames
    x_reconstructed = unembed(patches)
    print(f"Reconstructed shape: {x_reconstructed.shape}")

    # Verify shapes match
    assert x.shape == x_reconstructed.shape, "Shapes should match!"
    print("All tests passed!")
