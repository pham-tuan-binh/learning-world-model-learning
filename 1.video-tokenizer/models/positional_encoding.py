"""
Positional Encoding implementation from first principles.

Transformers are permutation-invariant - they don't inherently understand the
order of tokens. Positional encoding solves this by adding position-dependent
signals to the embeddings.

For video, we need three types of position information:
1. Spatial X: horizontal position of patch within frame
2. Spatial Y: vertical position of patch within frame
3. Temporal: which frame in the sequence

We use sinusoidal positional encoding as introduced in "Attention is All You Need":
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

This encoding has nice properties:
- Deterministic (no learnable parameters)
- Can extrapolate to longer sequences than seen during training
- Different frequencies capture different scales of position
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for 1D sequences.

    Args:
        embed_dim: Dimension of embeddings (D)
        max_len: Maximum sequence length to support

    The encoding is computed as:
        PE(pos, 2i)   = sin(pos / 10000^(2i/D))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
    """

    def __init__(self, embed_dim: int, max_len: int = 1024):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_len = max_len

        # Create position encoding matrix
        # pe: (max_len, embed_dim)
        pe = self._create_pe(max_len, embed_dim)

        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer("pe", pe)

    def _create_pe(self, max_len: int, embed_dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding matrix.

        Args:
            max_len: Maximum sequence length
            embed_dim: Embedding dimension

        Returns:
            pe: Positional encoding matrix, shape (max_len, embed_dim)
        """
        # Position indices: [0, 1, 2, ..., max_len-1]
        # position: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1).float()

        # Compute positional encoding
        # pe: (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)

        # Number of even and odd dimensions
        # For embed_dim=5: even [0,2,4] = 3, odd [1,3] = 2
        num_even = (embed_dim + 1) // 2  # ceil(embed_dim / 2)
        num_odd = embed_dim // 2         # floor(embed_dim / 2)

        # Dimension indices for even positions: [0, 2, 4, ...]
        # div_term[i] = 1 / 10000^(2i/embed_dim)
        div_term_even = torch.exp(
            torch.arange(0, num_even).float() * 2 * (-math.log(10000.0) / embed_dim)
        )

        # Even dimensions: sin
        pe[:, 0::2] = torch.sin(position * div_term_even)

        # Dimension indices for odd positions: [1, 3, 5, ...]
        if num_odd > 0:
            div_term_odd = torch.exp(
                torch.arange(0, num_odd).float() * 2 * (-math.log(10000.0) / embed_dim)
            )
            # Odd dimensions: cos
            pe[:, 1::2] = torch.cos(position * div_term_odd)

        return pe

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Look up positional encodings for given positions.

        Args:
            positions: Position indices, shape (*)
                       Values should be in [0, max_len)

        Returns:
            encodings: Positional encodings, shape (*, embed_dim)
        """
        # Use positions as indices into pre-computed table
        return self.pe[positions]


class SpatioTemporalPositionalEncoding(nn.Module):
    """
    Positional encoding for video data with spatial (x, y) and temporal positions.

    Splits the embedding dimension into three parts:
    - x position: ~1/3 of dimensions (horizontal position in frame)
    - y position: ~1/3 of dimensions (vertical position in frame)
    - t position: ~1/3 of dimensions (temporal frame index)

    This allows the model to distinguish patches by their spatial location
    within a frame AND their temporal location across frames.

    Args:
        embed_dim: Total embedding dimension (E)
        grid_size: Number of patches per row/column (G)
                   For 128x128 image with 8x8 patches: G = 16
        max_frames: Maximum number of frames to support
    """

    def __init__(
        self,
        embed_dim: int = 128,
        grid_size: int = 16,
        max_frames: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.max_frames = max_frames

        # Split embed_dim into three roughly equal parts for x, y, t
        # We allocate 2/3 to spatial (x + y) and 1/3 to temporal (t)
        # This emphasizes spatial structure while still encoding time
        self.spatial_dim = (embed_dim * 2) // 3  # For x and y combined
        self.temporal_dim = embed_dim - self.spatial_dim  # For t

        # Further split spatial into x and y
        self.x_dim = self.spatial_dim // 2
        self.y_dim = self.spatial_dim - self.x_dim

        # Create separate PE modules for each axis
        self.pe_x = SinusoidalPositionalEncoding(self.x_dim, max_len=grid_size)
        self.pe_y = SinusoidalPositionalEncoding(self.y_dim, max_len=grid_size)
        self.pe_t = SinusoidalPositionalEncoding(self.temporal_dim, max_len=max_frames)

        # Pre-compute spatial grid positions
        # For a grid_size x grid_size grid, we need x and y coordinates for each patch
        # x_positions: (grid_size^2,) with values [0,0,...,0, 1,1,...,1, ...]
        # y_positions: (grid_size^2,) with values [0,1,...,G-1, 0,1,...,G-1, ...]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing="ij",
        )
        # Flatten to 1D: (N,) where N = grid_size^2
        self.register_buffer("x_positions", x_coords.flatten())
        self.register_buffer("y_positions", y_coords.flatten())

    def forward(self, x: torch.Tensor, add_temporal: bool = True) -> torch.Tensor:
        """
        Add positional encoding to patch embeddings.

        Args:
            x: Patch embeddings
               Shape: (B, T, N, E)
               - B: batch size
               - T: number of frames
               - N: num_patches = grid_size^2
               - E: embed_dim
            add_temporal: Whether to add temporal positional encoding
                          Set False if you only want spatial encoding

        Returns:
            x_pe: Embeddings with positional encoding added
                  Shape: (B, T, N, E) (same as input)

        The encoding is added (not concatenated) to preserve dimensions.
        Each position (x, y, t) contributes to different parts of the embedding.
        """
        B, T, N, E = x.shape

        # Get spatial positional encodings
        # pe_x: (N, x_dim), pe_y: (N, y_dim)
        pe_x = self.pe_x(self.x_positions)  # (N, x_dim)
        pe_y = self.pe_y(self.y_positions)  # (N, y_dim)

        # Concatenate spatial encodings
        # pe_spatial: (N, spatial_dim) where spatial_dim = x_dim + y_dim
        pe_spatial = torch.cat([pe_x, pe_y], dim=-1)

        # Expand to match batch and time dimensions
        # (N, spatial_dim) -> (1, 1, N, spatial_dim) -> (B, T, N, spatial_dim)
        pe_spatial = pe_spatial.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Add spatial encoding to the first spatial_dim dimensions
        # This modifies the first 2/3 of the embedding
        x = x.clone()
        x[..., : self.spatial_dim] = x[..., : self.spatial_dim] + pe_spatial

        if add_temporal:
            # Get temporal positional encodings
            # t_positions: (T,) with values [0, 1, 2, ..., T-1]
            t_positions = torch.arange(T, device=x.device)
            pe_t = self.pe_t(t_positions)  # (T, temporal_dim)

            # Expand to match batch and patch dimensions
            # (T, temporal_dim) -> (1, T, 1, temporal_dim) -> (B, T, N, temporal_dim)
            pe_t = pe_t.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)

            # Add temporal encoding to the last temporal_dim dimensions
            # This modifies the last 1/3 of the embedding
            x[..., self.spatial_dim :] = x[..., self.spatial_dim :] + pe_t

        return x

    def get_spatial_encoding(self) -> torch.Tensor:
        """
        Get just the spatial positional encoding (for visualization/debugging).

        Returns:
            pe_spatial: Spatial encoding, shape (N, spatial_dim)
                        where N = grid_size^2
        """
        pe_x = self.pe_x(self.x_positions)
        pe_y = self.pe_y(self.y_positions)
        return torch.cat([pe_x, pe_y], dim=-1)


if __name__ == "__main__":
    # Quick test to verify the implementation
    print("Testing Positional Encoding...")

    # Test 1D sinusoidal PE
    print("\n--- 1D Sinusoidal PE ---")
    pe_1d = SinusoidalPositionalEncoding(embed_dim=64, max_len=100)
    positions = torch.arange(10)
    encodings = pe_1d(positions)
    print(f"Positions shape: {positions.shape}")
    print(f"Encodings shape: {encodings.shape}")

    # Test spatio-temporal PE
    print("\n--- Spatio-Temporal PE ---")
    pe_st = SpatioTemporalPositionalEncoding(
        embed_dim=128,
        grid_size=16,  # 16x16 = 256 patches
        max_frames=32,
    )
    print(f"Spatial dim (x+y): {pe_st.spatial_dim}")
    print(f"  - X dim: {pe_st.x_dim}")
    print(f"  - Y dim: {pe_st.y_dim}")
    print(f"Temporal dim: {pe_st.temporal_dim}")

    # Test with video embeddings
    # (B=2, T=4, N=256, E=128)
    x = torch.randn(2, 4, 256, 128)
    print(f"\nInput shape: {x.shape}")

    x_pe = pe_st(x, add_temporal=True)
    print(f"Output shape: {x_pe.shape}")

    # Verify shapes match
    assert x.shape == x_pe.shape, "Shapes should match!"

    # Check that PE was actually added (values should differ)
    diff = (x - x_pe).abs().sum().item()
    print(f"Total difference after adding PE: {diff:.2f}")
    assert diff > 0, "PE should change the values!"

    print("\nAll tests passed!")
