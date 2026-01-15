"""
Spatio-Temporal Transformer implementation from first principles.

The spatio-temporal transformer processes video data by applying two types of attention:

1. Spatial Attention: Each patch attends to all other patches within the SAME frame.
   This captures relationships between different parts of an image (e.g., objects, backgrounds).
   Shape transformation: (B, T, N, E) -> process N patches for each of B*T frame instances

2. Temporal Attention: Each patch attends to the SAME patch position across different frames.
   This captures motion and temporal dynamics (e.g., object movement, scene changes).
   Shape transformation: (B, T, N, E) -> process T frames for each of B*N patch positions
   Uses causal masking to prevent looking at future frames.

Key components built from scratch:
- RMSNorm: Root Mean Square Layer Normalization (simpler than LayerNorm)
- Multi-Head Attention: Q, K, V projections with scaled dot-product attention
- SwiGLU Feed-Forward: Gated linear unit with SiLU activation
- Causal masking for temporal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm - only scales, doesn't shift.
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale

    This is computationally cheaper than LayerNorm and works well in practice.
    Used in LLaMA and other modern architectures.

    Args:
        dim: Feature dimension to normalize over
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        # Learnable scale parameter, initialized to 1
        # scale: (dim,)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor, shape (*, dim)
               The last dimension is normalized

        Returns:
            Normalized tensor, same shape as input
        """
        # Compute RMS over last dimension
        # x^2: (*, dim)
        # mean(x^2): (*, 1) after keepdim
        # rms: (*, 1)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        # x / rms: (*, dim)
        # * self.scale: (*, dim) broadcast with (dim,)
        return (x / rms) * self.scale


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention from first principles.

    Splits the embedding dimension into multiple heads, computes attention
    independently for each head, then concatenates and projects back.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        embed_dim: Total embedding dimension (E)
        num_heads: Number of attention heads (H)
                   E must be divisible by H
        dropout: Dropout probability for attention weights (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim  # E - total embedding dimension
        self.num_heads = num_heads  # H - number of attention heads
        self.head_dim = embed_dim // num_heads  # d_k = E / H - dimension per head

        # Scale factor for attention scores
        # Prevents dot products from getting too large with high dimensions
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        # Each projects from E to E dimensions
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection after concatenating heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor, shape (B, S, E)
               - B: batch size
               - S: sequence length
               - E: embed_dim
            mask: Optional attention mask, shape (S, S) or (B, S, S)
                  True/1 values are MASKED (not attended to)
                  Used for causal attention

        Returns:
            Output tensor, shape (B, S, E)
        """
        B, S, E = x.shape

        # Project to Q, K, V
        # (B, S, E) -> (B, S, E)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (B, S, E) -> (B, S, H, d_k) -> (B, H, S, d_k)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # Q @ K^T: (B, H, S, d_k) @ (B, H, d_k, S) -> (B, H, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided (for causal attention)
        if mask is not None:
            # mask shape: (S, S) or (B, S, S)
            # Expand to (1, 1, S, S) or (B, 1, S, S) for broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            # Set masked positions to -inf so softmax gives 0
            attn_scores = attn_scores.masked_fill(mask.bool(), float("-inf"))

        # Softmax to get attention weights
        # (B, H, S, S)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (B, H, S, S) @ (B, H, S, d_k) -> (B, H, S, d_k)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)

        # Final output projection
        output = self.out_proj(attn_output)

        return output


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network from first principles.

    SwiGLU is a gated variant of the feed-forward network that uses:
        SwiGLU(x) = (SiLU(W1 @ x) * (W2 @ x)) @ W3

    where SiLU(x) = x * sigmoid(x) is the Swish activation.

    This is more expressive than standard FFN and is used in LLaMA, PaLM, etc.

    Args:
        embed_dim: Input/output dimension (E)
        hidden_dim: Hidden dimension (typically 4 * E or computed as 2/3 * 4 * E)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # If hidden_dim not specified, use 4 * embed_dim scaled by 2/3
        # This accounts for the gating which effectively halves the capacity
        if hidden_dim is None:
            hidden_dim = int(4 * embed_dim * 2 / 3)
            # Round to nearest multiple of 8 for efficiency
            hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # W1: gate projection (for sigmoid)
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        # W2: value projection (multiplied by gate)
        self.w2 = nn.Linear(embed_dim, hidden_dim, bias=False)
        # W3: output projection
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor, shape (*, E)

        Returns:
            Output tensor, shape (*, E)

        The computation is:
            1. gate = SiLU(W1 @ x) = W1 @ x * sigmoid(W1 @ x)
            2. value = W2 @ x
            3. hidden = gate * value (element-wise)
            4. output = W3 @ hidden
        """
        # Gate: SiLU activation (Swish)
        # SiLU(x) = x * sigmoid(x)
        gate = F.silu(self.w1(x))

        # Value (no activation)
        value = self.w2(x)

        # Gated hidden state
        hidden = gate * value
        hidden = self.dropout(hidden)

        # Output projection
        output = self.w3(hidden)

        return output


class SpatioTemporalBlock(nn.Module):
    """
    A single block of the Spatio-Temporal Transformer.

    Each block consists of:
    1. Spatial attention (attend within each frame)
    2. Temporal attention (attend across frames, causally)
    3. Feed-forward network

    All with residual connections and RMSNorm.

    Args:
        embed_dim: Embedding dimension (E)
        num_heads: Number of attention heads (H)
        hidden_dim: FFN hidden dimension (optional, defaults to ~8/3 * E)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Normalization layers (pre-norm architecture)
        self.norm_spatial = RMSNorm(embed_dim)
        self.norm_temporal = RMSNorm(embed_dim)
        self.norm_ffn = RMSNorm(embed_dim)

        # Attention layers
        self.spatial_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.temporal_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.ffn = SwiGLU(embed_dim, hidden_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply one spatio-temporal transformer block.

        Args:
            x: Input tensor, shape (B, T, N, E)
               - B: batch size
               - T: number of frames
               - N: number of patches per frame
               - E: embed_dim
            causal_mask: Optional causal mask for temporal attention
                         Shape: (T, T), True values are masked

        Returns:
            Output tensor, shape (B, T, N, E)
        """
        B, T, N, E = x.shape

        # === Spatial Attention ===
        # Attend within each frame: each patch attends to all patches in same frame
        # Reshape: (B, T, N, E) -> (B*T, N, E)
        # Use contiguous() to ensure tensor is contiguous after any prior operations
        x_spatial = x.contiguous().view(B * T, N, E)

        # Apply attention (no mask - all patches can attend to each other)
        x_spatial = self.norm_spatial(x_spatial)
        attn_out = self.spatial_attn(x_spatial)
        x_spatial = x_spatial + self.dropout(attn_out)  # Residual

        # Reshape back: (B*T, N, E) -> (B, T, N, E)
        x = x_spatial.view(B, T, N, E)

        # === Temporal Attention ===
        # Attend across frames: each patch attends to same patch position in other frames
        # Reshape: (B, T, N, E) -> (B*N, T, E)
        # Use contiguous() after permute to enable view
        x_temporal = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, E)

        # Apply attention with causal mask
        x_temporal = self.norm_temporal(x_temporal)
        attn_out = self.temporal_attn(x_temporal, mask=causal_mask)
        x_temporal = x_temporal + self.dropout(attn_out)  # Residual

        # Reshape back: (B*N, T, E) -> (B, N, T, E) -> (B, T, N, E)
        x = x_temporal.view(B, N, T, E).permute(0, 2, 1, 3).contiguous()

        # === Feed-Forward Network ===
        # Apply to each token independently
        x_ffn = self.norm_ffn(x)
        ffn_out = self.ffn(x_ffn)
        x = x + self.dropout(ffn_out)  # Residual

        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Stack of Spatio-Temporal Transformer blocks.

    Processes video data with alternating spatial and temporal attention.

    Args:
        embed_dim: Embedding dimension (E)
        num_heads: Number of attention heads per block
        num_blocks: Number of transformer blocks to stack
        hidden_dim: FFN hidden dimension (optional)
        dropout: Dropout probability
        causal_temporal: Whether to use causal masking in temporal attention
                         (True = can't look at future frames)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        causal_temporal: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.causal_temporal = causal_temporal

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Final normalization
        self.final_norm = RMSNorm(embed_dim)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal attention mask.

        The mask prevents attending to future positions.
        mask[i, j] = True if j > i (position j is in the future of position i)

        Args:
            seq_len: Sequence length (T - number of frames)
            device: Device to create mask on

        Returns:
            mask: Causal mask, shape (seq_len, seq_len)
                  Upper triangular matrix with True above diagonal
        """
        # Create upper triangular mask (True above diagonal)
        # [0, 1, 1, 1]
        # [0, 0, 1, 1]
        # [0, 0, 0, 1]
        # [0, 0, 0, 0]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatio-temporal transformer.

        Args:
            x: Input tensor, shape (B, T, N, E)
               - B: batch size
               - T: number of frames
               - N: number of patches per frame
               - E: embed_dim

        Returns:
            Output tensor, shape (B, T, N, E)
        """
        B, T, N, E = x.shape

        # Create causal mask for temporal attention if needed
        causal_mask = None
        if self.causal_temporal:
            causal_mask = self._create_causal_mask(T, x.device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        # Final normalization
        x = self.final_norm(x)

        return x


if __name__ == "__main__":
    # Quick tests to verify the implementation
    print("Testing Spatio-Temporal Transformer components...")

    # Test RMSNorm
    print("\n--- RMSNorm ---")
    norm = RMSNorm(128)
    x = torch.randn(2, 10, 128)
    y = norm(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    # Check output has unit RMS
    rms = y.pow(2).mean(dim=-1).sqrt()
    print(f"Output RMS (should be ~1): {rms.mean().item():.4f}")

    # Test MultiHeadAttention
    print("\n--- MultiHeadAttention ---")
    attn = MultiHeadAttention(embed_dim=128, num_heads=8)
    x = torch.randn(2, 16, 128)  # (B, S, E)
    y = attn(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test with causal mask
    mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
    y_causal = attn(x, mask=mask)
    print(f"Causal output shape: {y_causal.shape}")

    # Test SwiGLU
    print("\n--- SwiGLU ---")
    ffn = SwiGLU(embed_dim=128)
    print(f"Hidden dim: {ffn.hidden_dim}")
    x = torch.randn(2, 16, 128)
    y = ffn(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test SpatioTemporalBlock
    print("\n--- SpatioTemporalBlock ---")
    block = SpatioTemporalBlock(embed_dim=128, num_heads=8)
    x = torch.randn(2, 4, 256, 128)  # (B, T, N, E)
    y = block(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test full SpatioTemporalTransformer
    print("\n--- SpatioTemporalTransformer ---")
    transformer = SpatioTemporalTransformer(
        embed_dim=128,
        num_heads=8,
        num_blocks=4,
        causal_temporal=True,
    )
    x = torch.randn(2, 4, 256, 128)  # (B, T, N, E)
    print(f"Input shape: {x.shape}")
    y = transformer(x)
    print(f"Output shape: {y.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"Number of parameters: {num_params:,}")

    print("\nAll tests passed!")
