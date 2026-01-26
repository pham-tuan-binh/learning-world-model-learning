"""
Finite Scalar Quantization (FSQ) implementation from first principles.

FSQ is a simpler alternative to VQ-VAE that avoids codebook collapse without
requiring auxiliary losses. The key insight is to directly quantize each
dimension of the latent vector to a finite set of values.

Reference: "Language Model Beats Diffusion - Tokenizer is Key to Visual Generation"
https://arxiv.org/abs/2310.05737

How FSQ works:
1. Bound the continuous latent values to [-1, 1] using tanh
2. Scale to [0, num_bins-1] and round to nearest integer
3. Use straight-through estimator (STE) for gradient flow during backprop
4. The codebook size is num_bins^latent_dim (e.g., 4^5 = 1024 tokens)
"""

import torch
import torch.nn as nn
from typing import Tuple


class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ) module.

    Quantizes continuous latent vectors to discrete tokens by independently
    quantizing each dimension to a finite set of values.

    Args:
        latent_dim: Number of dimensions in the latent space (D)
                    Each dimension is quantized independently
        num_bins: Number of quantization levels per dimension (L)
                  Higher = finer granularity but larger codebook

    The total codebook size is num_bins^latent_dim.
    For latent_dim=5, num_bins=4: codebook_size = 4^5 = 1024 tokens
    """

    def __init__(self, latent_dim: int = 5, num_bins: int = 4):
        super().__init__()

        self.latent_dim = latent_dim  # D - number of quantized dimensions
        self.num_bins = num_bins      # L - quantization levels per dimension

        # Total number of unique tokens in the codebook
        # codebook_size = L^D (e.g., 4^5 = 1024)
        self.codebook_size = num_bins ** latent_dim

        # Pre-compute the positional basis for index conversion
        # basis: [D] where basis[i] = num_bins^i
        # Used to convert between multi-dimensional indices and flat indices
        # Example for D=3, L=4: basis = [1, 4, 16]
        # index = d0 * 1 + d1 * 4 + d2 * 16
        basis = torch.tensor([num_bins ** i for i in range(latent_dim)])
        self.register_buffer("basis", basis)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent vectors to discrete values.

        Args:
            z: Continuous latent vectors
               Shape: (*, D) where * is any batch dimensions and D = latent_dim
               Example: (B, T, N, D) for video with B=batch, T=frames, N=patches

        Returns:
            z_q: Quantized latent vectors (same shape as z)
                 Values are in [-1, 1] but discrete (only L possible values)
            indices: Flat codebook indices
                     Shape: (*) - same as z but without last dimension
                     Values are integers in [0, codebook_size)

        The forward pass uses straight-through estimator (STE):
        - Forward: use quantized values
        - Backward: gradients flow through as if no quantization happened
        """
        # Step 1: Bound values to [-1, 1] using tanh
        # This ensures our inputs are in a known range for quantization
        # z_bounded: (*, D) with values in [-1, 1]
        z_bounded = torch.tanh(z)

        # Step 2: Scale from [-1, 1] to [0, num_bins - 1]
        # Linear transformation: x in [-1,1] -> (x+1)/2 * (L-1) in [0, L-1]
        # z_scaled: (*, D) with values in [0, num_bins - 1]
        z_scaled = (z_bounded + 1) / 2 * (self.num_bins - 1)

        # Step 3: Round to nearest integer (quantization)
        # z_rounded: (*, D) with integer values in {0, 1, ..., num_bins - 1}
        z_rounded = torch.round(z_scaled)

        # Step 4: Straight-through estimator (STE)
        # During forward pass: use quantized values
        # During backward pass: pretend rounding didn't happen (gradients flow through)
        # This is achieved by: z_q_scaled = z_scaled + (z_rounded - z_scaled).detach()
        z_q_scaled = z_scaled + (z_rounded - z_scaled).detach()

        # Step 5: Unscale back to [-1, 1]
        # Inverse of step 2: x in [0, L-1] -> x / (L-1) * 2 - 1 in [-1, 1]
        # z_q: (*, D) with values in {-1, -1+2/(L-1), ..., 1}
        z_q = z_q_scaled / (self.num_bins - 1) * 2 - 1

        # Step 6: Convert quantized values to flat indices
        # Each combination of D quantized values maps to a unique index
        # indices: (*) - integer indices in [0, codebook_size)
        indices = self._latent_to_indices(z_rounded.long())

        return z_q, indices

    def _latent_to_indices(self, z_discrete: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete latent vectors to flat codebook indices.

        Each dimension d has a value in {0, 1, ..., num_bins-1}.
        The flat index is computed as: sum(z[d] * basis[d]) for d in [0, D)

        Args:
            z_discrete: Discrete values per dimension
                        Shape: (*, D) with values in {0, ..., num_bins-1}

        Returns:
            indices: Flat indices
                     Shape: (*) with values in {0, ..., codebook_size-1}

        Example for D=3, L=4, z_discrete=[2, 1, 3]:
            index = 2 * 1 + 1 * 4 + 3 * 16 = 2 + 4 + 48 = 54
        """
        # Dot product along last dimension with basis vector
        # z_discrete: (*, D), self.basis: (D,)
        # indices: (*)
        indices = (z_discrete * self.basis).sum(dim=-1)
        return indices

    def indices_to_latent(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert flat codebook indices back to quantized latent vectors.

        This is the inverse of _latent_to_indices. We extract each dimension
        by repeated division and modulo operations.

        Args:
            indices: Flat indices
                     Shape: (*) with values in {0, ..., codebook_size-1}

        Returns:
            z_q: Quantized latent vectors
                 Shape: (*, D) with values in [-1, 1] (discrete)

        Example for D=3, L=4, index=54:
            d0 = 54 % 4 = 2
            d1 = (54 // 4) % 4 = 13 % 4 = 1
            d2 = (54 // 16) % 4 = 3 % 4 = 3
            z_discrete = [2, 1, 3]
        """
        # Add dimension for latent_dim
        # indices: (*) -> (*, 1)
        idx = indices.unsqueeze(-1)

        # Extract each dimension using integer division and modulo
        # For dimension d: value = (index // num_bins^d) % num_bins
        # z_discrete: (*, D)
        z_discrete = (idx // self.basis) % self.num_bins

        # Convert from {0, ..., num_bins-1} to [-1, 1]
        z_q = z_discrete.float() / (self.num_bins - 1) * 2 - 1

        return z_q


if __name__ == "__main__":
    # Quick test to verify the implementation
    print("Testing Finite Scalar Quantizer...")

    # Create quantizer with default settings
    # latent_dim=5, num_bins=4 -> 1024 possible tokens
    fsq = FiniteScalarQuantizer(latent_dim=5, num_bins=4)
    print(f"Codebook size: {fsq.codebook_size}")

    # Test with random input
    # Shape: (B=2, T=4, N=16, D=5) - batch, time, patches, latent_dim
    z = torch.randn(2, 4, 16, 5)
    print(f"Input shape: {z.shape}")

    # Quantize
    z_q, indices = fsq(z)
    print(f"Quantized shape: {z_q.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Index range: [{indices.min().item()}, {indices.max().item()}]")

    # Verify reconstruction from indices
    z_reconstructed = fsq.indices_to_latent(indices)
    print(f"Reconstructed shape: {z_reconstructed.shape}")

    # Check that reconstruction matches quantized output
    error = (z_q - z_reconstructed).abs().max().item()
    print(f"Reconstruction error: {error}")
    assert error < 1e-6, "Reconstruction should be exact!"

    print("All tests passed!")
