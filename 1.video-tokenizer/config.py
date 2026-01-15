"""
Configuration for Video Tokenizer training and evaluation.

All hyperparameters are centralized here for easy experimentation.
Use dataclasses for type hints and default values.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Dimensions explanation:
    - frame_size (H=W): Height and width of input frames in pixels
    - num_frames (T): Number of consecutive frames in temporal context
    - patch_size (P): Size of each square patch (P×P pixels)
    - embed_dim (E): Dimension of token embeddings after patch projection
    - num_heads (H): Number of parallel attention heads
    - num_blocks: Number of stacked transformer blocks
    - latent_dim (D): Dimension of quantized latent space
    - num_bins (L): Quantization levels per latent dimension

    Derived dimensions:
    - grid_size = frame_size / patch_size = number of patches per row/column
    - num_patches = grid_size^2 = total patches per frame
    - codebook_size = num_bins^latent_dim = total unique tokens
    """

    # Input dimensions
    in_channels: int = 3          # C - RGB channels
    frame_size: int = 128         # H = W - frame resolution
    num_frames: int = 4           # T - temporal context length

    # Architecture
    patch_size: int = 8           # P - patch dimensions (P×P pixels)
    embed_dim: int = 128          # E - embedding dimension
    num_heads: int = 8            # Number of attention heads (E must be divisible by this)
    num_blocks: int = 4           # Number of transformer blocks

    # Quantization (FSQ)
    latent_dim: int = 5           # D - dimensions in latent space
    num_bins: int = 4             # L - quantization levels per dimension
    # codebook_size = L^D = 4^5 = 1024 tokens

    # Regularization
    dropout: float = 0.0          # Dropout probability (0 = no dropout)

    @property
    def grid_size(self) -> int:
        """Number of patches per row/column: H/P"""
        return self.frame_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total patches per frame: (H/P)^2"""
        return self.grid_size ** 2

    @property
    def codebook_size(self) -> int:
        """Total unique tokens: L^D"""
        return self.num_bins ** self.latent_dim


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.

    Learning rate schedule:
    - Warmup: Linear increase from 0 to lr over warmup_steps
    - Decay: Cosine annealing from lr to min_lr over remaining steps
    """

    # Optimization
    learning_rate: float = 1e-3       # Peak learning rate
    min_lr: float = 1e-5              # Minimum learning rate (end of cosine decay)
    weight_decay: float = 0.01        # Weight decay for AdamW
    betas: tuple = (0.9, 0.999)       # Adam betas

    # Schedule
    num_epochs: int = 100             # Total training epochs
    warmup_steps: int = 500           # Linear warmup steps

    # Batching
    batch_size: int = 8               # Batch size per device
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps

    # Regularization
    max_grad_norm: float = 1.0        # Gradient clipping threshold

    # Logging and checkpointing
    log_interval: int = 100           # Log every N steps
    save_interval: int = 1000         # Save checkpoint every N steps
    eval_interval: int = 500          # Evaluate every N steps

    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """
    Dataset configuration.

    Supports multiple data sources:
    - dummy: Random tensors (for testing/debugging)
    - folder: Load videos from a directory
    - images: Treat images as single frames
    """

    # Data source
    data_type: str = "dummy"          # "dummy", "folder", or "images"
    data_path: Optional[str] = None   # Path to data directory

    # Video sampling
    frame_skip: int = 1               # Sample every N-th frame (1 = consecutive)
    train_split: float = 0.9          # Fraction of data for training

    # Preprocessing
    normalize: bool = True            # Normalize to [0, 1]

    # Data loading
    num_workers: int = 4              # DataLoader workers
    pin_memory: bool = True           # Pin memory for faster GPU transfer

    # Dummy data settings (only used when data_type="dummy")
    dummy_num_samples: int = 1000     # Number of dummy samples


@dataclass
class Config:
    """
    Complete configuration combining model, training, and data settings.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Mixed precision
    use_amp: bool = True              # Use automatic mixed precision

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check embed_dim is divisible by num_heads
        assert self.model.embed_dim % self.model.num_heads == 0, (
            f"embed_dim ({self.model.embed_dim}) must be divisible by "
            f"num_heads ({self.model.num_heads})"
        )

        # Check frame_size is divisible by patch_size
        assert self.model.frame_size % self.model.patch_size == 0, (
            f"frame_size ({self.model.frame_size}) must be divisible by "
            f"patch_size ({self.model.patch_size})"
        )


def get_config(**overrides) -> Config:
    """
    Create configuration with optional overrides.

    Args:
        **overrides: Keyword arguments to override defaults.
                     Use dot notation for nested configs:
                     get_config(model_embed_dim=256, training_lr=1e-4)

    Returns:
        Config object with overrides applied.

    Example:
        config = get_config(
            model_frame_size=64,
            training_batch_size=16,
            data_data_type="folder",
        )
    """
    config = Config()

    # Apply overrides
    for key, value in overrides.items():
        parts = key.split("_", 1)
        if len(parts) == 2 and hasattr(config, parts[0]):
            sub_config = getattr(config, parts[0])
            if hasattr(sub_config, parts[1]):
                setattr(sub_config, parts[1], value)
            else:
                raise ValueError(f"Unknown config key: {key}")
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")

    return config


if __name__ == "__main__":
    # Print default configuration
    config = get_config()

    print("=== Video Tokenizer Configuration ===\n")

    print("Model Configuration:")
    print(f"  Input: {config.model.in_channels} channels, "
          f"{config.model.frame_size}x{config.model.frame_size} frames, "
          f"{config.model.num_frames} frames")
    print(f"  Patches: {config.model.patch_size}x{config.model.patch_size} -> "
          f"{config.model.grid_size}x{config.model.grid_size} = "
          f"{config.model.num_patches} patches/frame")
    print(f"  Embedding: {config.model.embed_dim} dim, "
          f"{config.model.num_heads} heads, "
          f"{config.model.num_blocks} blocks")
    print(f"  Quantization: {config.model.latent_dim} dims x "
          f"{config.model.num_bins} bins = "
          f"{config.model.codebook_size} tokens")

    print("\nTraining Configuration:")
    print(f"  LR: {config.training.learning_rate} -> {config.training.min_lr}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Warmup steps: {config.training.warmup_steps}")

    print("\nData Configuration:")
    print(f"  Type: {config.data.data_type}")
    print(f"  Path: {config.data.data_path}")

    print(f"\nDevice: {config.device}")
    print(f"AMP: {config.use_amp}")
