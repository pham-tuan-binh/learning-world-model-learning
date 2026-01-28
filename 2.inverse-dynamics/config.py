"""
Configuration for Latent Action Model (Inverse Dynamics) training.

All hyperparameters are centralized here for easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Key difference from video tokenizer:
    - n_actions: Size of discrete action vocabulary (power of 2)
    - action_dim: Computed as log2(n_actions)
    - use_adaptive_conditioning: Enable adaptive layer norm with action conditioning
    """

    # Input dimensions
    frame_size: int = 128         # H = W
    num_frames: int = 4           # T - temporal context

    # Architecture
    patch_size: int = 8           # P
    embed_dim: int = 128          # E
    num_heads: int = 8
    num_blocks: int = 4

    # Action space
    n_actions: int = 8            # Discrete action vocabulary (2^3 = 8)

    # Conditioning
    use_adaptive_conditioning: bool = True  # Enable adaptive layer norm conditioning

    # Regularization
    dropout: float = 0.0

    @property
    def grid_size(self) -> int:
        return self.frame_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size ** 2

    @property
    def action_dim(self) -> int:
        """Action dimension = log2(n_actions)"""
        import math
        return int(math.log2(self.n_actions))


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 1e-4       # Lower than video tokenizer
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Schedule
    num_epochs: int = 50
    warmup_steps: int = 500

    # Batching
    batch_size: int = 8

    # Regularization
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 100
    save_interval: int = 1000

    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset configuration."""

    data_type: str = "dummy"
    data_path: Optional[str] = None
    frame_skip: int = 1
    train_split: float = 0.9
    num_workers: int = 4
    pin_memory: bool = True
    dummy_num_samples: int = 1000


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = True

    def __post_init__(self):
        assert self.model.embed_dim % self.model.num_heads == 0
        assert self.model.frame_size % self.model.patch_size == 0
        # n_actions must be power of 2
        import math
        assert math.log2(self.model.n_actions).is_integer()


def get_config(**overrides) -> Config:
    """Create configuration with optional overrides."""
    config = Config()

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
    config = get_config()
    print("=== Latent Action Model Configuration ===\n")
    print(f"Frame: {config.model.frame_size}x{config.model.frame_size}, {config.model.num_frames} frames")
    print(f"Patches: {config.model.grid_size}x{config.model.grid_size} = {config.model.num_patches}")
    print(f"Actions: {config.model.n_actions} discrete (dim={config.model.action_dim})")
    print(f"Device: {config.device}")
