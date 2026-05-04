"""
Configuration for Token Dynamics Model training.

The dynamics model is the final piece of the small Genie-style pipeline:
video tokens + latent actions -> next video tokens.
"""

from dataclasses import dataclass, field
from typing import Optional
import math
import torch


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Dimensions:
    - T: number of frames in a training clip
    - N: number of visual tokens per frame
    - V: tokenizer vocabulary size
    - A: latent action dimension
    - E: transformer embedding dimension
    """

    # Tokenized frame shape
    frame_size: int = 128
    num_frames: int = 4
    patch_size: int = 8

    # Tokenizer vocabulary
    latent_dim: int = 3
    num_bins: int = 8

    # Latent action space
    n_actions: int = 4

    # Transformer
    embed_dim: int = 128
    num_heads: int = 8
    num_blocks: int = 4
    dropout: float = 0.0
    use_adaptive_conditioning: bool = False

    @property
    def grid_size(self) -> int:
        """Number of patches per row/column: H/P."""
        return self.frame_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total patches per frame: (H/P)^2."""
        return self.grid_size ** 2

    @property
    def vocab_size(self) -> int:
        """Tokenizer vocabulary size: num_bins^latent_dim."""
        return self.num_bins ** self.latent_dim

    @property
    def action_dim(self) -> int:
        """Binary latent action dimension: log2(n_actions)."""
        return int(math.log2(self.n_actions))


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    num_epochs: int = 50
    warmup_steps: int = 500
    batch_size: int = 8
    max_grad_norm: float = 1.0

    log_interval: int = 100
    save_interval: int = 1000
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset configuration."""

    data_type: str = "folder"
    data_path: Optional[str] = "3.dynamics/data"
    tokenizer_checkpoint: str = "1.video-tokenizer/checkpoints/best_model.pt"
    inverse_dynamics_checkpoint: str = "2.inverse-dynamics/checkpoints/best_model.pt"
    frame_skip: int = 1
    train_split: float = 0.9
    num_workers: int = 4
    pin_memory: bool = True
    dummy_num_samples: int = 1000
    refresh_cache: bool = False
    precompute_batch_size: int = 16
    precompute_device: Optional[str] = None


@dataclass
class Config:
    """Complete dynamics configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "3.dynamics/checkpoints"
    use_amp: bool = True

    def __post_init__(self):
        assert self.model.embed_dim % self.model.num_heads == 0
        assert self.model.frame_size % self.model.patch_size == 0
        assert math.log2(self.model.n_actions).is_integer()


def get_config(**overrides) -> Config:
    """Create configuration with optional flattened overrides."""
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

    config.__post_init__()
    return config


if __name__ == "__main__":
    config = get_config()
    print("=== Token Dynamics Configuration ===\n")
    print(f"Frames per clip: {config.model.num_frames}")
    print(f"Patches per frame: {config.model.num_patches}")
    print(f"Vocabulary size: {config.model.vocab_size}")
    print(f"Action dim: {config.model.action_dim}")
    print(f"Device: {config.device}")
