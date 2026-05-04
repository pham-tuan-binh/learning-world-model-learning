"""Dataset utilities for token dynamics training.

The dynamics model does not train on pixels directly. This loader turns each
video clip into:

    tokens:  (T, N)     visual token ids from the folder 1 video tokenizer
    actions: (T-1, A)   latent action vectors from the folder 2 inverse model

Those tensors are cached under ``data/.cache`` after the first preprocessing
pass so repeated training runs do not re-tokenize the videos.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from video_tokenizer import VideoFolderDataset, VideoTokenizer
from inverse_dynamics import LatentActionModel


DEFAULT_DATA_PATH = "3.dynamics/data"
DEFAULT_TOKENIZER_CHECKPOINT = "1.video-tokenizer/checkpoints/best_model.pt"
DEFAULT_INVERSE_DYNAMICS_CHECKPOINT = "2.inverse-dynamics/checkpoints/best_model.pt"


class DummyTokenDynamicsDataset(Dataset):
    """
    Random token/action dataset for wiring checks.

    This does not teach meaningful dynamics. It only verifies that the model,
    loss, and training loop agree on tensor shapes.

    Args:
        num_samples: Number of samples to generate
        num_frames: Frames per clip (T)
        num_patches: Patch tokens per frame (N)
        vocab_size: Visual token vocabulary size (V)
        action_dim: Latent action dimension (A)
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 4,
        num_patches: int = 256,
        vocab_size: int = 1024,
        action_dim: int = 3,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.vocab_size = vocab_size
        self.action_dim = action_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Return one dummy training sample.

        Returns:
            sample:
                tokens: (T, N) integer token ids
                actions: (T-1, A) quantized latent action vectors in {-1, 1}
        """
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.num_frames, self.num_patches),
            dtype=torch.long,
        )
        actions = torch.randint(
            low=0,
            high=2,
            size=(self.num_frames - 1, self.action_dim),
            dtype=torch.long,
        ).float()
        actions = actions * 2 - 1

        return {"tokens": tokens, "actions": actions}


class TokenDynamicsDataset(Dataset):
    """Video dataset for token dynamics training.

    Loads ``.mp4`` files from ``data_path``, samples short clips, then
    uses the frozen models from the previous folders as preprocessors:

    1. ``VideoTokenizer`` maps frames to discrete visual token ids.
    2. ``LatentActionModel`` maps adjacent frame pairs to quantized actions.

    Args:
        data_path: Directory containing ``.mp4`` files
        tokenizer_checkpoint: Trained checkpoint from ``1.video-tokenizer``
        inverse_dynamics_checkpoint: Trained checkpoint from
            ``2.inverse-dynamics``
        num_frames: Frames per clip (T)
        frame_size: Height/width used by the tokenizer and inverse model
        num_patches: Expected patches per frame (N)
        action_dim: Expected latent action dimension (A)
        frame_skip: Sample every N-th frame from each video
        cache_dir: Optional cache directory. Defaults to ``data_path/.cache``
        refresh_cache: Rebuild cached tensors even if a matching cache exists
        precompute_batch_size: Batch size used while running frozen models
        precompute_device: Device for preprocessing. Defaults to CUDA if present

    Returns:
        Samples shaped for ``TokenDynamicsModel.training_step``:
            ``{"tokens": Tensor[T, N], "actions": Tensor[T-1, A]}``
    """

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        tokenizer_checkpoint: str = DEFAULT_TOKENIZER_CHECKPOINT,
        inverse_dynamics_checkpoint: str = DEFAULT_INVERSE_DYNAMICS_CHECKPOINT,
        num_frames: int = 4,
        frame_size: int = 128,
        num_patches: int = 256,
        action_dim: int = 3,
        frame_skip: int = 1,
        cache_dir: Optional[str] = None,
        refresh_cache: bool = False,
        precompute_batch_size: int = 16,
        precompute_device: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer_checkpoint = Path(tokenizer_checkpoint)
        self.inverse_dynamics_checkpoint = Path(inverse_dynamics_checkpoint)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_patches = num_patches
        self.action_dim = action_dim
        self.frame_skip = frame_skip
        self.precompute_batch_size = precompute_batch_size
        self.precompute_device = precompute_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else self.data_path / ".cache"
        )
        self.cache_path = self.cache_dir / self._cache_filename()

        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}.")

        self.metadata = self._build_metadata()
        cache = None if refresh_cache else self._load_cache_if_valid()
        if cache is None:
            cache = self._precompute_and_cache()

        self.tokens = cache["tokens"].long()
        self.actions = cache["actions"].float()

        if self.tokens.dim() != 3:
            raise ValueError("Cached tokens must have shape (S, T, N)")
        if self.actions.dim() != 3:
            raise ValueError("Cached actions must have shape (S, T-1, A)")
        if self.tokens.shape[1:] != (self.num_frames, self.num_patches):
            raise ValueError(
                "Cached token shape does not match the requested dataset shape: "
                f"expected (*, {self.num_frames}, {self.num_patches}), "
                f"got {tuple(self.tokens.shape)}"
            )
        if self.actions.shape[1:] != (self.num_frames - 1, self.action_dim):
            raise ValueError(
                "Cached action shape does not match the requested dataset shape: "
                f"expected (*, {self.num_frames - 1}, {self.action_dim}), "
                f"got {tuple(self.actions.shape)}"
            )

        print(
            "Loaded token dynamics dataset: "
            f"{len(self.tokens)} clips, tokens {tuple(self.tokens.shape[1:])}, "
            f"actions {tuple(self.actions.shape[1:])}"
        )

    def __len__(self) -> int:
        """Return the number of cached token/action clips."""
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Return one token/action training sample.

        Args:
            idx: Sample index

        Returns:
            sample:
                tokens: (T, N) integer visual token ids
                actions: (T-1, A) quantized latent action vectors
        """
        return {
            "tokens": self.tokens[idx],
            "actions": self.actions[idx],
        }

    def _cache_filename(self) -> str:
        """Create a cache filename tied to the clip geometry."""
        return (
            "token_dynamics_"
            f"T{self.num_frames}_H{self.frame_size}_N{self.num_patches}_"
            f"A{self.action_dim}_skip{self.frame_skip}.pt"
        )

    def _build_metadata(self) -> dict:
        """Build metadata used to decide whether a cache is still valid."""
        video_paths = sorted(self.data_path.glob("**/*.mp4"))
        if len(video_paths) == 0:
            raise ValueError(f"No .mp4 files found in {self.data_path}.")

        return {
            "num_frames": self.num_frames,
            "frame_size": self.frame_size,
            "num_patches": self.num_patches,
            "action_dim": self.action_dim,
            "frame_skip": self.frame_skip,
            "video_count": len(video_paths),
            "video_latest_mtime": max(path.stat().st_mtime for path in video_paths),
            "tokenizer_checkpoint": str(self.tokenizer_checkpoint),
            "tokenizer_checkpoint_mtime": self._path_mtime(self.tokenizer_checkpoint),
            "inverse_dynamics_checkpoint": str(self.inverse_dynamics_checkpoint),
            "inverse_dynamics_checkpoint_mtime": self._path_mtime(
                self.inverse_dynamics_checkpoint
            ),
        }

    def _load_cache_if_valid(self) -> Optional[dict]:
        """Load cached tensors when the stored metadata matches this dataset."""
        if not self.cache_path.exists():
            return None

        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != self.metadata:
            print(f"Ignoring stale token/action cache: {self.cache_path}")
            return None

        print(f"Using cached token/action tensors: {self.cache_path}")
        return payload

    def _precompute_and_cache(self) -> dict:
        """Run frozen tokenizer and inverse models over the video clips."""
        print("Building token/action cache from videos...")
        print(f"  Data path: {self.data_path}")
        print(f"  Cache path: {self.cache_path}")
        print(f"  Precompute device: {self.precompute_device}")

        tokenizer = self._load_video_tokenizer()
        inverse_model = self._load_inverse_dynamics_model()

        video_dataset = VideoFolderDataset(
            root_dir=str(self.data_path),
            num_frames=self.num_frames,
            frame_size=self.frame_size,
            frame_skip=self.frame_skip,
        )
        video_loader = DataLoader(
            video_dataset,
            batch_size=self.precompute_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.precompute_device.startswith("cuda"),
        )

        token_batches = []
        action_batches = []

        with torch.no_grad():
            for batch_idx, frames in enumerate(video_loader):
                # frames: (B, T, C, H, W) - raw Doom clips in [0, 1]
                frames = frames.to(self.precompute_device)

                # tokens: (B, T, N) - visual token ids per frame
                _, tokens, _ = tokenizer.encoder(frames)

                # actions: (B, T-1, A) - quantized latent action vectors
                actions = inverse_model.encode(frames)

                token_batches.append(tokens.cpu())
                action_batches.append(actions.cpu())

                if (batch_idx + 1) % 25 == 0:
                    processed = min(
                        (batch_idx + 1) * self.precompute_batch_size,
                        len(video_dataset),
                    )
                    print(f"  Preprocessed {processed}/{len(video_dataset)} clips")

        tokens = torch.cat(token_batches, dim=0).long()
        actions = torch.cat(action_batches, dim=0).float()

        payload = {
            "metadata": self.metadata,
            "tokens": tokens,
            "actions": actions,
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.cache_path)
        print(f"Saved token/action cache to {self.cache_path}")

        return payload

    def _load_video_tokenizer(self) -> VideoTokenizer:
        """Load the trained folder 1 video tokenizer for frame tokenization."""
        checkpoint = self._load_checkpoint(self.tokenizer_checkpoint)
        checkpoint_config = checkpoint.get("config")
        model_config = checkpoint_config.model if checkpoint_config else None

        model = VideoTokenizer(
            in_channels=getattr(model_config, "in_channels", 3),
            frame_size=getattr(model_config, "frame_size", self.frame_size),
            num_frames=getattr(model_config, "num_frames", self.num_frames),
            patch_size=getattr(model_config, "patch_size", 8),
            embed_dim=getattr(model_config, "embed_dim", 128),
            num_heads=getattr(model_config, "num_heads", 8),
            num_blocks=getattr(model_config, "num_blocks", 4),
            latent_dim=getattr(model_config, "latent_dim", 5),
            num_bins=getattr(model_config, "num_bins", 4),
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.precompute_device)
        model.eval()
        return model

    def _load_inverse_dynamics_model(self) -> LatentActionModel:
        """Load the trained folder 2 inverse dynamics model for action inference."""
        checkpoint = self._load_checkpoint(self.inverse_dynamics_checkpoint)
        checkpoint_config = checkpoint.get("config")
        model_config = checkpoint_config.model if checkpoint_config else None

        model = LatentActionModel(
            frame_size=getattr(model_config, "frame_size", self.frame_size),
            n_actions=getattr(model_config, "n_actions", 8),
            patch_size=getattr(model_config, "patch_size", 8),
            embed_dim=getattr(model_config, "embed_dim", 128),
            num_heads=getattr(model_config, "num_heads", 8),
            num_blocks=getattr(model_config, "num_blocks", 4),
            use_adaptive_conditioning=getattr(
                model_config,
                "use_adaptive_conditioning",
                True,
            ),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.precompute_device)
        model.eval()
        return model

    @staticmethod
    def _path_mtime(path: Path) -> float:
        """Return file mtime and raise a helpful error if the file is missing."""
        if not path.exists():
            raise ValueError(f"Checkpoint not found: {path}")
        return path.stat().st_mtime

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        """Load a PyTorch checkpoint, with a clear error for Git LFS pointers."""
        if not path.exists():
            raise ValueError(f"Checkpoint not found: {path}")

        with path.open("rb") as handle:
            prefix = handle.read(64)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{path} is a Git LFS pointer, not the checkpoint bytes. "
                "Run `git lfs pull` before building the dynamics dataset."
            )

        return torch.load(path, map_location="cpu", weights_only=False)


def split_dataset(
    dataset: Dataset,
    train_split: float = 0.9,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and validation sets."""
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def get_dataloader(
    data_type: str = "dummy",
    data_path: Optional[str] = None,
    tokenizer_checkpoint: str = DEFAULT_TOKENIZER_CHECKPOINT,
    inverse_dynamics_checkpoint: str = DEFAULT_INVERSE_DYNAMICS_CHECKPOINT,
    batch_size: int = 8,
    num_frames: int = 4,
    frame_size: int = 128,
    num_patches: int = 256,
    vocab_size: int = 1024,
    action_dim: int = 3,
    frame_skip: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    dummy_num_samples: int = 1000,
    refresh_cache: bool = False,
    precompute_batch_size: int = 16,
    precompute_device: Optional[str] = None,
) -> DataLoader:
    """
    Create a dataloader for dynamics training.

    Args:
        data_type: "dummy" for random tensors, "folder" for real videos.
        data_path: Path to video directory
        tokenizer_checkpoint: Checkpoint used to tokenize frames
        inverse_dynamics_checkpoint: Checkpoint used to infer latent actions
        batch_size: Batch size
        num_frames: Frames per clip (T)
        frame_size: Height/width of frames (H=W)
        num_patches: Patch tokens per frame (N)
        vocab_size: Visual token vocabulary size (V)
        action_dim: Latent action dimension (A)
        frame_skip: Sample every N-th frame from source videos
        num_workers: DataLoader worker count
        pin_memory: Whether to pin CPU memory
        dummy_num_samples: Number of dummy samples
        refresh_cache: Rebuild the token/action cache
        precompute_batch_size: Batch size for frozen tokenizer/action models
        precompute_device: Device for preprocessing

    Returns:
        dataloader: PyTorch DataLoader yielding {"tokens", "actions"}
    """
    if data_type == "dummy":
        dataset = DummyTokenDynamicsDataset(
            num_samples=dummy_num_samples,
            num_frames=num_frames,
            num_patches=num_patches,
            vocab_size=vocab_size,
            action_dim=action_dim,
        )
    elif data_type == "folder":
        dataset = TokenDynamicsDataset(
            data_path=data_path or DEFAULT_DATA_PATH,
            tokenizer_checkpoint=tokenizer_checkpoint,
            inverse_dynamics_checkpoint=inverse_dynamics_checkpoint,
            num_frames=num_frames,
            frame_size=frame_size,
            num_patches=num_patches,
            action_dim=action_dim,
            frame_skip=frame_skip,
            refresh_cache=refresh_cache,
            precompute_batch_size=precompute_batch_size,
            precompute_device=precompute_device,
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
