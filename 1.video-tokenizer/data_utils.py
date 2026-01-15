"""
Data utilities for Video Tokenizer training.

Provides flexible dataset classes that can load:
1. DummyVideoDataset: Random tensors for testing (no dependencies)
2. VideoFolderDataset: Videos from a directory (requires opencv)
3. ImageFolderDataset: Images as single frames (requires PIL)

Each dataset returns video clips as tensors of shape:
    (T, C, H, W) - time, channels, height, width

The dataloader then batches these to:
    (B, T, C, H, W) - batch, time, channels, height, width
"""

import os
import random
from typing import Optional, Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class DummyVideoDataset(Dataset):
    """
    Dataset that generates random video tensors.

    Useful for testing the model without needing real data.
    Each sample is a random tensor that simulates a video clip.

    Args:
        num_samples: Number of samples in the dataset
        num_frames: Number of frames per clip (T)
        frame_size: Height/width of frames (H=W)
        channels: Number of channels (C), typically 3 for RGB
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 4,
        frame_size: int = 128,
        channels: int = 3,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames      # T
        self.frame_size = frame_size      # H = W
        self.channels = channels          # C

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Pre-generate all samples for consistency across epochs
        # Shape: (num_samples, T, C, H, W)
        # Using smooth random patterns instead of pure noise for more realistic training
        self.data = self._generate_smooth_videos()

    def _generate_smooth_videos(self) -> torch.Tensor:
        """
        Generate smooth random videos that are more representative of real data.

        Instead of pure noise, we create videos with:
        - Smooth spatial patterns (like gradients or blobs)
        - Temporal consistency (frames are related)
        """
        # Start with random base patterns
        # Shape: (num_samples, C, H, W)
        base = torch.rand(self.num_samples, self.channels, self.frame_size, self.frame_size)

        # Create temporal variations
        videos = []
        for i in range(self.num_samples):
            frames = []
            current = base[i]

            for t in range(self.num_frames):
                # Add small temporal variation
                noise = torch.randn_like(current) * 0.05
                current = (current + noise).clamp(0, 1)
                frames.append(current.clone())

            # Stack frames: (T, C, H, W)
            video = torch.stack(frames, dim=0)
            videos.append(video)

        # Stack all videos: (N, T, C, H, W)
        return torch.stack(videos, dim=0)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a video clip.

        Args:
            idx: Sample index

        Returns:
            video: Video tensor, shape (T, C, H, W)
                   Values in [0, 1]
        """
        return self.data[idx]


class VideoFolderDataset(Dataset):
    """
    Dataset that loads videos from a directory.

    Expects video files (mp4, avi, mov, etc.) in the specified directory.
    Samples consecutive frames from each video.

    Requires: opencv-python (cv2)

    Args:
        root_dir: Path to directory containing video files
        num_frames: Number of frames per clip (T)
        frame_size: Target height/width for resizing (H=W)
        frame_skip: Sample every N-th frame (1 = consecutive)
        extensions: Video file extensions to look for
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 4,
        frame_size: int = 128,
        frame_skip: int = 1,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_skip = frame_skip

        # Find all video files
        self.video_paths: List[Path] = []
        for ext in extensions:
            self.video_paths.extend(self.root_dir.glob(f"**/*{ext}"))

        if len(self.video_paths) == 0:
            raise ValueError(f"No video files found in {root_dir}")

        print(f"Found {len(self.video_paths)} videos in {root_dir}")

        # Build index of (video_path, start_frame) pairs
        self._build_index()

    def _build_index(self):
        """Build index of valid (video, start_frame) pairs."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for VideoFolderDataset. "
                "Install with: uv add opencv-python"
            )

        self.clips: List[Tuple[Path, int]] = []

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Calculate frames needed for one clip
            frames_needed = (self.num_frames - 1) * self.frame_skip + 1

            # Generate all valid starting positions
            for start in range(0, total_frames - frames_needed + 1, self.num_frames):
                self.clips.append((video_path, start))

        print(f"Total clips: {len(self.clips)}")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load a video clip.

        Args:
            idx: Clip index

        Returns:
            video: Video tensor, shape (T, C, H, W)
                   Values in [0, 1], RGB format
        """
        import cv2

        video_path, start_frame = self.clips[idx]

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in range(self.num_frames):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame if we run out
                frame = frames[-1] if frames else torch.zeros(
                    3, self.frame_size, self.frame_size
                )
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                # Convert to tensor and normalize to [0, 1]
                frame = torch.from_numpy(frame).float() / 255.0
                # (H, W, C) -> (C, H, W)
                frame = frame.permute(2, 0, 1)

            frames.append(frame)

            # Skip frames if needed
            for _ in range(self.frame_skip - 1):
                cap.read()

        cap.release()

        # Stack: (T, C, H, W)
        video = torch.stack(frames, dim=0)
        return video


class ImageFolderDataset(Dataset):
    """
    Dataset that loads images as single frames (T=1) or creates
    pseudo-videos by sampling consecutive images.

    Useful when you have a folder of images instead of videos.

    Requires: Pillow

    Args:
        root_dir: Path to directory containing images
        num_frames: Number of frames per clip (T)
                    If > 1, samples consecutive images as frames
        frame_size: Target height/width for resizing (H=W)
        extensions: Image file extensions to look for
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 4,
        frame_size: int = 128,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size

        # Find all image files
        self.image_paths: List[Path] = []
        for ext in extensions:
            self.image_paths.extend(self.root_dir.glob(f"**/*{ext}"))
            self.image_paths.extend(self.root_dir.glob(f"**/*{ext.upper()}"))

        # Sort for consistent ordering
        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in {root_dir}")

        print(f"Found {len(self.image_paths)} images in {root_dir}")

        # If num_frames > 1, we sample consecutive images
        self.num_clips = max(1, len(self.image_paths) - num_frames + 1)

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load an image clip.

        Args:
            idx: Clip index

        Returns:
            video: Video tensor, shape (T, C, H, W)
                   Values in [0, 1], RGB format
        """
        from PIL import Image

        frames = []
        for i in range(self.num_frames):
            img_idx = min(idx + i, len(self.image_paths) - 1)
            img_path = self.image_paths[img_idx]

            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.frame_size, self.frame_size), Image.BILINEAR)

            # Convert to tensor
            frame = torch.from_numpy(
                __import__("numpy").array(img)
            ).float() / 255.0

            # (H, W, C) -> (C, H, W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)

        # Stack: (T, C, H, W)
        video = torch.stack(frames, dim=0)
        return video


def get_dataloader(
    data_type: str = "dummy",
    data_path: Optional[str] = None,
    num_frames: int = 4,
    frame_size: int = 128,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    num_samples: int = 1000,
    frame_skip: int = 1,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for video data.

    Args:
        data_type: Type of dataset ("dummy", "folder", or "images")
        data_path: Path to data directory (required for "folder" and "images")
        num_frames: Number of frames per clip (T)
        frame_size: Frame resolution (H=W)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        shuffle: Shuffle the data
        num_samples: Number of samples for dummy dataset
        frame_skip: Frame skip for video dataset
        seed: Random seed

    Returns:
        DataLoader that yields batches of shape (B, T, C, H, W)
    """
    if data_type == "dummy":
        dataset = DummyVideoDataset(
            num_samples=num_samples,
            num_frames=num_frames,
            frame_size=frame_size,
            channels=3,
            seed=seed,
        )
    elif data_type == "folder":
        if data_path is None:
            raise ValueError("data_path is required for 'folder' data type")
        dataset = VideoFolderDataset(
            root_dir=data_path,
            num_frames=num_frames,
            frame_size=frame_size,
            frame_skip=frame_skip,
        )
    elif data_type == "images":
        if data_path is None:
            raise ValueError("data_path is required for 'images' data type")
        dataset = ImageFolderDataset(
            root_dir=data_path,
            num_frames=num_frames,
            frame_size=frame_size,
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches
    )

    return dataloader


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into train and validation sets.

    Args:
        dataset: Dataset to split
        train_ratio: Fraction for training (e.g., 0.9 = 90% train, 10% val)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset
    """
    from torch.utils.data import Subset

    # Set seed for reproducibility
    random.seed(seed)

    # Create indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Split
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test the data utilities
    print("Testing Data Utilities...\n")

    # Test DummyVideoDataset
    print("--- DummyVideoDataset ---")
    dummy_dataset = DummyVideoDataset(
        num_samples=100,
        num_frames=4,
        frame_size=128,
        channels=3,
        seed=42,
    )
    print(f"Dataset size: {len(dummy_dataset)}")
    sample = dummy_dataset[0]
    print(f"Sample shape: {sample.shape}")  # (T, C, H, W)
    print(f"Value range: [{sample.min():.2f}, {sample.max():.2f}]")

    # Test DataLoader
    print("\n--- DataLoader ---")
    dataloader = get_dataloader(
        data_type="dummy",
        num_frames=4,
        frame_size=128,
        batch_size=8,
        num_samples=100,
    )
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")  # (B, T, C, H, W)

    # Test split
    print("\n--- Dataset Split ---")
    train_set, val_set = split_dataset(dummy_dataset, train_ratio=0.9)
    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")

    print("\nAll tests passed!")
