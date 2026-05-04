"""
Infer actions from a video using the trained inverse dynamics model.

Given a video, outputs the chain of predicted actions for each frame transition.
This is the core of inverse dynamics: frames -> actions.

Usage (from repo root):
    uv run ./2.inverse-dynamics/debug/infer_actions.py \
        --checkpoint ./2.inverse-dynamics/checkpoints/best_model.pt \
        --video ./2.inverse-dynamics/data/sample.mp4
"""

import argparse
import json
from pathlib import Path

import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from inverse_dynamics import LatentActionModel


# Action index to vector mapping (0-3 -> [-1/+1, -1/+1])
ACTION_VECTORS = {
    0: [-1, -1],
    1: [+1, -1],
    2: [-1, +1],
    3: [+1, +1],
}


def load_model(checkpoint_path: str, device: str = None):
    """Load a trained LatentActionModel from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = LatentActionModel(
        frame_size=config.model.frame_size,
        n_actions=config.model.n_actions,
        patch_size=config.model.patch_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_blocks=config.model.num_blocks,
        use_adaptive_conditioning=config.model.use_adaptive_conditioning,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Action vocabulary: {config.model.n_actions}")

    return model, config


def load_video_frames(video_path: str, frame_size: int, max_frames: int = None, frame_skip: int = 1):
    """Load frames from a video file."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV required. Install with: uv add opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")

    return torch.stack(frames, dim=0)


def action_vector_to_index(action_vector) -> int:
    """Convert action vector to index (0-3)."""
    bits = [(1 if a > 0 else 0) for a in action_vector]
    return bits[0] * 1 + bits[1] * 2


def infer_actions(model, frames: torch.Tensor, device: str, chunk_size: int = 16) -> list:
    """Infer actions for all frame transitions in a video."""
    T = frames.shape[0]
    all_actions = []

    with torch.no_grad():
        start = 0
        while start < T - 1:
            end = min(start + chunk_size, T)
            chunk = frames[start:end]
            chunk_batch = chunk.unsqueeze(0).to(device)

            actions = model.encode(chunk_batch)
            actions = actions[0].cpu()

            for i in range(actions.shape[0]):
                frame_from = start + i
                frame_to = start + i + 1
                action_vec = actions[i].tolist()
                action_idx = action_vector_to_index(action_vec)

                all_actions.append({
                    "transition": f"{frame_from} -> {frame_to}",
                    "frame_from": frame_from,
                    "frame_to": frame_to,
                    "action_index": action_idx,
                    "action_vector": action_vec,
                    "action_quantized": ACTION_VECTORS[action_idx],
                })

            start = end - 1

    return all_actions


def print_action_summary(actions: list):
    """Print a summary of the inferred actions."""
    print(f"\n{'='*60}")
    print("ACTION SEQUENCE")
    print(f"{'='*60}")

    action_counts = {}
    for a in actions:
        idx = a["action_index"]
        action_counts[idx] = action_counts.get(idx, 0) + 1

    print(f"\nTotal transitions: {len(actions)}")
    print(f"\nFrame transitions and predicted actions:")
    print("-" * 60)

    for a in actions:
        vec_str = "[" + ", ".join(f"{v:+.2f}" for v in a["action_vector"]) + "]"
        quant_str = str(a["action_quantized"])
        print(f"  {a['transition']:>12}  ->  Action {a['action_index'] + 1}  {quant_str:>15}  (raw: {vec_str})")

    print(f"\n{'='*60}")
    print("ACTION DISTRIBUTION")
    print(f"{'='*60}")

    for idx in sorted(action_counts.keys()):
        count = action_counts[idx]
        pct = 100 * count / len(actions)
        bar = "#" * int(pct / 2)
        print(f"  Action {idx + 1} {str(ACTION_VECTORS[idx]):>15}: {count:4d} ({pct:5.1f}%) {bar}")

    unique_actions = len(action_counts)
    print(f"\nUnique actions used: {unique_actions} / 4")

    print(f"\n{'='*60}")
    print("COMPACT SEQUENCE")
    print(f"{'='*60}")
    seq = "".join(str(a["action_index"] + 1) for a in actions)
    for i in range(0, len(seq), 50):
        print(f"  {seq[i:i+50]}")


def main():
    parser = argparse.ArgumentParser(description="Infer actions from video")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model(args.checkpoint, device)

    print(f"\nLoading video: {args.video}")
    frames = load_video_frames(
        args.video,
        frame_size=config.model.frame_size,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )
    print(f"Loaded {frames.shape[0]} frames")

    print("\nInferring actions...")
    actions = infer_actions(model, frames, device)

    if not args.quiet:
        print_action_summary(actions)
    else:
        seq = "".join(str(a["action_index"] + 1) for a in actions)
        print(f"\nAction sequence ({len(actions)} transitions):")
        print(seq)

    if args.output:
        output_data = {
            "video": args.video,
            "num_frames": frames.shape[0],
            "num_transitions": len(actions),
            "actions": actions,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved actions to: {args.output}")


if __name__ == "__main__":
    main()
