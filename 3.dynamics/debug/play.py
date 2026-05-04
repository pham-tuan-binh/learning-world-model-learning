"""
Interactive Dynamics Model Player.

Loads a starting frame from a Doom video, tokenizes it with the video tokenizer,
then lets you step forward in token space using 1-4 to pick actions. Each step
runs the dynamics model to predict the next token frame, then decodes back to
pixels for display.

Usage (from repo root):
    uv run ./3.dynamics/debug/play.py \
        --tokenizer-checkpoint ./1.video-tokenizer/checkpoints/best_model.pt \
        --dynamics-checkpoint  ./3.dynamics/checkpoints/best_model.pt \
        --start-video          ./assets/doom-samples/doom_0000.mp4

Controls:
    1-4: Apply action and step forward
    R: Reset to starting frame
    S: Save current displayed frame
    Q/ESC: Quit
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from video_tokenizer import VideoTokenizer
from dynamics import TokenDynamicsModel


ACTION_VECTORS = {
    0: [-1, -1],
    1: [+1, -1],
    2: [-1, +1],
    3: [+1, +1],
}

ACTION_NAMES = {
    0: "[-1,-1]",
    1: "[+1,-1]",
    2: "[-1,+1]",
    3: "[+1,+1]",
}


def load_tokenizer(checkpoint_path: str, device: str) -> tuple:
    """Load trained VideoTokenizer from checkpoint."""
    print(f"Loading tokenizer from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    mc = cfg.model if cfg else None

    model = VideoTokenizer(
        in_channels=getattr(mc, "in_channels", 3),
        frame_size=getattr(mc, "frame_size", 128),
        num_frames=getattr(mc, "num_frames", 4),
        patch_size=getattr(mc, "patch_size", 8),
        embed_dim=getattr(mc, "embed_dim", 128),
        num_heads=getattr(mc, "num_heads", 8),
        num_blocks=getattr(mc, "num_blocks", 4),
        latent_dim=getattr(mc, "latent_dim", 3),
        num_bins=getattr(mc, "num_bins", 8),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Tokenizer: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Frame size: {model.encoder.frame_size}, vocab: {model.codebook_size}")
    return model, cfg


def load_dynamics(checkpoint_path: str, device: str) -> tuple:
    """Load trained TokenDynamicsModel from checkpoint."""
    print(f"Loading dynamics model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    mc = cfg.model if cfg else None

    model = TokenDynamicsModel(
        vocab_size=getattr(mc, "vocab_size", 512),
        num_patches=getattr(mc, "num_patches", 256),
        action_dim=getattr(mc, "action_dim", 2),
        n_actions=getattr(mc, "n_actions", 4),
        embed_dim=getattr(mc, "embed_dim", 128),
        num_heads=getattr(mc, "num_heads", 8),
        num_blocks=getattr(mc, "num_blocks", 4),
        grid_size=getattr(mc, "grid_size", 16),
        dropout=0.0,
        use_adaptive_conditioning=getattr(mc, "use_adaptive_conditioning", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Dynamics: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, cfg


def load_start_frame(path: str, frame_size: int) -> torch.Tensor:
    """Read the first frame from an image or video file, return (C, H, W) in [0,1]."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV required: uv add opencv-python")

    path = Path(path)
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".webm"}:
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video: {path}")
    else:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (frame_size, frame_size))
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def encode_frame(tokenizer: VideoTokenizer, frame: torch.Tensor, device: str) -> torch.Tensor:
    """Encode a single (C, H, W) frame to token indices (N,)."""
    # tokenizer expects (B, T, C, H, W)
    x = frame.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        indices, _ = tokenizer.encode(x)  # (1, 1, N)
    return indices[0, 0]  # (N,)


def decode_tokens(tokenizer: VideoTokenizer, tokens: torch.Tensor, device: str) -> torch.Tensor:
    """Decode token indices (N,) to a (C, H, W) frame in [0,1]."""
    # decode_indices expects (B, T, N)
    idx = tokens.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        x_hat = tokenizer.decode_indices(idx)  # (1, 1, C, H, W)
    return x_hat[0, 0].clamp(0, 1).cpu()


def predict_next_tokens(
    dynamics: TokenDynamicsModel,
    token_history: list,
    action_history: list,
    device: str,
    temperature: float,
    sample: bool,
) -> torch.Tensor:
    """
    Run one dynamics step given accumulated token and action histories.

    token_history:  list of (N,) tensors, length >= 1
    action_history: list of (A,) tensors, length == len(token_history)
    Returns next token ids (N,).
    """
    ctx_tokens = torch.stack(token_history, dim=0).unsqueeze(0).to(device)     # (1, T, N)
    ctx_actions = torch.stack(action_history, dim=0).unsqueeze(0).to(device)   # (1, T, A)

    with torch.no_grad():
        next_tokens = dynamics.predict_next(
            context_tokens=ctx_tokens,
            context_actions=ctx_actions,
            temperature=temperature,
            sample=sample,
        )  # (1, N)

    return next_tokens[0].cpu()


def tensor_to_numpy(frame: torch.Tensor, display_size: int) -> np.ndarray:
    """(C, H, W) float tensor -> (H, W, C) uint8 numpy, resized for display."""
    arr = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    if CV2_AVAILABLE and arr.shape[0] != display_size:
        arr = cv2.resize(arr, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    return arr


def action_vector_to_index(vec) -> int:
    return sum((1 if v > 0 else 0) << i for i, v in enumerate(vec))


# ---------------------------------------------------------------------------
# Pygame renderer
# ---------------------------------------------------------------------------

def run_pygame(tokenizer, dynamics, start_tokens, start_frame_px, device, display_size, temperature, sample):
    if not PYGAME_AVAILABLE:
        raise ImportError("pygame required: uv add pygame")

    pygame.init()
    screen = pygame.display.set_mode((display_size, display_size + 60))
    pygame.display.set_caption("Dynamics Model Player - Press 1-4 for actions")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    action_dim = dynamics.action_dim

    token_history = [start_tokens.clone()]
    action_history = []
    current_frame_px = start_frame_px.clone()
    frame_count = 0
    last_action_idx = None
    save_count = 0

    key_to_action = {
        getattr(pygame, f"K_{i+1}"): ACTION_VECTORS[i] for i in range(4)
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    token_history = [start_tokens.clone()]
                    action_history = []
                    current_frame_px = start_frame_px.clone()
                    frame_count = 0
                    last_action_idx = None
                    print("Reset to starting frame")
                elif event.key == pygame.K_s:
                    arr = tensor_to_numpy(current_frame_px, display_size)
                    path = f"frame_{save_count:04d}.png"
                    if CV2_AVAILABLE:
                        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                    save_count += 1
                    print(f"Saved {path}")
                elif event.key in key_to_action:
                    action_vec = key_to_action[event.key]
                    action_t = torch.tensor(action_vec, dtype=torch.float32)
                    action_history_step = action_history + [action_t]

                    next_tokens = predict_next_tokens(
                        dynamics, token_history, action_history_step, device, temperature, sample
                    )
                    current_frame_px = decode_tokens(tokenizer, next_tokens, device)

                    token_history.append(next_tokens)
                    action_history = action_history_step
                    frame_count += 1
                    last_action_idx = action_vector_to_index(action_vec)
                    print(f"Frame {frame_count}: Action {last_action_idx + 1} {ACTION_NAMES[last_action_idx]}")

        arr = tensor_to_numpy(current_frame_px, display_size)
        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        screen.fill((30, 30, 30))
        screen.blit(surface, (0, 0))

        info_y = display_size + 5
        screen.blit(font.render(f"Frame: {frame_count}", True, (255, 255, 255)), (10, info_y))
        if last_action_idx is not None:
            screen.blit(
                font.render(f"Action: {last_action_idx + 1} {ACTION_NAMES[last_action_idx]}", True, (100, 255, 100)),
                (10, info_y + 20),
            )
        screen.blit(
            font.render("1-4: Action | R: Reset | S: Save | Q: Quit", True, (150, 150, 150)),
            (10, info_y + 40),
        )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ---------------------------------------------------------------------------
# OpenCV renderer
# ---------------------------------------------------------------------------

def run_cv2(tokenizer, dynamics, start_tokens, start_frame_px, device, display_size, temperature, sample):
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV required: uv add opencv-python")

    action_dim = dynamics.action_dim

    token_history = [start_tokens.clone()]
    action_history = []
    current_frame_px = start_frame_px.clone()
    frame_count = 0
    save_count = 0

    key_to_action = {ord(str(i + 1)): ACTION_VECTORS[i] for i in range(4)}

    print("\nControls:")
    print("  1-4: Apply action and step forward")
    print("  R: Reset to starting frame")
    print("  S: Save current frame")
    print("  Q/ESC: Quit\n")

    while True:
        arr = tensor_to_numpy(current_frame_px, display_size)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"Frame: {frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr, "Press 1-4 for actions, Q to quit", (10, display_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("Dynamics Model Player", bgr)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            token_history = [start_tokens.clone()]
            action_history = []
            current_frame_px = start_frame_px.clone()
            frame_count = 0
            print("Reset to starting frame")
        elif key == ord("s"):
            path = f"frame_{save_count:04d}.png"
            cv2.imwrite(path, bgr)
            save_count += 1
            print(f"Saved {path}")
        elif key in key_to_action:
            action_vec = key_to_action[key]
            action_t = torch.tensor(action_vec, dtype=torch.float32)
            action_history_step = action_history + [action_t]

            next_tokens = predict_next_tokens(
                dynamics, token_history, action_history_step, device, temperature, sample
            )
            current_frame_px = decode_tokens(tokenizer, next_tokens, device)

            token_history.append(next_tokens)
            action_history = action_history_step
            frame_count += 1
            action_idx = action_vector_to_index(action_vec)
            print(f"Frame {frame_count}: Action {action_idx + 1} {ACTION_NAMES[action_idx]}")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive Dynamics Model Player")
    parser.add_argument("--tokenizer-checkpoint", type=str,
                        default="1.video-tokenizer/checkpoints/best_model.pt")
    parser.add_argument("--dynamics-checkpoint", type=str,
                        default="3.dynamics/checkpoints/best_model.pt")
    parser.add_argument("--start-video", type=str,
                        default="assets/doom-samples/doom_0000.mp4")
    parser.add_argument("--display-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true",
                        help="Sample from distribution instead of argmax")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-cv2", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer, _ = load_tokenizer(args.tokenizer_checkpoint, device)
    dynamics, _ = load_dynamics(args.dynamics_checkpoint, device)

    frame_size = tokenizer.encoder.frame_size
    print(f"\nLoading starting frame from: {args.start_video}")
    raw_frame = load_start_frame(args.start_video, frame_size)

    start_tokens = encode_frame(tokenizer, raw_frame, device)
    start_frame_px = decode_tokens(tokenizer, start_tokens, device)

    print(f"Starting frame tokenized: {start_tokens.shape} tokens")
    print(f"Display size: {args.display_size}x{args.display_size}")
    print(f"Temperature: {args.temperature}, sample: {args.sample}\n")

    use_cv2 = args.use_cv2 or not PYGAME_AVAILABLE
    if use_cv2:
        if not PYGAME_AVAILABLE:
            print("pygame not found, falling back to OpenCV")
        run_cv2(tokenizer, dynamics, start_tokens, start_frame_px, device,
                args.display_size, args.temperature, args.sample)
    else:
        run_pygame(tokenizer, dynamics, start_tokens, start_frame_px, device,
                   args.display_size, args.temperature, args.sample)


if __name__ == "__main__":
    main()
