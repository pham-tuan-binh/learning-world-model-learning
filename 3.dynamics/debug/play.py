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

import time

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
    max_context: int = 4,
) -> torch.Tensor:
    """
    Run one dynamics step given accumulated token and action histories.

    token_history:  list of (N,) tensors, length >= 1
    action_history: list of (A,) tensors, length == len(token_history)
    Returns next token ids (N,).
    """
    token_history = token_history[-max_context:]
    action_history = action_history[-max_context:]
    ctx_tokens = torch.stack(token_history, dim=0).unsqueeze(0).to(device)     # (1, T, N)
    ctx_actions = torch.stack(action_history, dim=0).unsqueeze(0).to(device)   # (1, T, A)

    with torch.no_grad():
        next_tokens = dynamics.predict_next(
            context_tokens=ctx_tokens,
            context_actions=ctx_actions,
            temperature=temperature,
            sample=sample,
        )  # (1, N)

    return next_tokens[0]


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

def run_pygame(tokenizer, dynamics, start_tokens, start_frame_px, device, display_size, temperature, sample, max_context=4):
    if not PYGAME_AVAILABLE:
        raise ImportError("pygame required: uv add pygame")

    pygame.init()

    # Two-zone panel:
    #   zone 1 (top 100px) — 4 action circles
    #   zone 2 (bottom 32px) — single info strip
    PANEL_H  = 132
    ZONE_H   = 100   # height of the button zone
    WIN_W    = display_size

    screen = pygame.display.set_mode((WIN_W, display_size + PANEL_H))
    pygame.display.set_caption("World Model · Doom Dynamics")

    def _font(size, bold=False):
        for name in ("IBM Plex Mono", "Menlo", "Courier New", "monospace"):
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size + 6)

    font_btn  = _font(22, bold=True)   # circle key labels
    font_vec  = _font(12)              # action vector below circles
    font_info = _font(14)              # info strip — bigger & readable

    clock = pygame.time.Clock()

    # Palette — web/tailwind.config.js verbatim
    C_PAPER  = (255, 251, 240)
    C_PAMPAS = (249, 245, 232)
    C_RULE   = (242, 238, 228)
    C_BORDER = (229, 225, 216)
    C_CLOUDY = (178, 175, 168)
    C_MUTE   = (204, 200, 192)
    C_MID    = (127, 125, 120)
    C_DARK   = ( 23,  23,  23)

    # State
    token_history    = [start_tokens.clone()]
    action_history   = []
    current_frame_px = start_frame_px.clone()
    frame_count      = 0
    last_action_idx  = None
    save_count       = 0
    dyn_ms = dec_ms  = 0.0

    key_map = {getattr(pygame, f"K_{i+1}"): i for i in range(4)}

    # Button geometry — vertically centered inside zone 1
    # block = circle (r=28) + 6px gap + vec label (~15px) = 77px tall
    BTN_R   = 28
    VEC_GAP = 6
    BLOCK_H = BTN_R * 2 + VEC_GAP + 15
    BTN_CY  = display_size + (ZONE_H - BLOCK_H) // 2 + BTN_R   # = display_size + 39
    SPACING = 72
    btn_x0  = (WIN_W - SPACING * 3) // 2
    btn_cx  = [btn_x0 + i * SPACING for i in range(4)]

    def step(action_idx):
        nonlocal current_frame_px, frame_count, last_action_idx, dyn_ms, dec_ms
        vec = ACTION_VECTORS[action_idx]
        at  = torch.tensor(vec, dtype=torch.float32)
        ah  = action_history + [at]
        t0  = time.perf_counter()
        nt  = predict_next_tokens(dynamics, token_history, ah, device, temperature, sample, max_context)
        t1  = time.perf_counter()
        current_frame_px = decode_tokens(tokenizer, nt, device)
        t2  = time.perf_counter()
        token_history.append(nt)
        action_history[:] = ah
        frame_count      += 1
        last_action_idx   = action_idx
        dyn_ms            = (t1 - t0) * 1000
        dec_ms            = (t2 - t1) * 1000
        print(f"Frame {frame_count}: Action {action_idx} {ACTION_NAMES[action_idx]} | "
              f"dyn {dyn_ms:.1f}ms  dec {dec_ms:.1f}ms")

    def draw_btn(idx, is_hot, is_prev):
        cx, cy = btn_cx[idx], BTN_CY
        if is_hot:
            pygame.draw.circle(screen, C_DARK,   (cx, cy), BTN_R)
            pygame.draw.circle(screen, C_DARK,   (cx, cy), BTN_R, 2)
            label_col = C_PAPER
            vec_col   = C_MUTE
        elif is_prev:
            pygame.draw.circle(screen, C_PAMPAS, (cx, cy), BTN_R)
            pygame.draw.circle(screen, C_CLOUDY, (cx, cy), BTN_R, 2)
            label_col = C_DARK
            vec_col   = C_MID
        else:
            pygame.draw.circle(screen, C_PAPER,  (cx, cy), BTN_R)
            pygame.draw.circle(screen, C_BORDER, (cx, cy), BTN_R, 2)
            label_col = C_MID
            vec_col   = C_MUTE

        num_s = font_btn.render(str(idx + 1), True, label_col)
        screen.blit(num_s, (cx - num_s.get_width() // 2, cy - num_s.get_height() // 2))

        vec_s = font_vec.render(ACTION_NAMES[idx], True, vec_col)
        screen.blit(vec_s, (cx - vec_s.get_width() // 2, cy + BTN_R + VEC_GAP))

    def blit_row(parts, y):
        """Render [(text, color), ...] left-to-right from x=16."""
        x = 16
        for text, col in parts:
            s = font_info.render(text, True, col)
            screen.blit(s, (x, y))
            x += s.get_width()

    def blit_row_right(parts, y):
        """Render [(text, color), ...] right-aligned, ending 16px from right edge."""
        surfs = [font_info.render(t, True, c) for t, c in parts]
        x = WIN_W - 16 - sum(s.get_width() for s in surfs)
        for s in surfs:
            screen.blit(s, (x, y))
            x += s.get_width()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    token_history[:] = [start_tokens.clone()]
                    action_history.clear()
                    current_frame_px = start_frame_px.clone()
                    frame_count = 0
                    last_action_idx = None
                    dyn_ms = dec_ms = 0.0
                    print("Reset to starting frame")
                elif event.key == pygame.K_s:
                    arr  = tensor_to_numpy(current_frame_px, display_size)
                    path = f"frame_{save_count:04d}.png"
                    if CV2_AVAILABLE:
                        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                    save_count += 1
                    print(f"Saved {path}")

        pressed = pygame.key.get_pressed()
        active_idx = None
        for pkey, idx in key_map.items():
            if pressed[pkey]:
                active_idx = idx
                step(idx)
                break

        # ── Render ─────────────────────────────────────────────────
        screen.fill((0, 0, 0))

        arr      = tensor_to_numpy(current_frame_px, display_size)
        img_surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        screen.blit(img_surf, (0, 0))

        py = display_size
        pygame.draw.rect(screen, C_PAPER, (0, py, WIN_W, PANEL_H))
        pygame.draw.line(screen, C_RULE,  (0, py), (WIN_W, py), 1)

        # ── Zone 1: action circles ──────────────────────────────────
        for i in range(4):
            draw_btn(i, active_idx == i, last_action_idx == i and active_idx != i)

        # ── Divider between zones ───────────────────────────────────
        pygame.draw.line(screen, C_RULE, (0, py + ZONE_H), (WIN_W, py + ZONE_H), 1)

        # ── Zone 2: info strip ──────────────────────────────────────
        # vertical center of the strip
        info_y = py + ZONE_H + (PANEL_H - ZONE_H - font_info.get_height()) // 2

        # left: frame count
        blit_row([
            ("frame  ", C_MID),
            (f"{frame_count:04d}", C_DARK),
        ], info_y)

        # center: last action (only shown after first step)
        if last_action_idx is not None:
            parts = [("last  ", C_MID), (ACTION_NAMES[last_action_idx], C_DARK)]
            total_w = sum(font_info.size(t)[0] for t, _ in parts)
            x = (WIN_W - total_w) // 2
            for text, col in parts:
                s = font_info.render(text, True, col)
                screen.blit(s, (x, info_y))
                x += s.get_width()

        # right: keyboard shortcuts
        blit_row_right([
            ("R", C_DARK), (" reset   ", C_MID),
            ("S", C_DARK), (" save   ",  C_MID),
            ("Q", C_DARK), (" quit",     C_MID),
        ], info_y)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ---------------------------------------------------------------------------
# OpenCV renderer
# ---------------------------------------------------------------------------

def run_cv2(tokenizer, dynamics, start_tokens, start_frame_px, device, display_size, temperature, sample, max_context=4):
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
                dynamics, token_history, action_history_step, device, temperature, sample, max_context
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

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}\n")

    tokenizer, _ = load_tokenizer(args.tokenizer_checkpoint, device)
    dynamics, dyn_cfg = load_dynamics(args.dynamics_checkpoint, device)

    max_context = getattr(getattr(dyn_cfg, "model", None), "num_frames", 4)
    print(f"Context window: {max_context} frames")

    print("Compiling models...")
    tokenizer.decode_indices = torch.compile(tokenizer.decode_indices, mode="reduce-overhead")
    dynamics.predict_next = torch.compile(dynamics.predict_next, mode="reduce-overhead")

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
                args.display_size, args.temperature, args.sample, max_context)
    else:
        run_pygame(tokenizer, dynamics, start_tokens, start_frame_px, device,
                   args.display_size, args.temperature, args.sample, max_context)


if __name__ == "__main__":
    main()
