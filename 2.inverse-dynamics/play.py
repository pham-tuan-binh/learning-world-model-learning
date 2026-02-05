"""
Interactive World Model Player.

Play the learned world model by pressing keys 1-8 to select actions.
The model predicts the next frame based on current frame + selected action.

Usage:
    uv run python play.py --checkpoint checkpoints/best_model.pt --start-image path/to/image.png
    uv run python play.py --checkpoint checkpoints/best_model.pt --start-video path/to/video.mp4

Controls:
    1-8: Select action (maps to 8 discrete action vectors)
    R: Reset to initial frame
    S: Save current frame
    Q/ESC: Quit
"""

import argparse
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

from inverse_dynamics import LatentActionModel


# Action mapping: key -> action vector
# With action_dim=3, we have 2^3 = 8 possible actions
# Each dimension is either -1 or +1 (binary quantization)
ACTION_MAP = {
    pygame.K_1 if PYGAME_AVAILABLE else ord('1'): [-1, -1, -1],  # Action 0
    pygame.K_2 if PYGAME_AVAILABLE else ord('2'): [+1, -1, -1],  # Action 1
    pygame.K_3 if PYGAME_AVAILABLE else ord('3'): [-1, +1, -1],  # Action 2
    pygame.K_4 if PYGAME_AVAILABLE else ord('4'): [+1, +1, -1],  # Action 3
    pygame.K_5 if PYGAME_AVAILABLE else ord('5'): [-1, -1, +1],  # Action 4
    pygame.K_6 if PYGAME_AVAILABLE else ord('6'): [+1, -1, +1],  # Action 5
    pygame.K_7 if PYGAME_AVAILABLE else ord('7'): [-1, +1, +1],  # Action 6
    pygame.K_8 if PYGAME_AVAILABLE else ord('8'): [+1, +1, +1],  # Action 7
}

# Reverse mapping for display
ACTION_NAMES = {
    0: "[-1,-1,-1]",
    1: "[+1,-1,-1]",
    2: "[-1,+1,-1]",
    3: "[+1,+1,-1]",
    4: "[-1,-1,+1]",
    5: "[+1,-1,+1]",
    6: "[-1,+1,+1]",
    7: "[+1,+1,+1]",
}


def load_model(checkpoint_path: str, device: str) -> LatentActionModel:
    """Load the trained model from checkpoint."""
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
    print(f"Frame size: {config.model.frame_size}")
    print(f"Action vocabulary: {config.model.n_actions}")

    return model, config


def load_start_frame(path: str, frame_size: int) -> torch.Tensor:
    """
    Load starting frame from image or video.

    Returns:
        frame: Tensor of shape (C, H, W) with values in [0, 1]
    """
    path = Path(path)

    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) required for loading images/videos. Install with: uv add opencv-python")

    if path.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm']:
        # Load first frame from video
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video: {path}")
    else:
        # Load image
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model's expected size
    frame = cv2.resize(frame, (frame_size, frame_size))

    # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    return frame


def create_dummy_frame(frame_size: int) -> torch.Tensor:
    """Create a dummy starting frame (random noise or solid color)."""
    # Create a simple gradient or random frame
    frame = torch.rand(3, frame_size, frame_size)
    return frame


def predict_next_frame(
    model: LatentActionModel,
    current_frame: torch.Tensor,
    action: list,
    device: str,
) -> torch.Tensor:
    """
    Predict the next frame given current frame and action.

    Args:
        model: The trained LatentActionModel
        current_frame: Current frame tensor (C, H, W) in [0, 1]
        action: Action vector [a1, a2, a3] where each is -1 or +1
        device: Device to run on

    Returns:
        next_frame: Predicted frame tensor (C, H, W) in [0, 1]
    """
    with torch.no_grad():
        # Prepare frame: (C, H, W) -> (1, 2, C, H, W)
        # The decoder expects T frames and uses frames[:, :-1] as input
        # So we provide [current_frame, dummy] and it uses current_frame
        frame_batch = current_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        frame_batch = frame_batch.repeat(1, 2, 1, 1, 1)  # (1, 2, C, H, W)
        frame_batch = frame_batch.to(device)

        # Prepare action: (1, 1, A)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        action_tensor = action_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, A)
        action_tensor = action_tensor.to(device)

        # Decode
        pred_frame = model.decode(frame_batch, action_tensor)  # (1, 1, C, H, W)

        # Extract single frame
        next_frame = pred_frame[0, 0].cpu()  # (C, H, W)

    return next_frame


def tensor_to_surface(frame: torch.Tensor, display_size: int) -> "pygame.Surface":
    """Convert frame tensor to pygame surface."""
    # (C, H, W) -> (H, W, C), scale to [0, 255]
    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Resize for display
    if CV2_AVAILABLE:
        frame_np = cv2.resize(frame_np, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

    # Create pygame surface
    surface = pygame.surfarray.make_surface(frame_np.swapaxes(0, 1))
    return surface


def action_vector_to_index(action: list) -> int:
    """Convert action vector to index (0-7)."""
    # [-1, +1] -> [0, 1]
    bits = [(a + 1) // 2 for a in action]
    # Binary to decimal
    return bits[0] * 1 + bits[1] * 2 + bits[2] * 4


def run_pygame(model, config, start_frame, device, display_size):
    """Run the interactive player using pygame."""
    if not PYGAME_AVAILABLE:
        raise ImportError("pygame required. Install with: uv add pygame")

    pygame.init()

    # Create window
    screen = pygame.display.set_mode((display_size, display_size + 60))
    pygame.display.set_caption("World Model Player - Press 1-8 for actions")

    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    current_frame = start_frame.clone()
    initial_frame = start_frame.clone()
    last_action = None
    frame_count = 0
    save_count = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset to initial frame
                    current_frame = initial_frame.clone()
                    frame_count = 0
                    last_action = None
                    print("Reset to initial frame")
                elif event.key == pygame.K_s:
                    # Save current frame
                    save_path = f"frame_{save_count:04d}.png"
                    frame_np = (current_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if CV2_AVAILABLE:
                        cv2.imwrite(save_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                        print(f"Saved frame to {save_path}")
                    save_count += 1
                elif event.key in ACTION_MAP:
                    # Predict next frame with selected action
                    action = ACTION_MAP[event.key]
                    current_frame = predict_next_frame(model, current_frame, action, device)
                    last_action = action
                    frame_count += 1

                    action_idx = action_vector_to_index(action)
                    print(f"Frame {frame_count}: Action {action_idx + 1} {ACTION_NAMES[action_idx]}")

        # Draw frame
        surface = tensor_to_surface(current_frame, display_size)
        screen.fill((30, 30, 30))
        screen.blit(surface, (0, 0))

        # Draw info bar
        info_y = display_size + 5
        frame_text = font.render(f"Frame: {frame_count}", True, (255, 255, 255))
        screen.blit(frame_text, (10, info_y))

        if last_action is not None:
            action_idx = action_vector_to_index(last_action)
            action_text = font.render(f"Action: {action_idx + 1} {ACTION_NAMES[action_idx]}", True, (100, 255, 100))
            screen.blit(action_text, (10, info_y + 20))

        controls_text = font.render("1-8: Action | R: Reset | S: Save | Q: Quit", True, (150, 150, 150))
        screen.blit(controls_text, (10, info_y + 40))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def run_cv2(model, config, start_frame, device, display_size):
    """Run the interactive player using OpenCV (fallback if pygame not available)."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV required. Install with: uv add opencv-python")

    current_frame = start_frame.clone()
    initial_frame = start_frame.clone()
    frame_count = 0
    save_count = 0

    print("\nControls:")
    print("  1-8: Select action")
    print("  R: Reset to initial frame")
    print("  S: Save current frame")
    print("  Q/ESC: Quit")
    print()

    cv2_action_map = {
        ord('1'): [-1, -1, -1],
        ord('2'): [+1, -1, -1],
        ord('3'): [-1, +1, -1],
        ord('4'): [+1, +1, -1],
        ord('5'): [-1, -1, +1],
        ord('6'): [+1, -1, +1],
        ord('7'): [-1, +1, +1],
        ord('8'): [+1, +1, +1],
    }

    while True:
        # Convert frame for display
        frame_np = (current_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_np = cv2.resize(frame_np, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Add text overlay
        cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame_bgr, "Press 1-8 for actions, Q to quit", (10, display_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("World Model Player", frame_bgr)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('r'):
            current_frame = initial_frame.clone()
            frame_count = 0
            print("Reset to initial frame")
        elif key == ord('s'):
            save_path = f"frame_{save_count:04d}.png"
            cv2.imwrite(save_path, frame_bgr)
            print(f"Saved frame to {save_path}")
            save_count += 1
        elif key in cv2_action_map:
            action = cv2_action_map[key]
            current_frame = predict_next_frame(model, current_frame, action, device)
            frame_count += 1

            action_idx = action_vector_to_index(action)
            print(f"Frame {frame_count}: Action {action_idx + 1} {ACTION_NAMES[action_idx]}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive World Model Player")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--start-image", type=str,
                        help="Path to starting image")
    parser.add_argument("--start-video", type=str,
                        help="Path to starting video (uses first frame)")
    parser.add_argument("--display-size", type=int, default=512,
                        help="Display window size (default: 512)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--use-cv2", action="store_true",
                        help="Use OpenCV instead of pygame for display")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load or create starting frame
    if args.start_image:
        start_frame = load_start_frame(args.start_image, config.model.frame_size)
        print(f"Loaded starting frame from: {args.start_image}")
    elif args.start_video:
        start_frame = load_start_frame(args.start_video, config.model.frame_size)
        print(f"Loaded starting frame from video: {args.start_video}")
    else:
        print("No starting image/video provided, using random noise")
        start_frame = create_dummy_frame(config.model.frame_size)

    print(f"\nStarting interactive player...")
    print(f"Frame size: {config.model.frame_size}x{config.model.frame_size}")
    print(f"Display size: {args.display_size}x{args.display_size}")
    print()

    # Run player
    if args.use_cv2 or not PYGAME_AVAILABLE:
        if not PYGAME_AVAILABLE:
            print("pygame not available, using OpenCV")
        run_cv2(model, config, start_frame, device, args.display_size)
    else:
        run_pygame(model, config, start_frame, device, args.display_size)


if __name__ == "__main__":
    main()
