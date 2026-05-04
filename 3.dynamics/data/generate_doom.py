#!/usr/bin/env python3
"""
Generate diverse Doom gameplay videos using VizDoom.

Self-contained: only requires `pip install vizdoom opencv-python-headless`.
freedoom2.wad is bundled with vizdoom — no external downloads needed.

Diversity comes from:
  - 6 different scenarios (corridor, maze, arena, survival, deathmatch, defend)
  - 4 agent personalities (explorer, fighter, wanderer, rusher)
  - Randomised episode length, action cadence, and decision noise per video
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python-headless", "-q"])
    import cv2

import vizdoom as vzd


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# Each entry: (cfg_name, frame_skip)
# Frame skip controls how "fast" the game feels — lower = smoother but slower
SCENARIOS = [
    ("deadly_corridor",   4),
    ("my_way_home",       4),
    ("defend_the_center", 3),
    ("health_gathering",  4),
    ("defend_the_line",   3),
    ("deathmatch",        2),
]


# ---------------------------------------------------------------------------
# Agent personalities
# ---------------------------------------------------------------------------

class Agent:
    """Base class — subclasses override `act`."""
    def __init__(self, num_buttons: int):
        self.num_buttons = num_buttons

    def act(self, state) -> list[bool]:
        raise NotImplementedError

    def _zeros(self) -> list[bool]:
        return [False] * self.num_buttons


class ExplorerAgent(Agent):
    """Moves forward most of the time; turns when stuck; shoots occasionally."""
    def __init__(self, num_buttons: int):
        super().__init__(num_buttons)
        self._steps = 0
        self._turn_dir = random.choice([-1, 1])
        self._turn_duration = 0

    def act(self, state) -> list[bool]:
        a = self._zeros()
        self._steps += 1

        # Randomly decide to turn for a stretch
        if self._turn_duration > 0:
            self._turn_duration -= 1
            btn = 1 if self._turn_dir > 0 else 0  # TURN_RIGHT / TURN_LEFT (buttons 0,1 in most scenarios)
            if btn < self.num_buttons:
                a[btn] = True
        else:
            # Move forward (button index 2 in most scenarios)
            if self.num_buttons > 2:
                a[2] = True
            # Randomly start a turn
            if random.random() < 0.05:
                self._turn_dir = random.choice([-1, 1])
                self._turn_duration = random.randint(5, 20)

        # Shoot occasionally
        shoot_btn = self.num_buttons - 1
        if random.random() < 0.15 and shoot_btn >= 0:
            a[shoot_btn] = True

        return a


class FighterAgent(Agent):
    """Aggressive: strafes, shoots often, turns rapidly."""
    def __init__(self, num_buttons: int):
        super().__init__(num_buttons)
        self._phase = 0
        self._phase_len = random.randint(8, 20)

    def act(self, state) -> list[bool]:
        a = self._zeros()
        self._phase += 1

        if self._phase >= self._phase_len:
            self._phase = 0
            self._phase_len = random.randint(8, 20)

        progress = self._phase / self._phase_len

        if progress < 0.4:                      # strafe left/right
            idx = 0 if random.random() < 0.5 else 1
            if idx < self.num_buttons:
                a[idx] = True
        elif progress < 0.7:                    # move forward + shoot
            if self.num_buttons > 2:
                a[2] = True
            shoot_btn = self.num_buttons - 1
            if random.random() < 0.6 and shoot_btn >= 0:
                a[shoot_btn] = True
        else:                                   # turn + shoot
            turn_idx = 0 if random.random() < 0.5 else 1
            if turn_idx < self.num_buttons:
                a[turn_idx] = True
            shoot_btn = self.num_buttons - 1
            if random.random() < 0.8 and shoot_btn >= 0:
                a[shoot_btn] = True

        return a


class WandererAgent(Agent):
    """Smooth random walk with momentum — changes direction gradually."""
    def __init__(self, num_buttons: int):
        super().__init__(num_buttons)
        self._action = self._zeros()
        self._hold = 0

    def act(self, state) -> list[bool]:
        if self._hold <= 0:
            self._action = self._zeros()
            # Pick 1–2 random buttons to press
            btns = random.sample(range(self.num_buttons), k=min(2, self.num_buttons))
            for b in btns:
                self._action[b] = True
            self._hold = random.randint(10, 30)
        else:
            self._hold -= 1

        # Small chance to also shoot
        shoot_btn = self.num_buttons - 1
        if random.random() < 0.1 and shoot_btn >= 0:
            self._action[shoot_btn] = True

        return list(self._action)


class RusherAgent(Agent):
    """Fast forward movement with sharp turns; very aggressive."""
    def __init__(self, num_buttons: int):
        super().__init__(num_buttons)
        self._countdown = 0
        self._current = self._zeros()

    def act(self, state) -> list[bool]:
        if self._countdown <= 0:
            self._current = self._zeros()
            r = random.random()
            if r < 0.5:        # surge forward
                if self.num_buttons > 2:
                    self._current[2] = True
                shoot_btn = self.num_buttons - 1
                if random.random() < 0.7 and shoot_btn >= 0:
                    self._current[shoot_btn] = True
                self._countdown = random.randint(15, 40)
            elif r < 0.75:     # sharp left turn
                if self.num_buttons > 0:
                    self._current[0] = True
                self._countdown = random.randint(5, 15)
            else:              # sharp right turn
                if self.num_buttons > 1:
                    self._current[1] = True
                self._countdown = random.randint(5, 15)
        else:
            self._countdown -= 1

        return list(self._current)


AGENTS = [ExplorerAgent, FighterAgent, WandererAgent, RusherAgent]


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def make_game(scenario_cfg: str, width: int, height: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    cfg_path = str(Path(vzd.scenarios_path) / f"{scenario_cfg}.cfg")
    game.load_config(cfg_path)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_render_hud(True)
    game.set_render_crosshair(False)
    game.set_render_decals(True)
    game.set_render_particles(True)
    game.init()
    return game


def generate_video(
    out_path: str,
    scenario_cfg: str,
    frame_skip: int,
    agent_cls,
    fps: int,
    duration_secs: int,
    width: int,
    height: int,
) -> None:
    game = make_game(scenario_cfg, width, height)
    num_buttons = game.get_available_buttons_size()
    agent = agent_cls(num_buttons)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (320, 240),
    )

    total_frames = fps * duration_secs
    game.new_episode()

    for _ in range(total_frames):
        if game.is_episode_finished():
            game.new_episode()
            agent = agent_cls(num_buttons)  # fresh agent each episode

        state = game.get_state()
        frame_rgb = state.screen_buffer     # (H, W, 3) uint8
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        action = agent.act(state)
        game.make_action(action, frame_skip)

    writer.release()
    game.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate diverse Doom gameplay videos")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=int, default=60, help="seconds per video")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    existing = len(list(Path(args.output_dir).glob("*.mp4")))
    if existing >= args.num_videos:
        print(f"Already have {existing} videos in {args.output_dir}, skipping.")
        return

    start = existing
    total = args.num_videos

    print(f"Generating {total - start} videos ({total} total) → {args.output_dir}")
    print(f"  {len(SCENARIOS)} scenarios × {len(AGENTS)} agents | {args.duration}s @ {args.fps}fps each")
    print()

    for i in range(start, total):
        scenario_cfg, frame_skip = random.choice(SCENARIOS)
        agent_cls = random.choice(AGENTS)
        out_path = str(Path(args.output_dir) / f"doom_{i:04d}.mp4")

        print(f"  [{i+1}/{total}] {scenario_cfg} + {agent_cls.__name__} → {out_path}")
        try:
            generate_video(
                out_path=out_path,
                scenario_cfg=scenario_cfg,
                frame_skip=frame_skip,
                agent_cls=agent_cls,
                fps=args.fps,
                duration_secs=args.duration,
                width=320,
                height=240,
            )
        except Exception as e:
            print(f"    WARNING: failed ({e}), skipping")

    generated = len(list(Path(args.output_dir).glob("*.mp4")))
    print(f"\nDone. {generated} videos in {args.output_dir}/")


if __name__ == "__main__":
    main()
