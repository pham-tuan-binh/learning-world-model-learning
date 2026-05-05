#!/usr/bin/env python3
"""
Export trained PyTorch checkpoints into a static web bundle.

The browser player needs two ONNX graphs:
1. dynamics.onnx: token context + latent actions -> next token frame
2. decoder.onnx: token frame -> RGB pixels

It also needs a manifest and seed token frames. Seed tokens should come from the
token dynamics cache built during training so the rollout starts from a real
Doom frame rather than random token ids.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = REPO_ROOT / "1.video-tokenizer"
INVERSE_DYNAMICS_DIR = REPO_ROOT / "2.inverse-dynamics"
DYNAMICS_DIR = REPO_ROOT / "3.dynamics"


def add_project_paths() -> None:
    """Add local workspace packages to sys.path for checkpoint loading."""
    for path in (DYNAMICS_DIR, INVERSE_DYNAMICS_DIR, TOKENIZER_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_checkpoint(path: Path, project_dir: Path) -> dict[str, Any]:
    """
    Load a PyTorch checkpoint saved by one of the numbered project folders.

    Args:
        path: Checkpoint path.
        project_dir: Folder that contains the matching ``config.py`` module.

    Returns:
        checkpoint: Loaded checkpoint dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with path.open("rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"{path} is a Git LFS pointer, not checkpoint bytes.")

    sys.modules.pop("config", None)
    project_path = str(project_dir)
    previous_path = list(sys.path)
    sys.path.insert(0, project_path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("config", None)


def get_nested_attr(obj: Any, path: str, default: Any = None) -> Any:
    """
    Read a dotted attribute path from a checkpoint config object.

    Args:
        obj: Root object.
        path: Dotted attribute path such as ``model.embed_dim``.
        default: Value returned when any part is missing.

    Returns:
        value: Attribute value or default.
    """
    current = obj
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def count_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    """
    Count transformer blocks in a checkpoint state dict.

    Args:
        state_dict: Model state dictionary.
        prefix: Key prefix ending before ``blocks.<index>``.

    Returns:
        count: Number of unique block indices.
    """
    indices = set()
    marker = f"{prefix}.blocks."
    for key in state_dict:
        if key.startswith(marker):
            remainder = key[len(marker) :]
            indices.add(int(remainder.split(".", 1)[0]))
    return len(indices)


def infer_num_bins(basis: torch.Tensor) -> int:
    """
    Infer FSQ bin count from the stored positional basis.

    Args:
        basis: Tensor like ``[1, L, L^2, ...]``.

    Returns:
        num_bins: Number of quantization bins per latent dimension.
    """
    if basis.numel() < 2:
        return 2
    return int(basis.flatten()[1].item())


def build_tokenizer(checkpoint: dict[str, Any], num_heads_override: int | None) -> nn.Module:
    """
    Build the video tokenizer from a checkpoint.

    Args:
        checkpoint: Loaded tokenizer checkpoint.
        num_heads_override: Optional attention-head count when config is absent.

    Returns:
        model: Loaded ``VideoTokenizer`` module in eval mode.
    """
    from video_tokenizer import VideoTokenizer

    state = checkpoint["model_state_dict"]
    config = checkpoint.get("config")
    patch_weight = state["encoder.patch_embed.proj.weight"]
    embed_dim = int(patch_weight.shape[0])
    in_channels = int(patch_weight.shape[1])
    patch_size = int(patch_weight.shape[2])
    grid_size = int(math.sqrt(state["encoder.pos_encoding.x_positions"].numel()))
    frame_size = grid_size * patch_size
    latent_dim = int(state["encoder.to_latent.weight"].shape[0])
    num_bins = infer_num_bins(state["encoder.quantizer.basis"])
    num_blocks = count_blocks(state, "encoder.transformer")

    num_heads = num_heads_override or get_nested_attr(config, "model.num_heads", 8)
    model = VideoTokenizer(
        in_channels=get_nested_attr(config, "model.in_channels", in_channels),
        frame_size=get_nested_attr(config, "model.frame_size", frame_size),
        num_frames=get_nested_attr(config, "model.num_frames", 4),
        patch_size=get_nested_attr(config, "model.patch_size", patch_size),
        embed_dim=get_nested_attr(config, "model.embed_dim", embed_dim),
        num_heads=num_heads,
        num_blocks=get_nested_attr(config, "model.num_blocks", num_blocks),
        latent_dim=get_nested_attr(config, "model.latent_dim", latent_dim),
        num_bins=get_nested_attr(config, "model.num_bins", num_bins),
        dropout=0.0,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def build_dynamics(checkpoint: dict[str, Any], num_heads_override: int | None) -> nn.Module:
    """
    Build the token dynamics model from a checkpoint.

    Args:
        checkpoint: Loaded dynamics checkpoint.
        num_heads_override: Optional attention-head count when config is absent.

    Returns:
        model: Loaded ``TokenDynamicsModel`` module in eval mode.
    """
    from dynamics import TokenDynamicsModel

    state = checkpoint["model_state_dict"]
    config = checkpoint.get("config")
    vocab_size, embed_dim = state["token_embed.weight"].shape
    action_dim = int(state["action_proj.0.weight"].shape[1])
    n_actions = int(state["action_id_embed.weight"].shape[0])
    num_blocks = count_blocks(state, "transformer")
    grid_size = int(math.sqrt(state["pos_encoding.x_positions"].numel()))
    use_adaptive = any("gate_mlp" in key for key in state)

    num_heads = num_heads_override or get_nested_attr(config, "model.num_heads", 8)
    model = TokenDynamicsModel(
        vocab_size=get_nested_attr(config, "model.vocab_size", int(vocab_size)),
        num_patches=get_nested_attr(config, "model.num_patches", grid_size * grid_size),
        action_dim=get_nested_attr(config, "model.action_dim", action_dim),
        n_actions=get_nested_attr(config, "model.n_actions", n_actions),
        embed_dim=get_nested_attr(config, "model.embed_dim", int(embed_dim)),
        num_heads=num_heads,
        num_blocks=get_nested_attr(config, "model.num_blocks", num_blocks),
        grid_size=get_nested_attr(config, "model.grid_size", grid_size),
        max_frames=32,
        dropout=0.0,
        use_adaptive_conditioning=get_nested_attr(
            config,
            "model.use_adaptive_conditioning",
            use_adaptive,
        ),
    )
    model.load_state_dict(state)
    model.eval()
    return model


class DynamicsStep(nn.Module):
    """
    ONNX wrapper that predicts exactly one next token frame.

    Args:
        model: Trained token dynamics model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next tokens from a token/action context.

        Args:
            tokens: Token context, shape ``(B, T, N)``.
            actions: Latent action context, shape ``(B, T, A)``.

        Returns:
            next_tokens: Next token frame, shape ``(B, N)``.
        """
        logits = self.model(tokens, actions)
        next_logits = logits[:, -1]
        return next_logits.argmax(dim=-1)


class TokenDecoder(nn.Module):
    """
    ONNX wrapper around ``VideoTokenizer.decode_indices``.

    Args:
        tokenizer: Trained video tokenizer.
    """

    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices into RGB frames.

        Args:
            indices: Token indices, shape ``(B, T, N)``.

        Returns:
            frames: RGB frames in roughly ``[0, 1]``, shape ``(B, T, 3, H, W)``.
        """
        frames = self.tokenizer.decode_indices(indices)
        return frames.clamp(0.0, 1.0)


def export_dynamics(model: nn.Module, output_path: Path, num_patches: int, action_dim: int) -> None:
    """
    Export the dynamics step wrapper to ONNX.

    Args:
        model: Loaded dynamics model.
        output_path: Destination ONNX path.
        num_patches: Number of visual tokens per frame.
        action_dim: Dimension of latent action vectors.
    """
    wrapper = DynamicsStep(model)
    tokens = torch.zeros(1, 1, num_patches, dtype=torch.long)
    actions = torch.full((1, 1, action_dim), -1.0, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            (tokens, actions),
            output_path,
            input_names=["tokens", "actions"],
            output_names=["next_tokens"],
            dynamic_axes={
                "tokens": {1: "context_frames"},
                "actions": {1: "context_frames"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )


def export_decoder(model: nn.Module, output_path: Path, num_patches: int) -> None:
    """
    Export the token decoder wrapper to ONNX.

    Args:
        model: Loaded tokenizer model.
        output_path: Destination ONNX path.
        num_patches: Number of visual tokens per frame.
    """
    wrapper = TokenDecoder(model)
    indices = torch.zeros(1, 1, num_patches, dtype=torch.long)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            indices,
            output_path,
            input_names=["indices"],
            output_names=["frames"],
            dynamic_axes={
                "indices": {1: "frames"},
                "frames": {1: "frames"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )


def quantize_onnx_model(input_path: Path, output_path: Path) -> None:
    """
    Quantize ONNX model weights for browser delivery.

    Args:
        input_path: FP32 ONNX model path.
        output_path: Quantized ONNX model path.

    Dynamic quantization keeps activations dynamic while storing supported
    matrix weights as int8. That is a practical default for static web hosting:
    no calibration data is needed, model files are smaller, and the browser can
    execute the result with ONNX Runtime Web's WASM backend.
    """
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.ERROR)
    try:
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    finally:
        root_logger.setLevel(previous_level)


def convert_onnx_to_fp16(input_path: Path, output_path: Path) -> None:
    """
    Convert an ONNX model to FP16 for WebGPU execution.

    Args:
        input_path: FP32 ONNX model path.
        output_path: Destination FP16 ONNX model path.

    ``keep_io_types=True`` keeps the browser-facing inputs and outputs in their
    original types. Internally, supported floating point weights and operators
    use FP16, which is a better match for GPU execution and cuts transfer size.
    """
    model = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)


def tokenize_videos(tokenizer: nn.Module, video_paths: list[Path]) -> torch.Tensor:
    """
    Tokenize the first frame of each video file into seed tokens.

    Args:
        tokenizer: Loaded video tokenizer.
        video_paths: List of video file paths.

    Returns:
        tokens: Tensor shaped ``(S, N)`` containing one seed per video.
    """
    import cv2

    frame_size = tokenizer.config["frame_size"]
    seeds = []
    for path in video_paths:
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video: {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        x = tensor.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            indices, _ = tokenizer.encode(x)
        seeds.append(indices[0, 0])
        print(f"  Tokenized {path.name}")
    return torch.stack(seeds)


def action_table(n_actions: int, action_dim: int) -> list[dict[str, Any]]:
    """
    Build the latent action lookup table served to the browser.

    Args:
        n_actions: Number of discrete latent actions.
        action_dim: Dimension of binary latent action vectors.

    Returns:
        actions: List of action metadata dictionaries.
    """
    actions = []
    for action_id in range(n_actions):
        vector = [1 if ((action_id >> bit) & 1) else -1 for bit in range(action_dim)]
        actions.append({"id": action_id, "label": f"A{action_id}", "vector": vector})
    return actions


def load_seed_tensor(path: Path) -> torch.Tensor:
    """
    Load seed token frames from a tensor, JSON file, or training cache.

    Args:
        path: Seed source path.

    Returns:
        tokens: Tensor shaped ``(S, N)`` containing seed token frames.
    """
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        seeds = payload["seeds"] if isinstance(payload, dict) else payload
        return torch.tensor([seed["tokens"] if isinstance(seed, dict) else seed for seed in seeds])

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "tokens" in payload:
        tokens = payload["tokens"]
        if tokens.dim() == 3:
            return tokens[:, 0, :]
        if tokens.dim() == 2:
            return tokens
    if torch.is_tensor(payload):
        if payload.dim() == 3:
            return payload[:, 0, :]
        if payload.dim() == 2:
            return payload
        if payload.dim() == 1:
            return payload.unsqueeze(0)

    raise ValueError(f"Could not read seed tokens from {path}")


def write_seeds(output_path: Path, seeds: torch.Tensor, count: int) -> None:
    """
    Write browser seed token JSON.

    Args:
        output_path: Destination ``seeds.json`` path.
        seeds: Tensor shaped ``(S, N)``.
        count: Maximum number of seeds to write.
    """
    seeds = seeds.long().cpu()
    selected = seeds[:count]
    payload = {
        "seeds": [
            {"name": f"Seed {idx + 1}", "tokens": seed.tolist()}
            for idx, seed in enumerate(selected)
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2))


def write_manifest(
    output_path: Path,
    tokenizer: nn.Module,
    dynamics: nn.Module,
    max_context_frames: int,
    quantized: bool,
    include_webgpu: bool,
    webgpu_precision: str,
) -> None:
    """
    Write browser manifest JSON.

    Args:
        output_path: Destination ``manifest.json`` path.
        tokenizer: Loaded video tokenizer.
        dynamics: Loaded dynamics model.
        max_context_frames: Number of token frames retained in browser context.
        quantized: Whether the served ONNX files were quantized.
    """
    payload = {
        "formatVersion": 1,
        "dynamicsModel": "dynamics.onnx",
        "decoderModel": "decoder.onnx",
        **(
            {
                "webgpuDynamicsModel": "dynamics.webgpu.onnx",
                "webgpuDecoderModel": "decoder.webgpu.onnx",
                "webgpuModelPrecision": webgpu_precision,
            }
            if include_webgpu
            else {}
        ),
        "quantized": bool(quantized),
        "modelPrecision": "int8-dynamic" if quantized else "fp32",
        "executionProviders": ["wasm"] if quantized else ["webgpu", "wasm"],
        "frameSize": int(tokenizer.config["frame_size"]),
        "numPatches": int(dynamics.num_patches),
        "gridSize": int(dynamics.grid_size),
        "vocabSize": int(dynamics.vocab_size),
        "actionDim": int(dynamics.action_dim),
        "nActions": int(dynamics.n_actions),
        "maxContextFrames": int(max_context_frames),
        "actions": action_table(int(dynamics.n_actions), int(dynamics.action_dim)),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Export a GitHub Pages web bundle.")
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "web/public/assets")
    parser.add_argument("--seed-cache", type=Path)
    parser.add_argument("--seed-tokens", type=Path)
    parser.add_argument("--seed-videos", type=Path, nargs="+", help="Video files to tokenize as seeds.")
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--tokenizer-num-heads", type=int)
    parser.add_argument("--dynamics-num-heads", type=int)
    parser.add_argument(
        "--max-context-frames",
        type=int,
        help="Browser rollout context length. Defaults to the training clip length.",
    )
    parser.add_argument(
        "--allow-placeholder-seed",
        action="store_true",
        help="Write an all-zero seed when no real seed source is provided.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Write FP32 ONNX models instead of int8 dynamic-quantized models.",
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep intermediate dynamics.fp32.onnx and decoder.fp32.onnx files.",
    )
    parser.add_argument(
        "--no-webgpu-fp32",
        action="store_true",
        help=(
            "Do not write WebGPU ONNX copies. By default, quantized WASM models "
            "and FP32 WebGPU models are both written."
        ),
    )
    parser.add_argument(
        "--webgpu-precision",
        choices=("fp16", "fp32"),
        default="fp32",
        help=(
            "Precision for WebGPU ONNX copies. FP32 is the compatibility default; "
            "FP16 is smaller but may fall back to WASM on some browser/GPU stacks."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Export ONNX models and static metadata for the browser player."""
    args = parse_args()
    add_project_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_checkpoint = load_checkpoint(args.tokenizer_checkpoint, TOKENIZER_DIR)
    dynamics_checkpoint = load_checkpoint(args.dynamics_checkpoint, DYNAMICS_DIR)

    tokenizer = build_tokenizer(tokenizer_checkpoint, args.tokenizer_num_heads)
    dynamics = build_dynamics(dynamics_checkpoint, args.dynamics_num_heads)

    if tokenizer.codebook_size != dynamics.vocab_size:
        raise ValueError(
            "Tokenizer codebook size and dynamics vocabulary size do not match: "
            f"{tokenizer.codebook_size} != {dynamics.vocab_size}"
        )
    if tokenizer.num_patches != dynamics.num_patches:
        raise ValueError(
            "Tokenizer patch count and dynamics patch count do not match: "
            f"{tokenizer.num_patches} != {dynamics.num_patches}"
        )

    quantized = not args.no_quantize
    dynamics_fp32_path = (
        args.output_dir / "dynamics.fp32.onnx"
        if quantized
        else args.output_dir / "dynamics.onnx"
    )
    decoder_fp32_path = (
        args.output_dir / "decoder.fp32.onnx"
        if quantized
        else args.output_dir / "decoder.onnx"
    )

    print(f"Exporting {dynamics_fp32_path.name}...")
    export_dynamics(
        dynamics,
        dynamics_fp32_path,
        num_patches=dynamics.num_patches,
        action_dim=dynamics.action_dim,
    )

    print(f"Exporting {decoder_fp32_path.name}...")
    export_decoder(
        tokenizer,
        decoder_fp32_path,
        num_patches=dynamics.num_patches,
    )

    if quantized:
        print("Quantizing dynamics.onnx...")
        quantize_onnx_model(dynamics_fp32_path, args.output_dir / "dynamics.onnx")
        print("Quantizing decoder.onnx...")
        quantize_onnx_model(decoder_fp32_path, args.output_dir / "decoder.onnx")
        include_webgpu = not args.no_webgpu_fp32
        if include_webgpu and args.webgpu_precision == "fp16":
            print("Converting dynamics.webgpu.onnx to FP16...")
            convert_onnx_to_fp16(dynamics_fp32_path, args.output_dir / "dynamics.webgpu.onnx")
            print("Converting decoder.webgpu.onnx to FP16...")
            convert_onnx_to_fp16(decoder_fp32_path, args.output_dir / "decoder.webgpu.onnx")
            if not args.keep_fp32:
                dynamics_fp32_path.unlink(missing_ok=True)
                decoder_fp32_path.unlink(missing_ok=True)
        elif include_webgpu:
            dynamics_fp32_path.replace(args.output_dir / "dynamics.webgpu.onnx")
            decoder_fp32_path.replace(args.output_dir / "decoder.webgpu.onnx")
        elif not args.keep_fp32:
            dynamics_fp32_path.unlink(missing_ok=True)
            decoder_fp32_path.unlink(missing_ok=True)
    else:
        include_webgpu = False

    seed_source = args.seed_tokens or args.seed_cache
    if args.seed_videos:
        print("Tokenizing seed videos...")
        seeds = tokenize_videos(tokenizer, args.seed_videos)
    elif seed_source is not None:
        seeds = load_seed_tensor(seed_source)
    elif args.allow_placeholder_seed:
        seeds = torch.zeros(1, dynamics.num_patches, dtype=torch.long)
    else:
        raise ValueError(
            "Provide --seed-videos, --seed-tokens, or --seed-cache so the player "
            "starts from real Doom tokens. Use --allow-placeholder-seed only for wiring tests."
        )

    max_context_frames = args.max_context_frames or get_nested_attr(
        dynamics_checkpoint.get("config"),
        "model.num_frames",
        4,
    )

    print("Writing manifest.json and seeds.json...")
    write_manifest(
        args.output_dir / "manifest.json",
        tokenizer,
        dynamics,
        max_context_frames=max_context_frames,
        quantized=quantized,
        include_webgpu=include_webgpu,
        webgpu_precision=args.webgpu_precision,
    )
    write_seeds(args.output_dir / "seeds.json", seeds, args.num_seeds)
    print(f"Web bundle written to {args.output_dir}")


if __name__ == "__main__":
    main()
