# Web Player

This folder is a static browser player for the trained Doom world model. It runs entirely in the browser via ONNX Runtime Web — no server required.

## How It Works

The browser needs two things the training pipeline doesn't produce directly:

- **ONNX models** — PyTorch checkpoints converted to a format the browser can execute. The export script wraps only the inference-time pieces: a dynamics step (`tokens + action → next tokens`) and a decoder (`tokens → RGB frame`).
- **Seed tokens** — a real tokenized Doom frame for the player to start from. The export script tokenizes the first frame of each seed video using the trained tokenizer.

At runtime the player maps keyboard inputs to latent action IDs `0..3`, expands each into the binary vector in `{-1, 1}^3` that the dynamics model expects, and rolls the world model forward one step at a time.

## Deploy

Run all commands from the **project root**.

**Step 1 — Export models to ONNX**

Requires the trained checkpoints under `3.dynamics/checkpoints/`. Pull them from GCS first if needed:

```bash
gsutil cp -r gs://lwm-world-model-binhpham/checkpoints/3.dynamics/ 3.dynamics/checkpoints/
```

Then export:

```bash
uv run python web/scripts/export_web_bundle.py \
  --dynamics-checkpoint 3.dynamics/checkpoints/dynamics.pt \
  --tokenizer-checkpoint 3.dynamics/checkpoints/video-tokenizer.pt \
  --seed-videos assets/doom-samples/*.mp4 \
  --output-dir web/dist/assets
```

This writes into `web/dist/assets/`:

- `dynamics.onnx` / `dynamics.webgpu.onnx` — next-token predictor (int8 quantized for WASM, FP32 for WebGPU)
- `decoder.onnx` / `decoder.webgpu.onnx` — token-to-RGB decoder
- `manifest.json` — model dimensions and filenames
- `seeds.json` — tokenized starting frames from the seed videos

**Step 2 — Bundle the frontend**

```bash
cd web && npm ci && npm run build
```

Vite bundles the JS/CSS into `web/dist/`. The ONNX models from Step 1 are already in place and untouched.

**Step 3 — Push**

```bash
git add web/dist/assets
git commit -m "update web bundle"
git push
```

Pushing any change under `web/` to `main` or `dynamics` triggers `.github/workflows/deploy-web.yml`, which runs `npm run build` and deploys `web/dist/` to GitHub Pages. The ONNX files are committed and served as-is — the Python export step only runs locally.

## Local Preview

To preview the page without checkpoints:

```bash
cd web && npm install && npm run dev
```

Open the Vite URL with `?preview=1` (usually `http://localhost:5173/?preview=1`). Preview mode uses fake token rollouts and skips ONNX Runtime entirely.

After exporting assets, check the full build locally before pushing:

```bash
cd web && npm run build && npm run preview
```

## Export Options

| Flag | Default | Description |
| --- | --- | --- |
| `--seed-videos` | — | MP4 files to tokenize as starting frames |
| `--seed-tokens` | — | Pre-tokenized `.pt` or `seeds.json` file |
| `--num-seeds` | `8` | How many seeds to write to `seeds.json` |
| `--no-quantize` | off | Write FP32 ONNX instead of int8 |
| `--keep-fp32` | off | Keep intermediate FP32 files alongside quantized ones |
| `--webgpu-precision` | `fp32` | `fp16` or `fp32` for the WebGPU ONNX copies |
