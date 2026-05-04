#!/bin/bash
# Generate Doom gameplay videos using VizDoom (self-contained, no downloads).
# freedoom2.wad is bundled with the vizdoom pip package.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR}"
NUM_VIDEOS="${2:-100}"
DURATION="${3:-60}"

echo "Installing dependencies..."
pip install vizdoom opencv-python-headless -q

echo "Generating $NUM_VIDEOS Doom gameplay videos → $OUTPUT_DIR"
python3 "$SCRIPT_DIR/generate_doom.py" \
  --output-dir "$OUTPUT_DIR" \
  --num-videos "$NUM_VIDEOS" \
  --fps 15 \
  --duration "$DURATION"
