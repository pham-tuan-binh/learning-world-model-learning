#!/bin/bash
# =============================================================================
# Doom Gameplay Dataset Download Script
# =============================================================================
#
# Dataset: https://github.com/thavlik/doom-gameplay-dataset
# URL: https://doom-gameplay-dataset.nyc3.digitaloceanspaces.com/320x240.zip
# Resolution: 320x240 @ 15 FPS | Format: MP4 | Full size: ~25.8 GiB (~170 hours)
#
# Requirements: curl/wget, unzip, ffmpeg
#
# =============================================================================

set -e

# Configuration
DATASET_URL="https://doom-gameplay-dataset.nyc3.digitaloceanspaces.com/320x240.zip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_FILE="$SCRIPT_DIR/320x240.zip"
TEMP_DIR="$SCRIPT_DIR/.temp_extract"

# Processing parameters - adjust these as needed
MAX_VIDEOS=30           # Number of videos to keep
CLIP_DURATION=30        # Seconds per clip (30 videos × 30s = 15 min total)

echo "Doom Gameplay Dataset Downloader"
echo "================================"
echo "Will download and process $MAX_VIDEOS videos × ${CLIP_DURATION}s each"
echo "Output: $SCRIPT_DIR/*.mp4"
echo ""

# Step 1: Download
if [ -f "$ZIP_FILE" ]; then
    echo "[1/3] ZIP exists, skipping download..."
else
    echo "[1/3] Downloading (~25.8 GiB)..."
    curl -L -o "$ZIP_FILE" "$DATASET_URL" --progress-bar
fi

# Step 2: Extract to temp
echo "[2/3] Extracting..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
unzip -q -o "$ZIP_FILE" -d "$TEMP_DIR"

# Find videos
VIDEOS_DIR=$(find "$TEMP_DIR" -name "*.mp4" -type f -exec dirname {} \; | head -1)

# Step 3: Process and output to data/ (parallel)
echo "[3/3] Processing videos in parallel..."

# Detect number of CPU cores
if command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
else
    NUM_CORES=$(sysctl -n hw.ncpu)  # macOS fallback
fi
echo "      Using $NUM_CORES cores"

# Export variables for subshells
export SCRIPT_DIR CLIP_DURATION

# Process in parallel using xargs
find "$VIDEOS_DIR" -name "*.mp4" -type f | head -n "$MAX_VIDEOS" | \
    xargs -P "$NUM_CORES" -I {} bash -c '
        filename=$(basename "{}")
        echo "      Processing: $filename"
        ffmpeg -y -i "{}" \
            -t "$CLIP_DURATION" \
            -c:v libx264 -preset fast -crf 23 \
            -an -loglevel error \
            "$SCRIPT_DIR/$filename"
    '

# Cleanup
rm -rf "$TEMP_DIR"
echo ""
processed_count=$(ls -1 "$SCRIPT_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
echo "Done! Processed $processed_count videos in $SCRIPT_DIR/"
echo "Total size: $(du -sh "$SCRIPT_DIR" | cut -f1)"
