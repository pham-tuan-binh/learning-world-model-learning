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

# Step 3: Process and output to data/
echo "[3/3] Processing videos..."

count=0
for video in "$VIDEOS_DIR"/*.mp4; do
    [ $count -ge $MAX_VIDEOS ] && break

    filename=$(basename "$video")
    output="$SCRIPT_DIR/$filename"

    printf "  [%02d/%02d] %s\n" $((count + 1)) $MAX_VIDEOS "$filename"

    ffmpeg -y -i "$video" \
        -t "$CLIP_DURATION" \
        -c:v libx264 -preset fast -crf 23 \
        -an -loglevel error \
        "$output"

    count=$((count + 1))
done

# Cleanup
rm -rf "$TEMP_DIR"
echo ""
echo "Done! Processed $count videos in $SCRIPT_DIR/"
echo "Total: $(du -sh "$SCRIPT_DIR"/*.mp4 | tail -1 | cut -f1)"
