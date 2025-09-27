#!/bin/bash

# Script to extract frames from HeiChole dataset videos at 1 fps
# Usage: bash extract_heichole_frames.sh

# Set paths
VIDEO_DIR="/home/santhi/Documents/DACAT/src/Cholec80/data/HeiChole/videos"
FRAMES_DIR="/home/santhi/Documents/DACAT/src/Cholec80/data/HeiChole/frames_1fps"

echo "Starting frame extraction from HeiChole videos..."
echo "Video directory: $VIDEO_DIR"
echo "Output directory: $FRAMES_DIR"
echo ""

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it first:"
    echo "sudo apt update && sudo apt install ffmpeg"
    exit 1
fi

# Process each HeiChole video
for video_file in "$VIDEO_DIR"/Hei-Chole*.mp4; do
    if [[ -f "$video_file" ]]; then
        # Extract video number from filename (e.g., Hei-Chole1.mp4 -> 1)
        video_name=$(basename "$video_file" .mp4)
        video_number=$(echo "$video_name" | sed 's/Hei-Chole//')
        
        # Create output directory for this video (using 2-digit format)
        output_dir="$FRAMES_DIR/$(printf "%02d" "$video_number")"
        mkdir -p "$output_dir"
        
        echo "Processing $video_name -> $output_dir"
        
        # Extract frames at 1 fps using ffmpeg
        ffmpeg -hide_banner -i "$video_file" -r 1 -start_number 0 "$output_dir/%08d.jpg" -y
        
        # Check if extraction was successful
        if [[ $? -eq 0 ]]; then
            frame_count=$(ls "$output_dir"/*.jpg 2>/dev/null | wc -l)
            echo "✓ Successfully extracted $frame_count frames from $video_name"
        else
            echo "✗ Failed to extract frames from $video_name"
        fi
        echo ""
    fi
done

echo "Frame extraction completed!"
echo ""
echo "Summary:"
for dir in "$FRAMES_DIR"/*/; do
    if [[ -d "$dir" ]]; then
        video_num=$(basename "$dir")
        frame_count=$(ls "$dir"/*.jpg 2>/dev/null | wc -l)
        echo "Video $video_num: $frame_count frames"
    fi
done