import os
import cv2
import numpy as np

def add_smoke_effect(frame, smoke_texture, alpha=0.5):
    """
    Blends the given smoke_texture onto the frame.
    
    :param frame:         Original frame (H x W x 3) as a NumPy array
    :param smoke_texture: Smoke image (H x W x 3 or 4) as a NumPy array
    :param alpha:         Blend factor (0 to 1). Higher = more smoke.
    :return:              The frame with smoke blended in.
    """
    # Resize smoke texture to match the frame dimensions
    smoke_resized = cv2.resize(smoke_texture, (frame.shape[1], frame.shape[0]))
    
    # If the smoke texture has an alpha channel (RGBA)
    if smoke_resized.shape[2] == 4:
        # Separate the RGB and alpha channels
        smoke_rgb = smoke_resized[:, :, :3]
        smoke_alpha = smoke_resized[:, :, 3] / 255.0  # Range [0..1]

        # Combine the user alpha (function argument) with the texture's own alpha
        smoke_alpha = smoke_alpha * alpha
        
        # Blend using per-pixel alpha from the smoke texture
        smoke_blended = (
            smoke_alpha[..., None] * smoke_rgb +
            (1 - smoke_alpha[..., None]) * frame
        ).astype(np.uint8)
        return smoke_blended
    else:
        # If no alpha channel in the texture, do a simple linear blend
        return cv2.addWeighted(smoke_resized, alpha, frame, 1 - alpha, 0)


def generate_smoke_frames_for_video(input_dir, output_dir, smoke_texture_path, alpha=0.5):
    """
    Reads all image frames in input_dir, applies a synthetic smoke effect,
    and saves them to output_dir with no extra degradations.
    
    :param input_dir:          Directory containing original frames
    :param output_dir:         Directory to save smoke-degraded frames
    :param smoke_texture_path: Path to a smoke texture/image (PNG/JPG, possibly with alpha)
    :param alpha:              How visible the smoke is (0=none, 1=opaque)
    """
    # Read the smoke texture
    smoke_texture = cv2.imread(smoke_texture_path, cv2.IMREAD_UNCHANGED)
    if smoke_texture is None:
        raise ValueError(f"Could not read smoke texture from {smoke_texture_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect valid image files
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    frame_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    frame_files.sort()  # Sort the filenames (lexicographically)

    if not frame_files:
        print(f"No image files found in {input_dir}. Skipping.")
        return

    print(f"Found {len(frame_files)} frames in {input_dir}. Starting smoke effect (alpha={alpha})...")
    
    for i, frame_name in enumerate(frame_files, start=1):
        frame_path = os.path.join(input_dir, frame_name)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read {frame_path}, skipping.")
            # continue
        
        # Add smoke effect
        smoke_frame = add_smoke_effect(frame, smoke_texture, alpha=alpha)
        
        # Save the resulting frame
        output_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(output_path, smoke_frame)

        # Print progress
        if i == len(frame_files):
            print(f"Processed {i}/{len(frame_files)} frames in {input_dir}.")
    
    print(f"Done! Smoke-degraded frames saved to: {output_dir}")


def process_all_videos(
    root_input_dir,
    root_output_dir,
    smoke_texture_path,
    start_idx=1,
    end_idx=80
):
    """
    Loops through subfolders named '01'...'80' in root_input_dir, 
    applies the smoke effect to frames with different alpha values
    depending on the video index, and saves them to corresponding 
    subfolders in root_output_dir.
    
    :param root_input_dir:    Directory with subfolders (01..80) containing frames
    :param root_output_dir:   Directory to store smoke-degraded frames (subfolders will be created)
    :param smoke_texture_path: Path to the smoke texture (with or without alpha channel)
    :param start_idx:         First subfolder to process (default=1 => '01')
    :param end_idx:           Last subfolder to process (default=80 => '80')
    """
    for video_idx in range(start_idx, end_idx + 1):
        # Zero-pad the index to 2 digits => '01', '02', ..., '80'
        subfolder_name = f"{video_idx:02d}"
        
        input_dir = os.path.join(root_input_dir, subfolder_name)
        output_dir = os.path.join(root_output_dir, subfolder_name)
        
        if not os.path.isdir(input_dir):
            print(f"\nInput directory does not exist: {input_dir}. Skipping.")
            continue

        # -----------------------------------------------------------
        # Choose alpha (smoke intensity) based on the video index
        # -----------------------------------------------------------
        if 1 <= video_idx <= 2:
            alpha_value = 0.4   # Light smoke
        elif 3 <= video_idx <= 4:
            alpha_value = 0.6   # Medium smoke
        else:
            alpha_value = 0.8   # Heavy smoke

        print(f"\n=== Processing video folder: {subfolder_name} with alpha={alpha_value} ===")
        
        generate_smoke_frames_for_video(
            input_dir=input_dir,
            output_dir=output_dir,
            smoke_texture_path=smoke_texture_path,
            alpha=alpha_value
        )


if __name__ == "__main__":
    # Example usage:
    root_input_dir =  "/home/santhi/Documents/DACAT/src/Cholec80/data/frames_1fps"
    root_output_dir = "/home/santhi/Documents/DACAT/src/Cholec80/data/frames_1fps_smoke"
    
    # Path to your smoke texture (PNG/JPG). If it has transparency (RGBA), the alpha channel is used.
    smoke_texture_path = "/home/santhi/Documents/DACAT/src/Cholec80/data/smoke_img.jpg"
    
    process_all_videos(
        root_input_dir=root_input_dir,
        root_output_dir=root_output_dir,
        smoke_texture_path=smoke_texture_path,
        start_idx=1,   # Start subfolder
        end_idx=5     # End subfolder
    )
