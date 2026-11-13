#!/usr/bin/env python3
"""
videoSlowerWithCounter.py
Creates a slowed-down video at 10 FPS with a frame counter in the bottom right corner.
"""

from pathlib import Path
import cv2


def main():
    # Set paths relative to the script location (inside Scripts/)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent  # BeadDetection root folder

    #video_path = root_dir / "video1Copy.mp4"
    #output_video_path = root_dir / "video1_slowed_10fps_counter.mp4"

    video_path = root_dir / "video2TrimmedCopy.mp4"
    output_video_path = root_dir / "video2_slowed_10fps_counter.mp4"

    if not video_path.exists():
        print(f"[Error] Could not find video at: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path.name}")
        return

    # Get original properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Target frame rate for slow motion
    target_fps = 10.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, target_fps, (width, height))

    if not out.isOpened():
        print("[Error] Could not open video writer.")
        cap.release()
        return

    print(f"Processing video with frame counter at {target_fps} FPS...")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Format the text string: e.g., "Frame: 45 / 300"
        text = f"Frame: {frame_count} / {total_frames}"

        # Configure font scale and thickness based on video dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, width / 1200.0)
        thickness = max(1, int(width / 600.0))

        # Calculate text size to position it at the bottom-right
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Margin from the bottom-right edges
        margin_x = 20
        margin_y = 20
        org_x = width - text_width - margin_x
        org_y = height - margin_y

        # Optional: Add a dark drop shadow or background outline for visibility
        # Draw black outline/shadow first
        cv2.putText(frame, text, (org_x, org_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Draw white text on top
        cv2.putText(frame, text, (org_x, org_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)

        if frame_count % 30 == 0:
            tot_str = f"/{total_frames}" if total_frames else ""
            print(f"Processed {frame_count}{tot_str} frames...", end="\r")

    cap.release()
    out.release()
    print(f"\n[Done] Successfully saved slowed video with counter to: {output_video_path.name}")


if __name__ == "__main__":
    main()