from pathlib import Path
import cv2


def main():
    # Set paths relative to the script location (inside Scripts/)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent  # BeadDetection root folder

    # video_path = root_dir / "video1Copy.mp4"
    video_path = root_dir / "video2TrimmedCopy.mp4"
    output_dir = root_dir / "video2Trim"

    if not video_path.exists():
        print(f"[Error] Could not find video at: {video_path}")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output folder ready: {output_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Processing video at {fps} FPS...")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Save frame as frame_0001.png, frame_0002.png, etc.
        frame_name = f"frame_{frame_count:04d}.png"
        frame_file_path = output_dir / frame_name

        cv2.imwrite(str(frame_file_path), frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Extracted {frame_count} frames...", end="\r")

    cap.release()
    print(f"\n[Done] Successfully extracted {frame_count} frames into '{output_dir.name}/'.")


if __name__ == "__main__":
    main()