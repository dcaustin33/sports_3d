import cv2
from pathlib import Path
from typing import Optional


def extract_frames(
    video_path: str,
    start_time: float,
    output_dir: str = "extracted_frames",
    duration: float = 10.0,
) -> list[Path]:
    """
    Extract all frames from a video file for a specific time slice.

    Args:
        video_path: Path to the input MP4 file
        start_time: Starting time in seconds
        output_dir: Directory to save extracted frames
        duration: Duration in seconds to extract (default: 10.0)

    Returns:
        List of paths to extracted frame files
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}")

    if start_time >= video_duration:
        raise ValueError(
            f"start_time ({start_time}s) exceeds video duration ({video_duration:.2f}s)"
        )

    start_frame = int(start_time * fps)
    end_time = min(start_time + duration, video_duration)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted_paths = []
    frame_count = start_frame

    while frame_count < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_count / fps
        frame_filename = f"frame_{frame_count:06d}_t{timestamp:.3f}s.png"
        frame_path = output_path / frame_filename

        cv2.imwrite(str(frame_path), frame)
        extracted_paths.append(frame_path)

        frame_count += 1

    cap.release()

    return extracted_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <video_path> <start_time> [output_dir] [duration]")
        print("Example: python extract_frames.py video.mp4 5.0 frames 10.0")
        sys.exit(1)

    video_path = sys.argv[1]
    start_time = float(sys.argv[2])
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "extracted_frames"
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

    frames = extract_frames(video_path, start_time, output_dir, duration)
    print(f"Extracted {len(frames)} frames to {output_dir}/")
    print(f"Time range: {start_time}s - {start_time + duration}s")
