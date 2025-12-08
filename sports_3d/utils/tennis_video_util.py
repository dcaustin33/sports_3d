"""
Tennis trajectory video visualization utility.

Creates MP4 videos from frame sequences with text overlays showing
raw and filtered 3D positions and velocities for review.

Usage:
    .venv/bin/python -m sports_3d.utils.tennis_video_util \\
        <frames_dir> <trajectory_dir> <output_path> \\
        [--fps FPS] [--overlay_position POSITION] [--verbose]
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class FrameData:
    """Data for a single frame with trajectory information."""
    frame_number: int
    timestamp: float
    image_path: Path
    raw_position: np.ndarray
    filtered_position: np.ndarray
    velocity: np.ndarray
    is_discontinuity: bool
    is_outlier: bool


def discover_video_data(frames_dir: Path, trajectory_dir: Path) -> List[FrameData]:
    """
    Discover and load frame data with trajectory information.

    Args:
        frames_dir: Directory containing frame images (*.png)
        trajectory_dir: Directory containing trajectory JSON files

    Returns:
        List of FrameData objects sorted by frame number

    Raises:
        ValueError: If directories don't exist or no matching data found
    """
    frames_dir = Path(frames_dir)
    trajectory_dir = Path(trajectory_dir)

    if not frames_dir.exists():
        raise ValueError(f"Frames directory does not exist: {frames_dir}")
    if not trajectory_dir.exists():
        raise ValueError(f"Trajectory directory does not exist: {trajectory_dir}")

    frame_pattern = re.compile(r'(frame_\d+_t[\d.]+s)\.png')
    json_pattern = re.compile(r'frame_(\d+)_t([\d.]+)s')

    frame_files = list(frames_dir.glob("frame_*_t*s.png"))

    if not frame_files:
        raise ValueError(f"No frame images found in {frames_dir}")

    frame_data_list = []

    for frame_file in frame_files:
        match = frame_pattern.match(frame_file.name)
        if not match:
            continue

        base_name = match.group(1)
        json_file = trajectory_dir / f"{base_name}_trajectory.json"

        if not json_file.exists():
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'filtered_projections' not in data:
                print(f"Warning: {json_file.name} missing filtered_projections field")
                print("Run kalman_inference.py first to add filtered data")
                continue

            num_match = json_pattern.search(base_name)
            if not num_match:
                continue

            frame_number = int(num_match.group(1))
            timestamp = float(num_match.group(2))

            raw_position = np.array(data['trajectory_3d']['position_world_array_m'])
            filtered_position = np.array(
                data['filtered_projections']['position_filtered_array_m']
            )
            velocity = np.array(
                data['filtered_projections']['velocity_array_m_per_s']
            )

            filter_meta = data['filtered_projections']['filter_metadata']
            is_discontinuity = filter_meta.get('is_discontinuity', False)
            is_outlier = filter_meta.get('is_outlier', False)

            frame_data_list.append(FrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                image_path=frame_file,
                raw_position=raw_position,
                filtered_position=filtered_position,
                velocity=velocity,
                is_discontinuity=is_discontinuity,
                is_outlier=is_outlier
            ))

        except KeyError as e:
            print(f"Warning: Missing field in {json_file.name}: {e}, skipping")
            continue
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {json_file.name}: {e}, skipping")
            continue

    frame_data_list.sort(key=lambda x: x.frame_number)

    return frame_data_list


def create_overlay_text(frame_data: FrameData, frame_idx: int, total_frames: int) -> List[str]:
    """
    Generate formatted text lines for frame overlay.

    Args:
        frame_data: Frame data with trajectory information
        frame_idx: Current frame index (0-based)
        total_frames: Total number of frames in sequence

    Returns:
        List of formatted text lines
    """
    lines = [
        f"Frame: {frame_data.frame_number}  Time: {frame_data.timestamp:.3f}s  ({frame_idx + 1}/{total_frames})",
        "",
        "Raw Position (m):",
        f"  X: {frame_data.raw_position[0]:7.3f}",
        f"  Y: {frame_data.raw_position[1]:7.3f}",
        f"  Z: {frame_data.raw_position[2]:7.3f}",
        "",
        "Filtered Position (m):",
        f"  X: {frame_data.filtered_position[0]:7.3f}",
        f"  Y: {frame_data.filtered_position[1]:7.3f}",
        f"  Z: {frame_data.filtered_position[2]:7.3f}",
        "",
        "Velocity (m/s):",
        f"  vX: {frame_data.velocity[0]:7.3f}",
        f"  vY: {frame_data.velocity[1]:7.3f}",
        f"  vZ: {frame_data.velocity[2]:7.3f}",
    ]

    if frame_data.is_discontinuity:
        lines.append("")
        lines.append("WARNING: DISCONTINUITY")
    if frame_data.is_outlier:
        lines.append("WARNING: OUTLIER")

    return lines


def draw_overlay(
    frame: np.ndarray,
    text_lines: List[str],
    position: str = "top_left"
) -> np.ndarray:
    """
    Draw text overlay on frame with semi-transparent background.

    Args:
        frame: Input frame image
        text_lines: List of text lines to display
        position: Overlay position (top_left, top_right, bottom_left, bottom_right)

    Returns:
        Frame with overlay drawn
    """
    overlay = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    padding = 10
    line_height = 30

    if not text_lines:
        return frame

    max_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in text_lines
    )
    box_height = len(text_lines) * line_height + 2 * padding
    box_width = max_width + 2 * padding

    if position == "top_left":
        x0, y0 = padding, padding
    elif position == "top_right":
        x0 = frame.shape[1] - box_width - padding
        y0 = padding
    elif position == "bottom_left":
        x0 = padding
        y0 = frame.shape[0] - box_height - padding
    elif position == "bottom_right":
        x0 = frame.shape[1] - box_width - padding
        y0 = frame.shape[0] - box_height - padding
    else:
        x0, y0 = padding, padding

    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_width, y0 + box_height),
        bg_color,
        -1
    )

    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y = y0 + padding + 20
    for line in text_lines:
        cv2.putText(
            frame,
            line,
            (x0 + padding, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        y += line_height

    return frame


def calculate_fps_from_timestamps(frame_data_list: List[FrameData]) -> float:
    """
    Calculate FPS from frame timestamps.

    Args:
        frame_data_list: List of frame data

    Returns:
        Estimated FPS (fallback to 30.0 if calculation fails)
    """
    if len(frame_data_list) < 2:
        return 30.0

    timestamps = np.array([f.timestamp for f in frame_data_list])
    diffs = np.diff(timestamps)

    valid_diffs = diffs[diffs > 0]
    if len(valid_diffs) == 0:
        return 30.0

    median_diff = np.median(valid_diffs)

    if median_diff <= 0:
        return 30.0

    return 1.0 / median_diff


def create_video(
    frame_data_list: List[FrameData],
    output_path: Path,
    fps: float,
    overlay_position: str,
    verbose: bool
) -> None:
    """
    Create MP4 video with trajectory overlays.

    Args:
        frame_data_list: List of frame data to process
        output_path: Output video file path
        fps: Frames per second
        overlay_position: Position of text overlay
        verbose: Enable detailed logging
    """
    if not frame_data_list:
        raise ValueError("No frames to process")

    first_frame = cv2.imread(str(frame_data_list[0].image_path))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_data_list[0].image_path}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        total_frames = len(frame_data_list)
        for i, frame_data in enumerate(frame_data_list):
            frame = cv2.imread(str(frame_data.image_path))
            if frame is None:
                print(f"Warning: Could not read {frame_data.image_path}, skipping")
                continue

            text_lines = create_overlay_text(frame_data, i, total_frames)
            frame_with_overlay = draw_overlay(frame, text_lines, overlay_position)

            writer.write(frame_with_overlay)

            if verbose or (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frame_data_list)} frames...")

    finally:
        writer.release()

    if verbose:
        duration = len(frame_data_list) / fps
        print(f"\nVideo created: {output_path}")
        print(f"  Frames: {len(frame_data_list)}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Resolution: {width}x{height}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create video from trajectory frames with coordinate overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect FPS from timestamps
  .venv/bin/python -m sports_3d.utils.tennis_video_util \\
      data/sinner_ruud_Frames \\
      data/sinner_ruud_trajectory \\
      output_video.mp4

  # Custom FPS
  .venv/bin/python -m sports_3d.utils.tennis_video_util \\
      data/sinner_ruud_Frames \\
      data/sinner_ruud_trajectory \\
      output_video.mp4 \\
      --fps 30

  # Verbose output with custom overlay position
  .venv/bin/python -m sports_3d.utils.tennis_video_util \\
      data/sinner_ruud_Frames \\
      data/sinner_ruud_trajectory \\
      output_video.mp4 \\
      --fps 60 --overlay_position top_right --verbose
        """
    )

    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Directory containing frame images (*.png)"
    )
    parser.add_argument(
        "trajectory_dir",
        type=Path,
        help="Directory containing trajectory JSON files"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output video path (e.g., output.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate (default: auto-detect from timestamps)"
    )
    parser.add_argument(
        "--overlay_position",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        default="top_left",
        help="Position of text overlay (default: top_left)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    return parser.parse_args()


def main():
    """Command-line entry point."""
    args = parse_arguments()

    print("Discovering frames and trajectory data...")
    frame_data_list = discover_video_data(args.frames_dir, args.trajectory_dir)
    print(f"Found {len(frame_data_list)} frames with trajectory data")

    if len(frame_data_list) == 0:
        print("Error: No matching frames and trajectory files found")
        return

    if args.fps is None:
        fps = calculate_fps_from_timestamps(frame_data_list)
        print(f"Auto-detected FPS: {fps:.2f}")
    else:
        fps = args.fps
        print(f"Using FPS: {fps}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating video: {args.output_path}")
    create_video(
        frame_data_list,
        args.output_path,
        fps,
        args.overlay_position,
        args.verbose
    )

    print(f"\nComplete! Video saved to: {args.output_path}")


if __name__ == "__main__":
    main()
