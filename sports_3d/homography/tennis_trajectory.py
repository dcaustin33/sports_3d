"""
Batch processing script to convert tennis ball bounding boxes to 3D world coordinates.

This module processes frame sequences with tennis ball bounding boxes and converts them
to 3D trajectory data using camera calibration from court keypoint annotations. When
keypoints are not available for a frame, it interpolates 2D pixel positions between
neighboring frames and performs fresh camera calibration.

Usage:
    .venv/bin/python -m sports_3d.homography.tennis_trajectory \\
        <bbox_dir> <keypoints_dir> <frames_dir> <output_dir> \\
        [--ball_diameter 0.066] [--verbose]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sports_3d.homography.tennis import (
    bbox_to_world_coordinates,
    build_intrinsic_matrix,
    estimate_focal_range,
    get_2d_and_3d_keypoints,
    get_camera_pose_in_world,
    rvec_tvec_to_extrinsic,
    solve_pnp_with_focal_search,
)
from sports_3d.utils.labeling_utils import read_yolo_box
from sports_3d.utils.file_utils import parse_frame_identifier, serialize_vector_3d


@dataclass
class FrameMetadata:
    """Metadata for a single frame including file paths and identifiers."""
    frame_base_name: str
    frame_number: int
    timestamp: float
    bbox_path: Path
    frame_path: Path
    keypoints_path: Optional[Path]


@dataclass
class CameraCalibration:
    """Complete camera calibration data for a frame."""
    frame_base_name: str
    keypoints_2d: np.ndarray
    keypoints_3d: np.ndarray
    focal_length: float
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error: float
    intrinsic_matrix: np.ndarray
    extrinsic_matrix: np.ndarray
    camera_position: np.ndarray
    keypoint_source: str


def extract_frame_base_name(filename: str) -> Optional[str]:
    """Extract frame base name from filename.

    Args:
        filename: Filename like 'frame_004200_t70.000s_bbox.txt'

    Returns:
        Base name like 'frame_004200_t70.000s' or None if not found
    """
    parsed = parse_frame_identifier(filename)
    return parsed[0] if parsed else None


def discover_frames(bbox_dir: Path, keypoints_dir: Path, frames_dir: Path) -> List[FrameMetadata]:
    """Discover all frames and build metadata.

    Args:
        bbox_dir: Directory containing bbox files
        keypoints_dir: Directory containing keypoint files
        frames_dir: Directory containing frame images

    Returns:
        List of FrameMetadata sorted by frame number
    """
    bbox_dir = Path(bbox_dir)
    keypoints_dir = Path(keypoints_dir)
    frames_dir = Path(frames_dir)

    bbox_files = sorted([
        f for f in bbox_dir.glob("*_bbox.txt")
        if "_refined" not in f.name
    ])

    if not bbox_files:
        raise ValueError(f"No bbox files found in {bbox_dir}")

    frames = []
    for bbox_file in bbox_files:
        base_name = extract_frame_base_name(bbox_file.name)
        if not base_name:
            continue

        refined_path = bbox_dir / f"{base_name}_bbox_refined.txt"
        bbox_path = refined_path if refined_path.exists() else bbox_file

        keypoints_path = keypoints_dir / f"{base_name}_keypoints.txt"
        keypoints_path = keypoints_path if keypoints_path.exists() else None

        frame_path = frames_dir / f"{base_name}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame image not found: {frame_path}")

        parsed = parse_frame_identifier(base_name)
        if not parsed:
            continue

        _, frame_num, timestamp = parsed

        frames.append(FrameMetadata(
            frame_base_name=base_name,
            frame_number=frame_num,
            timestamp=timestamp,
            bbox_path=bbox_path,
            frame_path=frame_path,
            keypoints_path=keypoints_path
        ))

    return sorted(frames, key=lambda f: f.frame_number)


def validate_inputs(frames: List[FrameMetadata]) -> None:
    """Validate that required input data exists.

    Args:
        frames: List of frame metadata

    Raises:
        ValueError: If no keypoints are available anywhere
    """
    if not frames:
        raise ValueError("No frames found")

    has_keypoints = any(f.keypoints_path is not None for f in frames)
    if not has_keypoints:
        raise ValueError("No keypoint files found. At least one frame must have keypoints.")


class KeypointInterpolator:
    """Manages keypoint lookup and 2D pixel interpolation across frames."""

    def __init__(self, frames: List[FrameMetadata], verbose: bool = False):
        """Initialize interpolator with frame metadata.

        Args:
            frames: List of frame metadata sorted by frame number
            verbose: Enable detailed logging
        """
        self.frames = frames
        self.verbose = verbose
        self.keypoint_cache = {}

        for i, frame in enumerate(frames):
            if frame.keypoints_path is not None:
                indices, keypoints_2d, keypoints_3d = get_2d_and_3d_keypoints(str(frame.keypoints_path))
                self.keypoint_cache[i] = (indices, keypoints_2d, keypoints_3d)
                if self.verbose:
                    print(f"Loaded keypoints for frame {frame.frame_number} (index {i})")

        if not self.keypoint_cache:
            raise ValueError("No keypoints loaded")

        if self.verbose:
            print(f"KeypointInterpolator initialized with {len(self.keypoint_cache)} keypoint frames")

    def _find_nearest_keypoint_frame(self, frame_idx: int, direction: str) -> Optional[int]:
        """Find nearest frame with keypoints in specified direction.

        Args:
            frame_idx: Current frame index
            direction: 'prev' or 'next'

        Returns:
            Frame index with keypoints, or None if not found
        """
        if direction == 'prev':
            for i in range(frame_idx - 1, -1, -1):
                if i in self.keypoint_cache:
                    return i
        elif direction == 'next':
            for i in range(frame_idx + 1, len(self.frames)):
                if i in self.keypoint_cache:
                    return i
        return None

    def _interpolate_keypoints(self, target_idx: int, prev_idx: int, next_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Linearly interpolate 2D pixel coordinates for matching keypoint indices.

        Args:
            target_idx: Target frame index
            prev_idx: Previous keypoint frame index
            next_idx: Next keypoint frame index

        Returns:
            Tuple of (indices, interpolated_2d, original_3d, source_label)
            Only keypoints present in both frames are returned.
        """
        prev_indices, prev_2d, prev_3d = self.keypoint_cache[prev_idx]
        next_indices, next_2d, next_3d = self.keypoint_cache[next_idx]

        # Find common keypoint indices
        common_indices = np.intersect1d(prev_indices, next_indices)

        if len(common_indices) == 0:
            raise ValueError(f"No common keypoints between frames {prev_idx} and {next_idx}")

        # Create index mappings
        prev_mask = np.isin(prev_indices, common_indices)
        next_mask = np.isin(next_indices, common_indices)

        # Get sorted positions for matching
        prev_argsort = np.argsort(prev_indices[prev_mask])
        next_argsort = np.argsort(next_indices[next_mask])

        # Extract matching keypoints in consistent order
        matched_prev_2d = prev_2d[prev_mask][prev_argsort]
        matched_next_2d = next_2d[next_mask][next_argsort]
        matched_prev_3d = prev_3d[prev_mask][prev_argsort]
        matched_next_3d = next_3d[next_mask][next_argsort]

        # Verify 3D coordinates match
        if not np.allclose(matched_prev_3d, matched_next_3d):
            raise ValueError("3D keypoint coordinates mismatch for common indices")

        target_frame_num = self.frames[target_idx].frame_number
        prev_frame_num = self.frames[prev_idx].frame_number
        next_frame_num = self.frames[next_idx].frame_number

        alpha = (target_frame_num - prev_frame_num) / (next_frame_num - prev_frame_num)
        interpolated_2d = (1 - alpha) * matched_prev_2d + alpha * matched_next_2d

        if self.verbose:
            print(f"Interpolated {len(common_indices)} keypoints for frame {target_frame_num} " +
                  f"(alpha={alpha:.3f} between frames {prev_frame_num} and {next_frame_num})")
            if len(common_indices) < len(prev_indices) or len(common_indices) < len(next_indices):
                print(f"  Dropped {len(prev_indices) - len(common_indices)} from prev, " +
                      f"{len(next_indices) - len(common_indices)} from next")

        return common_indices, interpolated_2d, matched_prev_3d, "interpolated"

    def get_keypoints_for_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get keypoints for a frame - direct or interpolated.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (keypoints_2d, keypoints_3d, source_label)
        """
        if frame_idx in self.keypoint_cache:
            indices, kp_2d, kp_3d = self.keypoint_cache[frame_idx]
            return kp_2d, kp_3d, "direct"

        prev_idx = self._find_nearest_keypoint_frame(frame_idx, 'prev')
        next_idx = self._find_nearest_keypoint_frame(frame_idx, 'next')

        if prev_idx is None and next_idx is None:
            raise ValueError(f"No keypoints available for frame {frame_idx}")

        if next_idx is None:
            indices, kp_2d, kp_3d = self.keypoint_cache[prev_idx]
            if self.verbose:
                print(f"Using previous keypoints from frame {self.frames[prev_idx].frame_number}")
            return kp_2d, kp_3d, "previous"

        if prev_idx is None:
            indices, kp_2d, kp_3d = self.keypoint_cache[next_idx]
            if self.verbose:
                print(f"Using next keypoints from frame {self.frames[next_idx].frame_number}")
            return kp_2d, kp_3d, "next"

        indices, kp_2d, kp_3d, source = self._interpolate_keypoints(frame_idx, prev_idx, next_idx)
        return kp_2d, kp_3d, source


def compute_frame_calibration(
    frame: FrameMetadata,
    frame_idx: int,
    interpolator: KeypointInterpolator,
    img_width: int,
    img_height: int,
    verbose: bool = False
) -> CameraCalibration:
    """Compute complete camera calibration for a frame.

    Args:
        frame: Frame metadata
        frame_idx: Frame index in the sequence
        interpolator: Keypoint interpolator instance
        img_width: Image width in pixels
        img_height: Image height in pixels
        verbose: Enable detailed logging

    Returns:
        Complete camera calibration data
    """
    keypoints_2d, keypoints_3d, keypoint_source = interpolator.get_keypoints_for_frame(frame_idx)

    min_f, max_f = estimate_focal_range(img_width, img_height)

    if verbose:
        print(f"Calibrating frame {frame.frame_number} " +
              f"(keypoints: {keypoint_source}, focal range: {min_f:.1f}-{max_f:.1f})")

    best_f, best_pose, best_error = solve_pnp_with_focal_search(
        keypoints_3d,
        keypoints_2d,
        focal_range=(min_f, max_f),
        principal_point=(img_width / 2, img_height / 2)
    )

    if best_pose is None:
        raise RuntimeError(f"Camera calibration failed for frame {frame.frame_number}")

    rvec, tvec = best_pose

    intrinsic_matrix = build_intrinsic_matrix(
        focal_length=best_f,
        image_width=img_width,
        image_height=img_height
    )

    extrinsic_matrix = rvec_tvec_to_extrinsic(rvec, tvec)
    camera_position, _ = get_camera_pose_in_world(rvec, tvec)

    if verbose:
        print(f"  Focal length: {best_f:.2f}px, Reprojection error: {best_error:.2f}px")

    return CameraCalibration(
        frame_base_name=frame.frame_base_name,
        keypoints_2d=keypoints_2d,
        keypoints_3d=keypoints_3d,
        focal_length=best_f,
        rvec=rvec,
        tvec=tvec,
        reprojection_error=best_error,
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
        camera_position=camera_position,
        keypoint_source=keypoint_source
    )


def create_trajectory_json(
    frame: FrameMetadata,
    calibration: CameraCalibration,
    bbox_yolo: list,
    world_position: np.ndarray,
    image_shape: tuple,
    ball_diameter: float
) -> dict:
    """Create JSON output structure for a frame.

    Args:
        frame: Frame metadata
        calibration: Camera calibration data
        bbox_yolo: YOLO format bbox [cx, cy, w, h] normalized
        world_position: 3D world coordinates
        image_shape: Image shape (height, width, channels)
        ball_diameter: Ball diameter in meters

    Returns:
        Dictionary ready for JSON serialization
    """
    img_height, img_width = image_shape[:2]

    cx_px = bbox_yolo[0] * img_width
    cy_px = bbox_yolo[1] * img_height
    w_px = bbox_yolo[2] * img_width
    h_px = bbox_yolo[3] * img_height

    distance = np.linalg.norm(world_position - calibration.camera_position)

    position_data = serialize_vector_3d(world_position, "position_world", "m")

    return {
        "metadata": {
            "frame_base_name": frame.frame_base_name,
            "frame_number": frame.frame_number,
            "timestamp_seconds": frame.timestamp,
            "image_path": str(frame.frame_path),
            "image_dimensions": {
                "width_px": img_width,
                "height_px": img_height
            }
        },
        "camera_calibration": {
            "focal_length_px": float(calibration.focal_length),
            "principal_point_px": [
                float(calibration.intrinsic_matrix[0, 2]),
                float(calibration.intrinsic_matrix[1, 2])
            ],
            "reprojection_error_px": float(calibration.reprojection_error),
            "camera_position_world_m": calibration.camera_position.flatten().tolist(),
            "rotation_vector": calibration.rvec.flatten().tolist(),
            "translation_vector": calibration.tvec.flatten().tolist(),
            "keypoint_source": calibration.keypoint_source
        },
        "ball_detection": {
            "bbox_yolo_normalized": bbox_yolo,
            "bbox_pixel_coords": {
                "center_x_px": float(cx_px),
                "center_y_px": float(cy_px),
                "width_px": float(w_px),
                "height_px": float(h_px)
            },
            "ball_diameter_m": ball_diameter
        },
        "trajectory_3d": {
            **position_data,
            "distance_from_camera_m": float(distance)
        }
    }


def process_trajectory(
    bbox_dir: Path,
    keypoints_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    ball_diameter: float = 0.066,
    verbose: bool = False
) -> None:
    """Process all frames and generate trajectory JSON files.

    Args:
        bbox_dir: Directory containing bbox files
        keypoints_dir: Directory containing keypoint files
        frames_dir: Directory containing frame images
        output_dir: Output directory for JSON files
        ball_diameter: Tennis ball diameter in meters
        verbose: Enable detailed logging
    """
    print(f"Discovering frames...")
    frames = discover_frames(bbox_dir, keypoints_dir, frames_dir)
    print(f"Found {len(frames)} frames")

    validate_inputs(frames)

    interpolator = KeypointInterpolator(frames, verbose=verbose)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nProcessing frames...")
    for i, frame in enumerate(frames):
        if verbose or (i % 10 == 0):
            print(f"[{i+1}/{len(frames)}] Processing {frame.frame_base_name}")

        bbox_yolo = read_yolo_box(str(frame.bbox_path))[0]

        image = cv2.imread(str(frame.frame_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {frame.frame_path}")

        img_height, img_width = image.shape[:2]

        calibration = compute_frame_calibration(
            frame, i, interpolator, img_width, img_height, verbose
        )

        world_position = bbox_to_world_coordinates(
            bbox_yolo=tuple(bbox_yolo),
            intrinsic_matrix=calibration.intrinsic_matrix,
            extrinsic_matrix=calibration.extrinsic_matrix,
            image_width=img_width,
            image_height=img_height,
            object_width_m=ball_diameter
        )

        output_data = create_trajectory_json(
            frame, calibration, bbox_yolo, world_position, image.shape, ball_diameter
        )

        output_path = output_dir / f"{frame.frame_base_name}_trajectory.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    print(f"\nComplete! Processed {len(frames)} frames")
    print(f"Output files: {output_dir}/*_trajectory.json")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Convert tennis ball bounding boxes to 3D world coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  .venv/bin/python -m sports_3d.homography.tennis_trajectory \\
      data/sinner_ruud_bbox \\
      data/sinner_ruud_keypoints \\
      data/sinner_ruud_Frames \\
      data/sinner_ruud_trajectory \\
      --verbose
        """
    )

    parser.add_argument("bbox_dir", type=Path,
                       help="Directory containing YOLO bbox files")
    parser.add_argument("keypoints_dir", type=Path,
                       help="Directory containing keypoint files")
    parser.add_argument("frames_dir", type=Path,
                       help="Directory containing frame images")
    parser.add_argument("output_dir", type=Path,
                       help="Output directory for trajectory JSON files")
    parser.add_argument("--ball_diameter", type=float, default=0.066,
                       help="Tennis ball diameter in meters (default: 0.066)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable detailed logging")

    args = parser.parse_args()

    process_trajectory(
        bbox_dir=args.bbox_dir,
        keypoints_dir=args.keypoints_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        ball_diameter=args.ball_diameter,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
