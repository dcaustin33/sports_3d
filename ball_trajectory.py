#!/usr/bin/env python3
"""
Ball Trajectory Tracking in 3D World Coordinates

This script performs camera calibration from tennis court keypoints,
then projects all ball bounding boxes to 3D world coordinates to
generate a complete trajectory.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from sports_3d.trajectory import (
    calibrate_camera_from_keypoints,
    load_bboxes_from_directory,
    project_bbox_to_world,
    save_trajectory_csv,
    plot_trajectory_3d
)
from sports_3d.trajectory.visualization import print_trajectory_statistics
from sports_3d.trajectory.data_loader import filter_valid_bboxes


three_d_keypoints = [
    (-10.97 / 2, 0, 23.77 / 2),
    (10.97 / 2, 0, 23.77 / 2),
    (-10.97 / 2, 0, -23.77 / 2),
    (10.97 / 2, 0, -23.77 / 2),
    (-8.23 / 2, 0, 23.77 / 2),
    (-8.23 / 2, 0, -23.77 / 2),
    (8.23 / 2, 0, 23.77 / 2),
    (8.23 / 2, 0, -23.77 / 2),
    (-8.23 / 2, 0, 6.4),
    (8.23 / 2, 0, 6.4),
    (-8.23 / 2, 0, -6.4),
    (8.23 / 2, 0, -6.4),
    (0, 0, 6.4),
    (0, 0, -6.4),
    (0, 0.914, 0)
]


def load_calibration_keypoints(keypoints_path: str):
    """
    Load keypoints from text file.

    Format: Each line contains "index x y"
    where index corresponds to three_d_keypoints array.

    Args:
        keypoints_path: Path to keypoints text file

    Returns:
        keypoints_2d: Nx2 array of image coordinates
        keypoints_3d: Nx3 array of world coordinates
    """
    keypoints_2d = []
    keypoints_3d = []

    with open(keypoints_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                print(f"Warning: Skipping invalid keypoint line: {line}")
                continue

            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])

            if idx >= len(three_d_keypoints):
                print(f"Warning: Invalid keypoint index {idx}, max is {len(three_d_keypoints)-1}")
                continue

            keypoints_2d.append([x, y])
            keypoints_3d.append(three_d_keypoints[idx])

    return np.array(keypoints_2d, dtype=np.float32), np.array(keypoints_3d, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Track ball trajectory in 3D world coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  .venv/bin/python ball_trajectory.py \\
    --calibration-image data/frame_001.png \\
    --calibration-keypoints data/frame_001_keypoints.txt \\
    --bbox-dir data/ball_bboxes/

  # With custom output and visualization
  .venv/bin/python ball_trajectory.py \\
    --calibration-image data/frame_001.png \\
    --calibration-keypoints data/frame_001_keypoints.txt \\
    --bbox-dir data/ball_bboxes/ \\
    --output-trajectory trajectory.csv \\
    --output-plot trajectory_3d.png \\
    --visualize
        """
    )

    parser.add_argument(
        '--calibration-image',
        required=True,
        help='Path to reference image for camera calibration'
    )
    parser.add_argument(
        '--calibration-keypoints',
        required=True,
        help='Path to keypoints file (format: "index x y" per line)'
    )
    parser.add_argument(
        '--bbox-dir',
        required=True,
        help='Directory containing bounding box annotations (YOLO format)'
    )
    parser.add_argument(
        '--output-trajectory',
        default='trajectory_world.csv',
        help='Output CSV file for trajectory (default: trajectory_world.csv)'
    )
    parser.add_argument(
        '--output-plot',
        default=None,
        help='Optional output path for 3D visualization plot'
    )
    parser.add_argument(
        '--ball-diameter',
        type=float,
        default=0.066,
        help='Tennis ball diameter in meters (default: 0.066)'
    )
    parser.add_argument(
        '--focal-range-min',
        type=float,
        default=None,
        help='Minimum focal length for search (pixels)'
    )
    parser.add_argument(
        '--focal-range-max',
        type=float,
        default=None,
        help='Maximum focal length for search (pixels)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Display interactive 3D plot'
    )
    parser.add_argument(
        '--filter-invalid',
        action='store_true',
        help='Filter out suspicious bounding boxes'
    )
    parser.add_argument(
        '--file-pattern',
        default='*.txt',
        help='File pattern for bbox files (default: *.txt)'
    )

    args = parser.parse_args()

    print("="*70)
    print("BALL TRAJECTORY TRACKING - 3D WORLD COORDINATES")
    print("="*70)

    if not Path(args.calibration_image).exists():
        print(f"Error: Calibration image not found: {args.calibration_image}")
        sys.exit(1)

    if not Path(args.calibration_keypoints).exists():
        print(f"Error: Calibration keypoints not found: {args.calibration_keypoints}")
        sys.exit(1)

    if not Path(args.bbox_dir).exists():
        print(f"Error: Bbox directory not found: {args.bbox_dir}")
        sys.exit(1)

    print("\n[1/5] Loading calibration data...")
    try:
        keypoints_2d, keypoints_3d = load_calibration_keypoints(args.calibration_keypoints)
        print(f"  ✓ Loaded {len(keypoints_2d)} keypoint correspondences")
    except Exception as e:
        print(f"Error loading keypoints: {e}")
        sys.exit(1)

    image = cv2.imread(args.calibration_image)
    if image is None:
        print(f"Error: Could not read calibration image")
        sys.exit(1)

    image_shape = (image.shape[0], image.shape[1])
    print(f"  ✓ Image shape: {image_shape[0]}x{image_shape[1]}")

    print("\n[2/5] Calibrating camera (focal length search + PnP)...")
    focal_range = None
    if args.focal_range_min and args.focal_range_max:
        focal_range = (args.focal_range_min, args.focal_range_max)
        print(f"  Using custom focal range: [{focal_range[0]:.0f}, {focal_range[1]:.0f}] px")

    try:
        calibration = calibrate_camera_from_keypoints(
            keypoints_2d,
            keypoints_3d,
            image_shape,
            focal_range=focal_range
        )
    except Exception as e:
        print(f"Error during calibration: {e}")
        sys.exit(1)

    print(f"  ✓ Focal length: {calibration['focal_length']:.1f} px")
    print(f"  ✓ Reprojection error: {calibration['reprojection_error']:.2f} px")

    cam_pos = calibration['camera_position'].ravel()
    print(f"  ✓ Camera position (world): X={cam_pos[0]:.2f}, Y={cam_pos[1]:.2f}, Z={cam_pos[2]:.2f} m")

    horizontal_fov = 2 * np.degrees(np.arctan(image_shape[1] / (2 * calibration['focal_length'])))
    print(f"  ✓ Horizontal FOV: {horizontal_fov:.1f}°")

    if calibration['reprojection_error'] > 20:
        print("  ⚠ Warning: High reprojection error - calibration may be inaccurate")

    print(f"\n[3/5] Loading bounding boxes from {args.bbox_dir}...")
    try:
        bbox_frames = load_bboxes_from_directory(
            args.bbox_dir,
            file_pattern=args.file_pattern
        )
        total_bboxes = sum(len(frame['bboxes']) for frame in bbox_frames)
        print(f"  ✓ Loaded {len(bbox_frames)} frames with {total_bboxes} bounding boxes")
    except Exception as e:
        print(f"Error loading bboxes: {e}")
        sys.exit(1)

    print("\n[4/5] Projecting bounding boxes to 3D world coordinates...")
    trajectory_points = []
    skipped_invalid = 0
    skipped_errors = 0

    for frame_data in bbox_frames:
        for bbox in frame_data['bboxes']:
            if args.filter_invalid and not filter_valid_bboxes(bbox, image_shape=image_shape):
                skipped_invalid += 1
                continue

            try:
                world_point = project_bbox_to_world(
                    bbox,
                    image_shape,
                    calibration['camera_matrix'],
                    calibration['camera_position'],
                    calibration['R_world'],
                    ball_diameter_m=args.ball_diameter
                )

                trajectory_points.append({
                    'frame_id': frame_data['frame_id'],
                    'timestamp': frame_data.get('timestamp'),
                    'x': world_point['x'],
                    'y': world_point['y'],
                    'z': world_point['z'],
                    'depth': world_point['depth'],
                    'bbox_width_px': world_point['bbox_width_px'],
                    'bbox_height_px': world_point['bbox_height_px']
                })
            except Exception as e:
                print(f"  Warning: Failed to project bbox in frame {frame_data['frame_id']}: {e}")
                skipped_errors += 1
                continue

    print(f"  ✓ Successfully projected {len(trajectory_points)} points")
    if skipped_invalid > 0:
        print(f"  ⚠ Skipped {skipped_invalid} invalid bboxes")
    if skipped_errors > 0:
        print(f"  ⚠ Skipped {skipped_errors} bboxes due to errors")

    if not trajectory_points:
        print("Error: No valid trajectory points generated")
        sys.exit(1)

    print(f"\n[5/5] Saving results...")
    try:
        save_trajectory_csv(trajectory_points, args.output_trajectory)
        print(f"  ✓ Trajectory saved to: {args.output_trajectory}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        sys.exit(1)

    print_trajectory_statistics(trajectory_points)

    if args.output_plot or args.visualize:
        print("Generating 3D visualization...")
        try:
            plot_trajectory_3d(
                trajectory_points,
                calibration,
                output_path=args.output_plot,
                show=args.visualize
            )
        except Exception as e:
            print(f"Error creating plot: {e}")

    print("\n" + "="*70)
    print("TRAJECTORY TRACKING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
