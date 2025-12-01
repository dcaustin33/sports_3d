"""
Kalman filter inference script for 3D trajectory smoothing.

Applies physics-aware filtering to existing trajectory JSON files:
- Z-axis: Kalman filter with gravity model
- X/Y-axes: Savitzky-Golay polynomial smoothing
- Automatic discontinuity detection (bounces, hits)

Usage:
    .venv/bin/python -m sports_3d.homography.kalman_inference \\
        data/sinner_ruud_trajectory \\
        --verbose --backup
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from sports_3d.utils.kalman import TrajectoryFilter, reprojection_to_3d_uncertainty


def discover_trajectory_files(input_dir: Path) -> List[Path]:
    """
    Discover and sort trajectory JSON files.

    Args:
        input_dir: Directory containing trajectory JSON files

    Returns:
        List of Path objects sorted by frame number

    Raises:
        ValueError: If no trajectory files found or directory doesn't exist
    """
    input_dir = Path(input_dir)

    json_files = list(input_dir.glob("*_trajectory.json"))

    if not json_files:
        raise ValueError(f"No trajectory JSON files found in {input_dir}")

    frame_pattern = re.compile(r'frame_(\d+)_t[\d.]+s_trajectory\.json')

    files_with_numbers = []
    for json_file in json_files:
        match = frame_pattern.search(json_file.name)
        if match:
            frame_num = int(match.group(1))
            files_with_numbers.append((frame_num, json_file))
        else:
            print(f"Warning: Skipping file with unexpected name: {json_file.name}")

    files_with_numbers.sort(key=lambda x: x[0])
    sorted_files = [f[1] for f in files_with_numbers]

    return sorted_files


def extract_trajectory_data(
    json_files: List[Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Extract trajectory data from JSON files.

    Args:
        json_files: List of trajectory JSON file paths

    Returns:
        Tuple of (timestamps, positions, uncertainties, json_dicts)
            - timestamps: (N,) array in seconds
            - positions: (N, 3) array [x, y, z] in meters
            - uncertainties: (N,) array in meters
            - json_dicts: List of original JSON dictionaries

    Raises:
        ValueError: If required fields are missing
    """
    timestamps = []
    positions = []
    uncertainties = []
    json_dicts = []

    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            timestamp = data['metadata']['timestamp_seconds']
            position = data['trajectory_3d']['position_world_array_m']

            reprojection_error = data['camera_calibration']['reprojection_error_px']
            focal_length = data['camera_calibration']['focal_length_px']
            distance = data['trajectory_3d']['distance_from_camera_m']

            uncertainty = reprojection_to_3d_uncertainty(
                reprojection_error_px=reprojection_error,
                focal_length_px=focal_length,
                distance_m=distance
            )

            timestamps.append(timestamp)
            positions.append(position)
            uncertainties.append(uncertainty)
            json_dicts.append(data)

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(json_files)} files...")

        except KeyError as e:
            raise ValueError(f"Missing required field in {json_file.name}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_file.name}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing {json_file.name}: {e}")

    return (
        np.array(timestamps),
        np.array(positions),
        np.array(uncertainties),
        json_dicts
    )


def apply_kalman_filter(
    timestamps: np.ndarray,
    positions: np.ndarray,
    uncertainties: np.ndarray,
    filter_params: Dict,
    verbose: bool
) -> Dict:
    """
    Apply Kalman filter to trajectory data.

    Args:
        timestamps: (N,) array of timestamps in seconds
        positions: (N, 3) array of positions in meters
        uncertainties: (N,) array of uncertainties in meters
        filter_params: Dictionary of filter parameters
        verbose: Enable detailed logging

    Returns:
        Filter result dictionary with filtered positions and velocities
    """
    trajectory_filter = TrajectoryFilter(
        gravity=filter_params['gravity'],
        process_noise_z=filter_params['process_noise_z'],
        window_size_xy=filter_params['window_size_xy'],
        poly_order=filter_params['poly_order'],
        accel_threshold_z=filter_params['accel_threshold_z'],
        accel_threshold_y=filter_params['accel_threshold_y'],
        verbose=verbose
    )

    result = trajectory_filter.filter(timestamps, positions, uncertainties)

    return result


def create_filtered_projections_entry(frame_idx: int, filter_result: Dict) -> Dict:
    """
    Create filtered_projections entry for a single frame.

    Args:
        frame_idx: Index of the frame in the trajectory sequence
        filter_result: Dictionary returned by TrajectoryFilter.filter()

    Returns:
        Dictionary with filtered position, velocity, and metadata
    """
    pos = filter_result['positions_filtered'][frame_idx]
    vel = filter_result['velocities'][frame_idx]

    is_discontinuity = frame_idx in filter_result['discontinuity_frames']
    is_outlier = frame_idx in filter_result['outlier_frames']

    segment_index = -1
    segment_bounds = None
    for seg_idx, (start, end) in enumerate(filter_result['segments']):
        if start <= frame_idx < end:
            segment_index = seg_idx
            segment_bounds = [int(start), int(end)]
            break

    return {
        "position_filtered_m": {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        },
        "position_filtered_array_m": [float(pos[0]), float(pos[1]), float(pos[2])],
        "velocity_m_per_s": {
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2])
        },
        "velocity_array_m_per_s": [float(vel[0]), float(vel[1]), float(vel[2])],
        "filter_metadata": {
            "is_discontinuity": bool(is_discontinuity),
            "is_outlier": bool(is_outlier),
            "segment_index": segment_index,
            "segment_bounds": segment_bounds
        }
    }


def update_json_files(
    json_files: List[Path],
    json_dicts: List[dict],
    filter_result: Dict,
    backup: bool,
    dry_run: bool
) -> None:
    """
    Update JSON files with filtered projections.

    Args:
        json_files: List of JSON file paths
        json_dicts: List of original JSON dictionaries
        filter_result: Filter result dictionary
        backup: Create backup files before overwriting
        dry_run: Preview changes without writing
    """
    for i, (json_file, json_dict) in tqdm(enumerate(zip(json_files, json_dicts))):
        filtered_entry = create_filtered_projections_entry(i, filter_result)
        json_dict['filtered_projections'] = filtered_entry

        if backup:
            backup_path = json_file.with_suffix('.json.bak')
            try:
                with open(json_file, 'r') as f:
                    backup_content = f.read()
                with open(backup_path, 'w') as f:
                    f.write(backup_content)
            except Exception as e:
                print(f"Warning: Failed to create backup for {json_file.name}: {e}")

        try:
            temp_path = json_file.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(json_dict, f, indent=2)
            temp_path.replace(json_file)
        except Exception as e:
            print(f"Error: Failed to write {json_file.name}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply Kalman filtering to trajectory JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory

  # With backup and verbose output
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory --backup --verbose

  # Dry run to preview changes
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory --dry_run --verbose

  # Custom filter parameters
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory \\
      --gravity -9.81 --window_size_xy 9 --poly_order 3 --verbose
        """
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing trajectory JSON files"
    )

    filter_group = parser.add_argument_group("filter parameters")
    filter_group.add_argument(
        "--gravity",
        type=float,
        default=-9.81,
        help="Gravity acceleration in m/s² (default: -9.81)"
    )
    filter_group.add_argument(
        "--process_noise_z",
        type=float,
        default=1.0,
        help="Process noise for Z-axis Kalman filter (default: 1.0)"
    )
    filter_group.add_argument(
        "--window_size_xy",
        type=int,
        default=7,
        help="Smoothing window size for X/Y axes (default: 7)"
    )
    filter_group.add_argument(
        "--poly_order",
        type=int,
        default=2,
        help="Polynomial order for X/Y smoothing (default: 2)"
    )
    filter_group.add_argument(
        "--accel_threshold_z",
        type=float,
        default=200.0,
        help="Z-axis acceleration threshold for discontinuity detection in m/s² (default: 200.0)"
    )
    filter_group.add_argument(
        "--accel_threshold_y",
        type=float,
        default=150.0,
        help="Y-axis acceleration threshold for discontinuity detection in m/s² (default: 150.0)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak files before overwriting originals"
    )

    return parser.parse_args()


def main():
    """Command-line entry point."""
    args = parse_arguments()

    print("Discovering trajectory files...")
    json_files = discover_trajectory_files(args.input_dir)
    print(f"Found {len(json_files)} trajectory files")

    if len(json_files) < 3:
        raise ValueError("Need at least 3 trajectory files for filtering")

    print("\nExtracting trajectory data...")
    timestamps, positions, uncertainties, json_dicts = extract_trajectory_data(json_files)
    print(f"Loaded {len(timestamps)} frames")

    print("\nApplying Kalman filter...")
    filter_params = {
        'gravity': args.gravity,
        'process_noise_z': args.process_noise_z,
        'window_size_xy': args.window_size_xy,
        'poly_order': args.poly_order,
        'accel_threshold_z': args.accel_threshold_z,
        'accel_threshold_y': args.accel_threshold_y
    }

    filter_result = apply_kalman_filter(
        timestamps, positions, uncertainties,
        filter_params, args.verbose
    )

    print(f"\nFilter Results:")
    print(f"  Discontinuities detected: {len(filter_result['discontinuity_frames'])}")
    if len(filter_result['discontinuity_frames']) > 0:
        print(f"    At frames: {filter_result['discontinuity_frames']}")
    print(f"  Outliers rejected: {len(filter_result['outlier_frames'])}")
    if len(filter_result['outlier_frames']) > 0:
        print(f"    At frames: {filter_result['outlier_frames']}")
    print(f"  Trajectory segments: {len(filter_result['segments'])}")
    for i, (start, end) in enumerate(filter_result['segments']):
        print(f"    Segment {i}: frames {start}-{end-1} ({end-start} frames)")

    mode = "DRY RUN" if args.dry_run else "Writing"
    backup_msg = " (with backup)" if args.backup else ""
    print(f"\n{mode}: Updating JSON files{backup_msg}...")

    update_json_files(
        json_files, json_dicts, filter_result,
        args.backup, args.dry_run
    )

    if args.dry_run:
        print("\nDry run complete! No files were modified.")
        print("Remove --dry_run flag to apply changes.")
    else:
        print(f"\nComplete! Updated {len(json_files)} files.")
        if args.backup:
            print(f"Backups saved as *.json.bak")


if __name__ == "__main__":
    main()
