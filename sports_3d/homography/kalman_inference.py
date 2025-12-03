"""
Trajectory filtering script for 3D trajectory smoothing.

Applies physics-aware filtering to existing trajectory JSON files:
- Z-axis: Quadratic fitting with velocity decay constraint
- X/Y-axes: Savitzky-Golay polynomial smoothing
- Automatic discontinuity detection (bounces, hits)

Usage:
    .venv/bin/python -m sports_3d.homography.kalman_inference \\
        data/sinner_ruud_trajectory \\
        data/sinner_ruud_events \\
        --verbose --backup
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from sports_3d.homography.tennis import (
    build_intrinsic_matrix,
    pixel_to_court_plane_point,
    pixel_to_court_plane_depth,
    rvec_tvec_to_extrinsic,
)
from sports_3d.utils.kalman import TrajectoryFilter, reprojection_to_3d_uncertainty


def discover_event_files(events_dir: Path) -> List[Path]:
    """
    Discover and sort event annotation files.

    Args:
        events_dir: Directory containing event annotation text files

    Returns:
        List of Path objects sorted by frame number

    Raises:
        ValueError: If no event files found or directory doesn't exist
    """
    events_dir = Path(events_dir)

    if not events_dir.exists():
        raise ValueError(f"Events directory does not exist: {events_dir}")

    txt_files = list(events_dir.glob("*_events.txt"))

    if not txt_files:
        raise ValueError(f"No event files (*_events.txt) found in {events_dir}")

    frame_pattern = re.compile(r'frame_(\d+)_t[\d.]+s_events\.txt')

    files_with_numbers = []
    for txt_file in txt_files:
        match = frame_pattern.search(txt_file.name)
        if match:
            frame_num = int(match.group(1))
            files_with_numbers.append((frame_num, txt_file))
        else:
            print(f"Warning: Skipping file with unexpected name: {txt_file.name}")

    files_with_numbers.sort(key=lambda x: x[0])
    sorted_files = [f[1] for f in files_with_numbers]

    return sorted_files


def parse_event_file(event_path: Path) -> Dict | None:
    """
    Parse a single event annotation file.

    Args:
        event_path: Path to event annotation file

    Returns:
        Dictionary with event data, or None if file is empty/malformed
            - For ground: {'type': 'ground', 'pixel': (x, y)}
            - For racquet: {'type': 'racquet', 'ball_pixel': (x1, y1), 'player_pixel': (x2, y2)}
    """
    try:
        with open(event_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return None

        parts = lines[0].split()
        if len(parts) < 3:
            print(f"Warning: Invalid event format in {event_path.name}")
            return None

        event_type = parts[0]

        if event_type == 'ground' and len(parts) == 3:
            x = float(parts[1])
            y = float(parts[2])
            return {'type': 'ground', 'pixel': (x, y)}

        elif event_type == 'racquet' and len(parts) == 5:
            ball_x = float(parts[1])
            ball_y = float(parts[2])
            player_x = float(parts[3])
            player_y = float(parts[4])
            return {
                'type': 'racquet',
                'ball_pixel': (ball_x, ball_y),
                'player_pixel': (player_x, player_y)
            }

        else:
            print(f"Warning: Unknown event type or invalid format in {event_path.name}")
            return None

    except Exception as e:
        print(f"Error parsing {event_path.name}: {e}")
        return None


def load_events(events_dir: Path, json_files: List[Path]) -> Dict[int, Dict]:
    """
    Load and align event annotations with trajectory frames.

    Args:
        events_dir: Directory containing event annotation files
        json_files: List of trajectory JSON file paths

    Returns:
        Dictionary mapping frame index to event data: {frame_idx: event_data}

    Raises:
        ValueError: If event exists but corresponding trajectory frame is missing
    """
    event_files = discover_event_files(events_dir)

    frame_pattern = re.compile(r'frame_(\d+)_t[\d.]+s')
    trajectory_frame_numbers = {}
    for idx, json_file in enumerate(json_files):
        match = frame_pattern.search(json_file.name)
        if match:
            frame_num = int(match.group(1))
            trajectory_frame_numbers[frame_num] = idx

    events = {}
    for event_file in event_files:
        match = frame_pattern.search(event_file.name)
        if not match:
            continue

        frame_num = int(match.group(1))
        

        if frame_num not in trajectory_frame_numbers:
            raise ValueError(
                f"Event annotation exists for frame {frame_num} but no corresponding "
                f"trajectory file found. Event file: {event_file.name}"
            )

        event_data = parse_event_file(event_file)
        if event_data is not None:
            frame_idx = trajectory_frame_numbers[frame_num]
            events[frame_idx] = event_data

    return events


def refine_position_hybrid(
    ball_pixel: Tuple[float, float],
    player_pixel: Tuple[float, float] | None,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    event_type: str
) -> np.ndarray | None:
    """
    Refine 3D position using hybrid approach.

    For racquet strikes: Uses ball 2D position + player z-depth (from court projection)
    For ground bounces: Full court plane projection (y=0 constraint)

    Args:
        ball_pixel: 2D pixel coordinates of ball (x, y)
        player_pixel: 2D pixel coordinates of player base (for z-depth), None for ground events
        intrinsic_matrix: 3x3 camera intrinsic matrix K
        extrinsic_matrix: 3x4 camera extrinsic matrix [R | t]
        event_type: 'racquet' or 'ground'

    Returns:
        Refined 3D position [x, y, z], or None if projection fails
    """
    ball_x, ball_y = ball_pixel
    # if player_pixel[0] == 1217:
    #     import pdb; pdb.set_trace()

    try:
        if event_type == 'ground':
            P_world = pixel_to_court_plane_point(
                ball_x, ball_y, intrinsic_matrix, extrinsic_matrix
            )
            return P_world

        elif event_type == 'racquet':
            if player_pixel is None:
                raise ValueError("Player pixel position required for racquet events")

            player_x, player_y = player_pixel
            z_world = pixel_to_court_plane_depth(
                player_x, player_y, intrinsic_matrix, extrinsic_matrix
            )

            f = intrinsic_matrix[0, 0]
            cx_img = intrinsic_matrix[0, 2]
            cy_img = intrinsic_matrix[1, 2]

            R = extrinsic_matrix[:, :3]
            t = extrinsic_matrix[:, 3]
            camera_pos = -R.T @ t
            R_world = R.T

            ray_cam = np.array([
                (ball_x - cx_img) / f,
                (ball_y - cy_img) / f,
                1.0
            ])
            ray_world = R_world @ ray_cam

            lambda_param = (z_world - camera_pos[2]) / ray_world[2]
            P_world = camera_pos + lambda_param * ray_world

            return P_world

        else:
            print(f"Warning: Unknown event type '{event_type}'")
            return None

    except ValueError as e:
        print(f"Warning: Failed to refine position for {event_type} event: {e}")
        return None


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
    verbose: bool,
    events: Dict[int, Dict],
    refined_positions: Dict[int, np.ndarray]
) -> Dict:
    """
    Apply trajectory filter to trajectory data.

    Args:
        timestamps: (N,) array of timestamps in seconds
        positions: (N, 3) array of positions in meters
        uncertainties: (N,) array of uncertainties in meters
        filter_params: Dictionary of filter parameters
        verbose: Enable detailed logging
        events: Dictionary mapping frame indices to event data
        refined_positions: Dictionary mapping frame indices to refined 3D positions

    Returns:
        Filter result dictionary with filtered positions and velocities
    """
    positions = positions.copy()
    for frame_idx, refined_pos in refined_positions.items():
        if 0 <= frame_idx < len(positions):
            positions[frame_idx] = refined_pos

    trajectory_filter = TrajectoryFilter(
        window_size_xy=filter_params['window_size_xy'],
        poly_order=filter_params['poly_order'],
        verbose=verbose
    )

    result = trajectory_filter.filter(timestamps, positions, uncertainties, event_dict=events)

    return result


def create_filtered_projections_entry(
    frame_idx: int,
    filter_result: Dict,
    events: Dict[int, Dict],
    refined_positions: Dict[int, np.ndarray]
) -> Dict:
    """
    Create filtered_projections entry for a single frame.

    Args:
        frame_idx: Index of the frame in the trajectory sequence
        filter_result: Dictionary returned by TrajectoryFilter.filter()
        events: Dictionary mapping frame indices to event data
        refined_positions: Dictionary mapping frame indices to refined positions

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

    event_metadata = None
    if frame_idx in events:
        event_data = events[frame_idx]
        if event_data['type'] == 'ground':
            event_metadata = {
                "event_type": "ground",
                "pixel_coords": list(event_data['pixel'])
            }
        elif event_data['type'] == 'racquet':
            event_metadata = {
                "event_type": "racquet",
                "ball_pixel": list(event_data['ball_pixel']),
                "player_pixel": list(event_data['player_pixel'])
            }

    position_refined_m = None
    if frame_idx in refined_positions:
        refined_pos = refined_positions[frame_idx]
        position_refined_m = {
            "x": float(refined_pos[0]),
            "y": float(refined_pos[1]),
            "z": float(refined_pos[2])
        }

    return {
        "position_filtered_m": {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        },
        "position_filtered_array_m": [float(pos[0]), float(pos[1]), float(pos[2])],
        "position_refined_m": position_refined_m,
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
        },
        "event_metadata": event_metadata
    }


def update_json_files(
    json_files: List[Path],
    json_dicts: List[dict],
    filter_result: Dict,
    backup: bool,
    events: Dict[int, Dict],
    refined_positions: Dict[int, np.ndarray]
) -> None:
    """
    Update JSON files with filtered projections.

    Args:
        json_files: List of JSON file paths
        json_dicts: List of original JSON dictionaries
        filter_result: Filter result dictionary
        backup: Create backup files before overwriting
        events: Dictionary mapping frame indices to event data
        refined_positions: Dictionary mapping frame indices to refined positions
    """
    for i, (json_file, json_dict) in tqdm(enumerate(zip(json_files, json_dicts))):
        filtered_entry = create_filtered_projections_entry(
            i, filter_result, events, refined_positions
        )
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
        description="Apply trajectory filtering to trajectory JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory \\
      data/sinner_ruud_events

  # With backup and verbose output
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory \\
      data/sinner_ruud_events \\
      --backup --verbose

  # Custom filter parameters
  .venv/bin/python -m sports_3d.homography.kalman_inference \\
      data/sinner_ruud_trajectory \\
      data/sinner_ruud_events \\
      --window_size_xy 9 --poly_order 3 --verbose
        """
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing trajectory JSON files"
    )

    parser.add_argument(
        "events_dir",
        type=Path,
        help="Directory containing event annotation files (*_events.txt)"
    )

    filter_group = parser.add_argument_group("filter parameters")
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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
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

    print("\nLoading event annotations...")
    events = load_events(args.events_dir, json_files)
    print(f"Loaded {len(events)} event annotations")

    print("Refining positions using event annotations...")
    refined_positions = {}
    for frame_idx, event_data in events.items():
        json_dict = json_dicts[frame_idx]
        K = build_intrinsic_matrix(
            json_dict['camera_calibration']['focal_length_px'],
            json_dict['camera_calibration']['principal_point_px']
        )
        extrinsic = rvec_tvec_to_extrinsic(
            np.array(json_dict['camera_calibration']['rotation_vector']),
            np.array(json_dict['camera_calibration']['translation_vector'])
        )

        if event_data['type'] == 'ground':
            ball_pixel = event_data['pixel']
            player_pixel = None
        else:
            ball_pixel = event_data['ball_pixel']
            player_pixel = event_data['player_pixel']

        refined_pos = refine_position_hybrid(
            ball_pixel,
            player_pixel,
            K,
            extrinsic,
            event_data['type']
        )

        if refined_pos is not None:
            refined_positions[frame_idx] = refined_pos

            if args.verbose:
                original_pos = positions[frame_idx]
                print(f"  Frame {frame_idx} ({event_data['type']}): "
                      f"original_z={original_pos[2]:.3f}m → refined_z={refined_pos[2]:.3f}m")


    print("\nApplying trajectory filter...")
    filter_params = {
        'window_size_xy': args.window_size_xy,
        'poly_order': args.poly_order
    }

    filter_result = apply_kalman_filter(
        timestamps, positions, uncertainties,
        filter_params, args.verbose,
        events=events,
        refined_positions=refined_positions
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

    backup_msg = " (with backup)" if args.backup else ""
    print(f"\nUpdating JSON files{backup_msg}...")

    update_json_files(
        json_files, json_dicts, filter_result,
        args.backup, events, refined_positions
    )

    print(f"\nComplete! Updated {len(json_files)} files.")
    if args.backup:
        print(f"Backups saved as *.json.bak")


if __name__ == "__main__":
    main()
