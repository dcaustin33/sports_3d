"""
Trajectory Data Loading for Blender

Loads tennis ball trajectory data from JSON files and processes it
for use in Blender animations.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional


def parse_frame_identifier(filename: str) -> Tuple[str, int, float]:
    """Extract frame info from filename.

    Args:
        filename: Filename like 'frame_004140_t69.000s_trajectory.json'

    Returns:
        Tuple of (base_name, frame_number, timestamp_seconds)
    """
    base_match = re.search(r'(frame_\d+_t[\d.]+s)', filename)
    frame_match = re.search(r'frame_(\d+)', filename)
    time_match = re.search(r't([\d.]+)s', filename)

    base = base_match.group(1) if base_match else filename
    frame_num = int(frame_match.group(1)) if frame_match else 0
    timestamp = float(time_match.group(1)) if time_match else 0.0

    return base, frame_num, timestamp


def load_trajectory_data(trajectory_dir: str) -> Tuple[List[dict], List[Tuple[float, float, float]]]:
    """Load trajectory JSONs and extract positions.

    Loads all trajectory JSON files from the specified directory,
    extracts the filtered positions, and negates Y values (since
    the calibration system produces negative Y values).

    Args:
        trajectory_dir: Path to directory containing trajectory JSON files

    Returns:
        Tuple of (metadata_list, positions_list)
        - metadata_list: List of dicts with frame_number, timestamp, file_path
        - positions_list: List of (x, y, z) tuples with corrected Y values
    """
    traj_path = Path(trajectory_dir)
    files = sorted(traj_path.glob("*_trajectory.json"))

    # Filter out backup files
    files = [f for f in files if not str(f).endswith('.bak')]

    metadata_list = []
    positions = []

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        _, frame_num, timestamp = parse_frame_identifier(file_path.name)

        # Get filtered position (preferred) or raw position
        filtered = data.get("filtered_projections", {})
        pos = filtered.get("position_filtered_m")

        if pos is None:
            raw = data.get("trajectory_3d", {})
            pos = raw.get("position_world_m")

        if pos is None:
            continue

        # Negate Y to correct for calibration coordinate system
        # Original data has negative Y values (camera below court)
        positions.append((pos["x"], -pos["y"], pos["z"]))
        metadata_list.append({
            "frame_number": frame_num,
            "timestamp": timestamp,
            "file_path": str(file_path),
        })

    return metadata_list, positions


def get_position_bounds(positions: List[Tuple[float, float, float]]) -> dict:
    """Calculate min/max bounds for positions.

    Args:
        positions: List of (x, y, z) position tuples

    Returns:
        Dict with x_min, x_max, y_min, y_max, z_min, z_max
    """
    if not positions:
        return {
            "x_min": 0, "x_max": 0,
            "y_min": 0, "y_max": 0,
            "z_min": 0, "z_max": 0,
        }

    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    z_vals = [p[2] for p in positions]

    return {
        "x_min": min(x_vals), "x_max": max(x_vals),
        "y_min": min(y_vals), "y_max": max(y_vals),
        "z_min": min(z_vals), "z_max": max(z_vals),
    }
