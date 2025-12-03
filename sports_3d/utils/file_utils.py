"""
Shared file discovery and parsing utilities for frame-based annotations.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional


FRAME_BASE_PATTERN = re.compile(r'(frame_\d+_t[\d.]+s)')
FRAME_NUMBER_PATTERN = re.compile(r'frame_(\d+)')
TIMESTAMP_PATTERN = re.compile(r't([\d.]+)s')


def parse_frame_identifier(filename: str) -> Optional[Tuple[str, int, float]]:
    """Extract base name, frame number, and timestamp from filename.

    Args:
        filename: Filename like 'frame_004200_t70.000s_bbox.txt'

    Returns:
        Tuple of (base_name, frame_number, timestamp) or None if parsing fails
    """
    base_match = FRAME_BASE_PATTERN.search(filename)
    if not base_match:
        return None

    base_name = base_match.group(1)

    frame_match = FRAME_NUMBER_PATTERN.search(base_name)
    timestamp_match = TIMESTAMP_PATTERN.search(base_name)

    if not frame_match or not timestamp_match:
        return None

    frame_number = int(frame_match.group(1))
    timestamp = float(timestamp_match.group(1))

    return base_name, frame_number, timestamp


def discover_files_by_pattern(
    directory: Path,
    glob_pattern: str,
    file_type_label: str,
    require_frame_pattern: bool = True
) -> List[Path]:
    """Generic file discovery with frame number sorting.

    Args:
        directory: Directory to search
        glob_pattern: Glob pattern (e.g., "*_events.txt")
        file_type_label: Human-readable label for error messages
        require_frame_pattern: If True, only include files matching frame pattern

    Returns:
        List of Path objects sorted by frame number

    Raises:
        ValueError: If directory doesn't exist or no files found
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"{file_type_label} directory does not exist: {directory}")

    files = list(directory.glob(glob_pattern))

    if not files:
        raise ValueError(f"No {file_type_label} files ({glob_pattern}) found in {directory}")

    files_with_numbers = []
    for file_path in files:
        parsed = parse_frame_identifier(file_path.name)
        if parsed:
            _, frame_num, _ = parsed
            files_with_numbers.append((frame_num, file_path))
        elif not require_frame_pattern:
            files_with_numbers.append((0, file_path))
        else:
            print(f"Warning: Skipping file with unexpected name: {file_path.name}")

    files_with_numbers.sort(key=lambda x: x[0])
    return [f[1] for f in files_with_numbers]


def serialize_vector_3d(vec, prefix: str, unit: str = "m") -> dict:
    """Serialize 3D vector to both dict and array formats.

    Args:
        vec: Array-like object with 3 elements [x, y, z]
        prefix: Name prefix (e.g., "position_filtered", "velocity")
        unit: Unit suffix (e.g., "m", "m_per_s")

    Returns:
        Dictionary with both named and array formats
    """
    return {
        f"{prefix}_{unit}": {
            "x": float(vec[0]),
            "y": float(vec[1]),
            "z": float(vec[2])
        },
        f"{prefix}_array_{unit}": [float(vec[0]), float(vec[1]), float(vec[2])]
    }
