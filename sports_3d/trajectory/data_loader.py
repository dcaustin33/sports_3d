import os
import re
from pathlib import Path
from typing import List, Optional
import glob


def parse_yolo_bbox(line: str) -> dict:
    """
    Parse single YOLO format bounding box line.

    Args:
        line: String in format "class_id cx cy w h" (normalized [0,1])

    Returns:
        Dictionary with parsed values
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO format: expected 5 values, got {len(parts)}")

    return {
        'class_id': int(parts[0]),
        'cx': float(parts[1]),
        'cy': float(parts[2]),
        'w': float(parts[3]),
        'h': float(parts[4])
    }


def get_frame_id_from_filename(filepath: str) -> tuple:
    """
    Extract frame identifier from filename for sorting.

    Supports patterns:
        - frame_004200_t70.000s.txt → (4200, 70.0)
        - frame_004200.txt → (4200, None)
        - img_001234.txt → (1234, None)

    Args:
        filepath: Path to file

    Returns:
        (frame_number, timestamp_seconds) tuple
    """
    filename = Path(filepath).stem

    frame_match = re.search(r'frame_(\d+)', filename)
    if frame_match:
        frame_num = int(frame_match.group(1))
    else:
        num_match = re.search(r'(\d+)', filename)
        frame_num = int(num_match.group(1)) if num_match else 0

    timestamp_match = re.search(r't([\d.]+)s', filename)
    timestamp = float(timestamp_match.group(1)) if timestamp_match else None

    return frame_num, timestamp


def load_bboxes_from_directory(
    bbox_dir: str,
    file_pattern: str = '*.txt',
    sort_by: str = 'filename'
) -> List[dict]:
    """
    Load all bounding box annotations from directory.

    Args:
        bbox_dir: Directory containing bbox annotation files
        file_pattern: Glob pattern to match files (default: '*.txt')
        sort_by: Sorting method - 'filename' or 'timestamp'

    Returns:
        List of dictionaries containing:
            - frame_id: Numeric frame identifier
            - timestamp: Timestamp in seconds (if available)
            - filepath: Full path to annotation file
            - bboxes: List of bbox dicts with {class_id, cx, cy, w, h}
    """
    bbox_dir = Path(bbox_dir)
    if not bbox_dir.exists():
        raise FileNotFoundError(f"Bbox directory not found: {bbox_dir}")

    pattern = str(bbox_dir / file_pattern)
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    frames_data = []

    for filepath in files:
        frame_num, timestamp = get_frame_id_from_filename(filepath)

        bboxes = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        bbox = parse_yolo_bbox(line)
                        bboxes.append(bbox)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid line in {filepath}: {e}")
                        continue

        frames_data.append({
            'frame_id': frame_num,
            'timestamp': timestamp,
            'filepath': filepath,
            'bboxes': bboxes
        })

    if sort_by == 'timestamp':
        frames_data.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else x['frame_id'])
    else:
        frames_data.sort(key=lambda x: x['frame_id'])

    return frames_data


def filter_valid_bboxes(
    bbox: dict,
    min_width_px: float = 10.0,
    max_aspect_ratio: float = 2.0,
    image_shape: Optional[tuple] = None
) -> bool:
    """
    Filter out invalid or suspicious bounding boxes.

    Args:
        bbox: Bbox dictionary with normalized dimensions
        min_width_px: Minimum width in pixels after denormalization
        max_aspect_ratio: Maximum width/height ratio
        image_shape: (height, width) for denormalization (if None, skip pixel checks)

    Returns:
        True if bbox is valid, False otherwise
    """
    w_norm = bbox['w']
    h_norm = bbox['h']

    if w_norm <= 0 or h_norm <= 0:
        return False

    aspect_ratio = w_norm / h_norm
    if aspect_ratio > max_aspect_ratio or aspect_ratio < 1.0 / max_aspect_ratio:
        return False

    if image_shape is not None:
        height, width = image_shape
        w_px = w_norm * width
        if w_px < min_width_px:
            return False

    return True
