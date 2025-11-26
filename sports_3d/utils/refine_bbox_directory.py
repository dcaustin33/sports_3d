import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sports_3d.utils.labeling_utils import refined_tennis_ball_box


def yolo_to_pixel(yolo_bbox: str, img_width: int, img_height: int) -> tuple[int, int, int, int]:
    """
    Convert YOLO format bbox to pixel coordinates.

    Args:
        yolo_bbox: String in format "class_id cx cy w h" (normalized)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    parts = yolo_bbox.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO format: {yolo_bbox}")

    class_id, cx, cy, w, h = parts
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)

    cx_pixel = cx * img_width
    cy_pixel = cy * img_height
    w_pixel = w * img_width
    h_pixel = h * img_height

    x1 = int(cx_pixel - w_pixel / 2)
    y1 = int(cy_pixel - h_pixel / 2)
    x2 = int(cx_pixel + w_pixel / 2)
    y2 = int(cy_pixel + h_pixel / 2)

    return x1, y1, x2, y2, class_id


def pixel_to_yolo(x1: int, y1: int, x2: int, y2: int, class_id: str, img_width: int, img_height: int) -> str:
    """
    Convert pixel coordinates to YOLO format.

    Args:
        x1, y1, x2, y2: Bounding box in pixel coordinates
        class_id: Class ID string
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        YOLO format string "class_id cx cy w h"
    """
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def expand_bbox(x1: int, y1: int, x2: int, y2: int, expansion_factor: float, img_width: int, img_height: int) -> tuple[int, int, int, int]:
    """
    Expand bounding box by a given factor while keeping center fixed.

    Args:
        x1, y1, x2, y2: Original bbox in pixel coordinates
        expansion_factor: Factor to expand by (e.g., 1.3 for 30% expansion)
        img_width: Image width for boundary clamping
        img_height: Image height for boundary clamping

    Returns:
        Expanded (x1, y1, x2, y2) clamped to image boundaries
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    new_w = w * expansion_factor
    new_h = h * expansion_factor

    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)

    return new_x1, new_y1, new_x2, new_y2


def process_bbox_file(
    bbox_path: Path,
    image_path: Path,
    expansion_factor: float = 1.3,
    overwrite: bool = True,
    backup: bool = True
) -> dict:
    """
    Process a single bbox file by expanding and refining bounding boxes.

    Args:
        bbox_path: Path to bbox annotation file (YOLO format)
        image_path: Path to corresponding image file
        expansion_factor: Factor to expand bbox before refinement
        overwrite: If True, overwrite original file; if False, save to new file
        backup: If True, create .bak file before overwriting

    Returns:
        Dictionary with processing statistics
    """
    if not bbox_path.exists():
        return {"status": "error", "message": f"Bbox file not found: {bbox_path}"}

    if not image_path.exists():
        return {"status": "error", "message": f"Image file not found: {image_path}"}

    image = cv2.imread(str(image_path))
    if image is None:
        return {"status": "error", "message": f"Could not load image: {image_path}"}

    img_height, img_width = image.shape[:2]

    with open(bbox_path, 'r') as f:
        original_lines = f.readlines()

    if not original_lines:
        return {"status": "skipped", "message": "Empty bbox file"}

    refined_lines = []
    total_boxes = len(original_lines)
    refined_count = 0

    for line in original_lines:
        line = line.strip()
        if not line:
            continue

        try:
            x1, y1, x2, y2, class_id = yolo_to_pixel(line, img_width, img_height)
        except ValueError as e:
            refined_lines.append(line + '\n')
            continue

        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = expand_bbox(
            x1, y1, x2, y2, expansion_factor, img_width, img_height
        )

        expanded_box = [(expanded_x1, expanded_y1), (expanded_x2, expanded_y2)]

        refined_box = refined_tennis_ball_box(expanded_box, image)

        if refined_box != expanded_box:
            (refined_x1, refined_y1), (refined_x2, refined_y2) = refined_box
            refined_line = pixel_to_yolo(
                int(refined_x1), int(refined_y1), int(refined_x2), int(refined_y2),
                class_id, img_width, img_height
            )
            refined_lines.append(refined_line + '\n')
            refined_count += 1
        else:
            refined_lines.append(line + '\n')

    if overwrite:
        if backup and bbox_path.exists():
            backup_path = bbox_path.with_suffix(bbox_path.suffix + '.bak')
            bbox_path.rename(backup_path)

        with open(bbox_path, 'w') as f:
            f.writelines(refined_lines)

        return {
            "status": "success",
            "total_boxes": total_boxes,
            "refined_count": refined_count,
            "file": str(bbox_path)
        }
    else:
        output_path = bbox_path.with_stem(bbox_path.stem + '_refined')
        with open(output_path, 'w') as f:
            f.writelines(refined_lines)

        return {
            "status": "success",
            "total_boxes": total_boxes,
            "refined_count": refined_count,
            "file": str(output_path)
        }


def find_image_for_bbox(bbox_path: Path, image_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find corresponding image file for a bbox annotation file.

    Args:
        bbox_path: Path to bbox file (e.g., image_001_bbox.txt)
        image_dir: Optional directory to search for images (defaults to bbox parent dir)

    Returns:
        Path to image file or None if not found
    """
    search_dir = image_dir if image_dir else bbox_path.parent

    base_name = bbox_path.stem.replace('_bbox', '')

    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    for ext in image_extensions:
        image_path = search_dir / (base_name + ext)
        if image_path.exists():
            return image_path

    return None


def process_directory(
    directory: Path,
    expansion_factor: float = 1.3,
    overwrite: bool = True,
    backup: bool = True,
    image_dir: Optional[Path] = None
):
    """
    Process all bbox files in a directory.

    Args:
        directory: Directory containing bbox annotation files
        expansion_factor: Factor to expand bboxes before refinement
        overwrite: If True, overwrite original files
        backup: If True, create backups before overwriting
        image_dir: Optional separate directory containing images
    """
    bbox_files = sorted(directory.glob('*_bbox.txt'))

    if not bbox_files:
        print(f"No bbox files found in {directory}")
        return

    print(f"Found {len(bbox_files)} bbox files to process")
    print(f"Expansion factor: {expansion_factor}x")
    print(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")
    print(f"Backup mode: {'ON' if backup else 'OFF'}")
    print()

    total_processed = 0
    total_refined = 0
    errors = []

    for i, bbox_path in enumerate(bbox_files, 1):
        image_path = find_image_for_bbox(bbox_path, image_dir)

        if image_path is None:
            error_msg = f"Could not find image for {bbox_path.name}"
            print(f"[{i}/{len(bbox_files)}] ERROR: {error_msg}")
            errors.append(error_msg)
            continue

        print(f"[{i}/{len(bbox_files)}] Processing {bbox_path.name}...", end=" ")

        result = process_bbox_file(bbox_path, image_path, expansion_factor, overwrite, backup)

        if result["status"] == "success":
            total_processed += 1
            total_refined += result["refined_count"]
            print(f"✓ Refined {result['refined_count']}/{result['total_boxes']} boxes")
        elif result["status"] == "skipped":
            print(f"⊘ Skipped: {result['message']}")
        else:
            print(f"✗ Error: {result['message']}")
            errors.append(result["message"])

    print()
    print("="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total files processed: {total_processed}/{len(bbox_files)}")
    print(f"Total boxes refined: {total_refined}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch refine bounding boxes by expanding and detecting tennis balls"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing bbox annotation files (*_bbox.txt)"
    )
    parser.add_argument(
        "--expansion_factor",
        type=float,
        default=1.3,
        help="Factor to expand bboxes before refinement (default: 1.3)"
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Create new files instead of overwriting (adds _refined suffix)"
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Don't create .bak backup files before overwriting"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Separate directory containing images (if different from bbox dir)"
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return

    image_dir = Path(args.image_dir) if args.image_dir else None
    if image_dir and not image_dir.is_dir():
        print(f"Error: Image directory not found: {image_dir}")
        return

    process_directory(
        directory,
        expansion_factor=args.expansion_factor,
        overwrite=not args.no_overwrite,
        backup=not args.no_backup,
        image_dir=image_dir
    )


if __name__ == "__main__":
    main()
