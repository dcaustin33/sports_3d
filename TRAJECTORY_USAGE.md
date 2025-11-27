# Ball Trajectory Tracking Usage Guide

## Overview

The `ball_trajectory.py` script performs camera calibration from tennis court keypoints and projects all ball bounding boxes to 3D world coordinates to generate a complete trajectory.

## Prerequisites

1. **Calibration frame**: One image with visible court keypoints
2. **Keypoints annotation**: Text file with court keypoint locations (format: `index x y`)
3. **Bounding box annotations**: Directory of YOLO format bbox files (format: `class_id cx cy w h`)

## Quick Start

### 1. Annotate Calibration Keypoints

Use the keypoint labeling tool on a clear frame showing the court:

```bash
.venv/bin/python -m sports_3d.utils.label_keypoints path/to/calibration_frame.png
```

This creates `calibration_frame_keypoints.txt` with format:
```
0 500.5 800.2
1 1400.3 802.1
...
```

### 2. Prepare Bounding Box Annotations

Ensure your bbox directory contains YOLO format files:
```
bboxes/
  frame_000001_bbox.txt
  frame_000002_bbox.txt
  ...
```

Each file contains:
```
0 0.5123 0.4567 0.0345 0.0389
```
(class_id, center_x_norm, center_y_norm, width_norm, height_norm)

### 3. Run Trajectory Tracking

```bash
.venv/bin/python ball_trajectory.py \
  --calibration-image data/frame_001.png \
  --calibration-keypoints data/frame_001_keypoints.txt \
  --bbox-dir data/ball_bboxes/ \
  --output-trajectory trajectory.csv \
  --output-plot trajectory_3d.png \
  --visualize
```

## Command-Line Options

### Required Arguments

- `--calibration-image PATH`: Reference frame for camera calibration
- `--calibration-keypoints PATH`: Keypoints file (format: `index x y` per line)
- `--bbox-dir PATH`: Directory with bounding box annotations (YOLO format)

### Optional Arguments

- `--output-trajectory PATH`: Output CSV file (default: `trajectory_world.csv`)
- `--output-plot PATH`: Save 3D visualization to file
- `--ball-diameter FLOAT`: Tennis ball diameter in meters (default: 0.066)
- `--focal-range-min FLOAT`: Minimum focal length for search (pixels)
- `--focal-range-max FLOAT`: Maximum focal length for search (pixels)
- `--visualize`: Display interactive 3D plot
- `--filter-invalid`: Filter out suspicious bounding boxes
- `--file-pattern STR`: File pattern for bbox files (default: `*.txt`)

## Output Format

### CSV Trajectory File

`trajectory_world.csv` contains:

```csv
frame_id,timestamp,x,y,z,depth,bbox_width_px,bbox_height_px
4200,70.000,2.45,1.83,-3.12,15.2,42.3,44.1
4201,70.033,2.51,1.95,-3.05,15.0,43.1,45.2
...
```

**Columns:**
- `frame_id`: Frame number
- `timestamp`: Time in seconds (if available)
- `x`: World X coordinate (meters, court width)
- `y`: World Y coordinate (meters, height above court)
- `z`: World Z coordinate (meters, court length)
- `depth`: Distance from camera (meters)
- `bbox_width_px`, `bbox_height_px`: Bounding box size (pixels)

### 3D Visualization

The plot shows:
- Ball trajectory as colored points (color = time)
- Tennis court outline (green lines)
- Camera position (red triangle)
- Ground projection (gray shadows)

## Coordinate System

**World Coordinates (origin at court center):**
- X-axis: Court width (-5.5m to 5.5m for singles)
- Y-axis: Height above ground (0m = court surface)
- Z-axis: Court length (-11.9m to 11.9m, positive toward far baseline)

**Court Keypoint Indices (0-13):**
```
0-3:   Outer court corners (doubles)
4-7:   Singles sideline intersections
8-11:  Service line intersections
12-13: Net center points
```

## Examples

### Basic Usage

```bash
.venv/bin/python ball_trajectory.py \
  --calibration-image calibration.png \
  --calibration-keypoints calibration_keypoints.txt \
  --bbox-dir bboxes/
```

### With Visualization

```bash
.venv/bin/python ball_trajectory.py \
  --calibration-image calibration.png \
  --calibration-keypoints calibration_keypoints.txt \
  --bbox-dir bboxes/ \
  --output-plot trajectory_visualization.png \
  --visualize
```

### Custom Focal Range

```bash
.venv/bin/python ball_trajectory.py \
  --calibration-image calibration.png \
  --calibration-keypoints calibration_keypoints.txt \
  --bbox-dir bboxes/ \
  --focal-range-min 1000 \
  --focal-range-max 3000
```

### Filter Invalid Detections

```bash
.venv/bin/python ball_trajectory.py \
  --calibration-image calibration.png \
  --calibration-keypoints calibration_keypoints.txt \
  --bbox-dir bboxes/ \
  --filter-invalid
```

## Implementation Details

### Camera Calibration Pipeline

1. **Focal length search** (50 iterations, typically 500-2000px)
2. **PnP solve** using IPPE algorithm (handles planar ambiguity)
3. **Solution disambiguation** using physical constraints (camera behind court, Z < 0)
4. **Reprojection error** computed for validation

### Depth Estimation

Depth is estimated using pinhole camera model:
```
Z = (ball_diameter_m * focal_px) / bbox_width_px
```

Assumes bbox width approximates ball diameter projection.

### 2D to 3D Projection

1. Estimate depth Z from bbox width
2. Back-project to camera frame: `(X_cam, Y_cam, Z_cam)`
3. Transform to world frame: `P_world = camera_pos + R_world @ P_camera`

## Troubleshooting

### High Reprojection Error (> 20px)

- Check keypoint annotations are accurate
- Ensure keypoints cover wide area of court
- Try adjusting focal range

### Invalid World Coordinates

- Points with Y < 0 (below ground) indicate depth estimation errors
- Large |X| values (> 15m) suggest bbox size issues
- Use `--filter-invalid` to remove suspicious detections

### Missing Trajectory Points

- Check bbox files are in correct YOLO format
- Verify file naming for sequential ordering
- Enable verbose output to see skipped frames

## Module API

The implementation is modular and can be used programmatically:

```python
from sports_3d.trajectory import (
    calibrate_camera_from_keypoints,
    project_bbox_to_world,
    load_bboxes_from_directory
)

# Calibrate camera
calibration = calibrate_camera_from_keypoints(
    keypoints_2d, keypoints_3d, image_shape
)

# Project single bbox
world_point = project_bbox_to_world(
    bbox, image_shape,
    calibration['camera_matrix'],
    calibration['camera_position'],
    calibration['R_world']
)
```

## Notes

- Camera position is assumed **stagnant** (fixed throughout video)
- Tennis ball diameter: 66mm (ITF regulation)
- Recommended: 6+ keypoints for robust calibration
- Best results with clear court view and accurate keypoints
