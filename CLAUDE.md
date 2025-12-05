# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3D sports computer vision project focused on tennis court analysis. It uses deep learning to detect court keypoints and solve camera calibration using homography and perspective-n-point (PnP) algorithms.

## Development Setup

This project uses `uv` for dependency management (modern Python package manager).

### Install dependencies
```bash
uv sync
```

### Running Python Commands

**IMPORTANT**: All Python commands must be executed using the `.venv` virtual environment located in the project root.

**Always use the virtual environment Python:**
```bash
.venv/bin/python main.py
.venv/bin/python sports_3d/homography/tennis.py
.venv/bin/python -m sports_3d.utils.annotate_bbox_tennis image.png
.venv/bin/python -m sports_3d.utils.label_keypoints image.png
```

**Do NOT use:**
```bash
python main.py  # ❌ Wrong - uses system Python, not venv
uv run python main.py  # ❌ Wrong - do not use uv run
```

The `.venv` directory is managed by `uv` but commands should directly invoke `.venv/bin/python`.

## Architecture

### Core Components

**sports_3d/homography/tennis.py** - Main computer vision pipeline containing:
- `BallTrackerNet`: Deep CNN (VGG-style encoder-decoder) that outputs 14-15 channel heatmaps for tennis court keypoint detection
- `three_d_keypoints`: 14 predefined 3D world coordinates for tennis court corners and lines (in meters, origin at court center)
- `get_keypoints()`: Extracts 2D keypoint locations from neural network heatmaps using circular Hough transform
- `refine_keypoints()`: Post-processes keypoints using corner detection and weighted centroids for sub-pixel accuracy
- `solve_pnp_with_focal_search()`: Iteratively searches for optimal camera focal length (500-2000px range) while solving PnP
- `solve_planar_pnp()`: Uses IPPE algorithm to handle planar point ambiguity (returns 2 possible camera poses)
- `select_valid_solution()`: Disambiguates camera pose by physical constraints (camera must be above court, Y > 0)

### Model Architecture

The `BallTrackerNet` is a fully convolutional network:
- Input: RGB image (640x360 after resizing)
- Encoder: 3 pooling stages with increasing channels (64 → 128 → 256 → 512)
- Decoder: 3 upsampling stages back to input resolution
- Output: 14-15 channel heatmap (sigmoid activation), one per keypoint
- Custom weight initialization: uniform(-0.05, 0.05) for conv layers

### Data Flow

1. Image preprocessing via `tracknet_transform`: resize to 640x360, normalize to [0,1], convert to tensor
2. Neural network inference produces heatmaps
3. Heatmap post-processing: threshold at 250/255, circular Hough transform to find peaks
4. Keypoint refinement: 20px window around each detection, corner detection or weighted centroid
5. Camera calibration: focal length search (50 iterations), PnP solve with IPPE
6. Pose disambiguation: select solution where camera Y > 0

### File Structure

- `checkpoints/model_tennis_court_det.pt` - Pre-trained model weights (42MB PyTorch checkpoint)
- `data/` - Input images and videos for processing
- `sports_3d/homography/` - Computer vision algorithms:
  - `tennis.py` - Core CV pipeline (keypoint detection, camera calibration)
  - `tennis_trajectory.py` - Batch conversion of bounding boxes to 3D trajectories
  - `kalman_inference.py` - Trajectory filtering with physics-aware smoothing
- `sports_3d/utils/` - Annotation and utility tools:
  - `annotation_base.py` - Base class for all annotation tools
  - `bbox_annotator_base.py` - Reusable bounding box annotation base class
  - `annotate_bbox_tennis.py` - Tennis ball bounding box labeling tool (YOLO format)
  - `label_keypoints.py` - Interactive keypoint labeling tool
  - `labeling_utils.py` - Tennis ball detection utilities (Hough+color)
  - `extract_frames.py` - Video frame extraction utility
  - `file_utils.py` - Shared utilities for file discovery and JSON serialization
  - `kalman.py` - Trajectory filtering implementation
- `main.py` - Entry point (currently just prints hello world)
- `sports_3d/blender/` - Blender visualization package:
  - `__init__.py` - Package init with exports
  - `config.py` - Configuration dataclass with ITF court dimensions
  - `data_loader.py` - Trajectory JSON loading and processing
  - `materials.py` - Blender material creation utilities
  - `court.py` - Tennis court 3D geometry generation
  - `ball.py` - Ball mesh, trail system, keyframe animation
  - `camera.py` - Camera setup and presets
  - `environment.py` - Nishita sky and lighting
  - `render.py` - EEVEE render configuration
  - `main.py` - Entry point for Blender script

## Dependencies

- PyTorch 2.9+ - Neural network inference
- OpenCV 4.11+ - Image processing, PnP solver, Hough circles
- NumPy 2.3+ - Array operations
- Pillow 12.0+ - Image I/O
- Matplotlib 3.10+ - Visualization (imported but not actively used in main code)

## Tennis Court Coordinate System

The 3D world coordinates use meters with origin at court center:
- X-axis: court width (singles court: 8.23m, doubles: 10.97m)
- Y-axis: vertical (ground plane at Y=0)
- Z-axis: court length (full court: 23.77m, service line: 6.4m from center)

Court keypoints (indices 0-13):
- 0-3: Outer court corners
- 4-7: Singles sideline intersections
- 8-11: Service line intersections
- 12-13: Net center points

## Annotation Tools

### Tennis Ball Bounding Box Annotator
```bash
.venv/bin/python -m sports_3d.utils.annotate_bbox_tennis IMAGE_PATH [--class_id 0] [--output_dir PATH]
```
- Drag to draw bounding boxes
- Saves in YOLO format: `class_id cx cy w h` (normalized coordinates)
- Keyboard: D=delete last, S=save, Q=quit, Ctrl+Click=zoom, R=reset zoom
- Support for batch processing (pass directory instead of file)
- Optional `--refine_tennis_ball` flag enables automatic ball detection using Hough circle + color segmentation
- Optional `--refine_current_boxes` flag to refine existing annotations

**Architecture:**
- Extends `BaseBBoxAnnotator` (reusable base class in `bbox_annotator_base.py`)
- Tennis-specific refinement logic in `TennisBBoxAnnotator.refine_box()` hook
- Easy to extend for other sports (soccer, basketball) by inheriting from `BaseBBoxAnnotator`

### Keypoint Labeler
```bash
.venv/bin/python -m sports_3d.utils.label_keypoints IMAGE_PATH [--output_dir PATH]
```
- Click to mark keypoint, then type index (0-13) in popup
- Saves as: `index x y` (pixel coordinates)
- Keyboard: D=delete last, S=save, Q=quit, ESC=cancel current point

Both tools:
- Support `--output_dir` for custom save location (defaults to image directory)
- Event-driven rendering for responsive performance
- Share common base class in `annotation_base.py`

## Trajectory Processing Pipeline

### 1. Bounding Box to 3D Trajectory Conversion
```bash
.venv/bin/python -m sports_3d.homography.tennis_trajectory \
    <bbox_dir> <keypoints_dir> <frames_dir> <output_dir> \
    [--ball_diameter 0.066] [--verbose]
```
Converts tennis ball bounding boxes to 3D world coordinates:
- Reads YOLO format bounding boxes and court keypoint annotations
- Performs camera calibration for each frame (with focal length search)
- Interpolates keypoints between frames when not available
- Projects 2D bounding boxes to 3D world coordinates using calibrated camera
- Outputs trajectory JSON files with camera calibration and 3D positions

**Key features:**
- `KeypointInterpolator`: Linearly interpolates 2D keypoints across frames
- `compute_frame_calibration()`: Per-frame camera calibration with PnP solver
- Eliminates duplicate image loading for performance
- Uses shared utilities from `file_utils.py` for file discovery

### 2. Trajectory Filtering and Smoothing
```bash
.venv/bin/python -m sports_3d.homography.kalman_inference \
    <trajectory_dir> <events_dir> \
    [--window_size_xy 7] [--poly_order 2] [--verbose] [--backup]
```
Applies physics-aware filtering to trajectory JSON files:
- Z-axis: Quadratic fitting with velocity decay constraint
- X/Y-axes: Savitzky-Golay polynomial smoothing
- Automatic discontinuity detection (bounces, racquet hits)
- Event-based refinement using ground/racquet contact annotations
- Adds `filtered_projections` to existing JSON files

**Key features:**
- Uses shared `discover_files_by_pattern()` for consistent file discovery
- `refine_position_hybrid()`: Improves positions using event annotations
- `serialize_vector_3d()`: Standardized JSON serialization for 3D vectors
- Supports backup files with `--backup` flag

## Shared Utilities

### `sports_3d/utils/file_utils.py`
Common utilities for file operations and data serialization across the codebase:

**Functions:**
- `parse_frame_identifier(filename)`: Extracts base name, frame number, and timestamp from filenames
- `discover_files_by_pattern(directory, glob_pattern, file_type_label)`: Generic file discovery with frame number sorting
- `serialize_vector_3d(vec, prefix, unit)`: Serializes 3D vectors to both dict and array JSON formats

**Constants:**
- `FRAME_BASE_PATTERN`: Regex for matching frame identifiers (`frame_\d+_t[\d.]+s`)
- `FRAME_NUMBER_PATTERN`: Regex for extracting frame numbers
- `TIMESTAMP_PATTERN`: Regex for extracting timestamps

**Usage:** Import and use these utilities when adding new file processing scripts to maintain consistency across the codebase.

## Blender 3D Visualization

### Overview
The `sports_3d/blender/` package creates high-quality 3D visualizations of tennis ball trajectories in Blender. Requires Blender 4.x.

### Running the Visualization

**From Blender GUI:**
1. Open Blender 4.x
2. Go to Scripting workspace (top tab bar)
3. Click "Open" and navigate to `sports_3d/blender/main.py`
4. Press Alt+P or click "Run Script"

**From Command Line:**
```bash
blender --python sports_3d/blender/main.py
```

**With Custom Parameters:**
```bash
blender --python sports_3d/blender/main.py -- \
    --trajectory_dir data/sinner_ruud_trajectory \
    --camera_z -20.0 \
    --trail_length 20
```

### Configuration
Edit `sports_3d/blender/config.py` to customize:
- Camera position and FOV
- Court colors (Australian Open blue by default)
- Trail length and opacity
- Render resolution and quality
- Sky sun position

### Interactive Controls
- **Space**: Play/pause animation
- **←/→**: Previous/next frame
- **Shift+←/→**: Jump 10 frames
- **Home/End**: Go to start/end
- **Numpad 0**: Camera view
- **N**: Toggle sidebar (switch cameras)
- **F12**: Render current frame
- **Ctrl+F12**: Render animation

### Camera Presets
The script creates multiple cameras for different viewing angles:
- `MainCamera`: User's configured position (default behind baseline)
- `Cam_BroadcastNegZ`: High broadcast angle from negative Z
- `Cam_BroadcastPosZ`: High broadcast angle from positive Z
- `Cam_Courtside`: Side view
- `Cam_Overhead`: Bird's eye view
- `Cam_NetLevel`: Low dramatic angle

Switch cameras in the sidebar (N key) under Scene > Camera.

### Key Technical Notes
- Y values in trajectory data are negated during import (original calibration produces negative Y)
- Uses EEVEE render engine for real-time preview
- Trail balls use graduated opacity (older = more transparent)
- Court uses procedural noise texture for subtle realism

## Coding Style Preferences

- Write clear, concise code without unnecessary comments
- Avoid adding tests unless explicitly requested
- Prioritize code readability through self-documenting function/variable names over extensive documentation
- Keep implementations simple and focused
- Use shared utilities from `file_utils.py` for file discovery and JSON serialization to avoid code duplication

## Important Notes

- The model expects specific input dimensions (640x360) - resizing is handled by `tracknet_transform`
- Camera focal length is unknown and must be estimated (typical range: 500-2000px for consumer cameras)
- Planar points create pose ambiguity - always use `select_valid_solution()` to pick physically valid camera position
- Hardcoded absolute paths in `tennis.py:377,384` should be updated to relative paths or command-line arguments for portability
