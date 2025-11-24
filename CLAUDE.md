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

### Activate virtual environment
```bash
source .venv/bin/activate
```

### Run the main script
```bash
python main.py
```

### Run tennis court detection
```bash
python sports_3d/homography/tennis.py
```

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
- `data/` - Input images for processing
- `sports_3d/homography/` - Computer vision algorithms
- `sports_3d/utils/` - Currently empty utility module
- `main.py` - Entry point (currently just prints hello world)

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

## Coding Style Preferences

- Write clear, concise code without unnecessary comments
- Avoid adding tests unless explicitly requested
- Prioritize code readability through self-documenting function/variable names over extensive documentation
- Keep implementations simple and focused

## Important Notes

- The model expects specific input dimensions (640x360) - resizing is handled by `tracknet_transform`
- Camera focal length is unknown and must be estimated (typical range: 500-2000px for consumer cameras)
- Planar points create pose ambiguity - always use `select_valid_solution()` to pick physically valid camera position
- Hardcoded absolute paths in `tennis.py:377,384` should be updated to relative paths or command-line arguments for portability
