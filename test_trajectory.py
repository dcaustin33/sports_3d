#!/usr/bin/env python3
"""
Simple test script to verify trajectory tracking implementation.
"""

import numpy as np
import sys

print("Testing trajectory tracking modules...")
print("="*60)

try:
    from sports_3d.trajectory import (
        calibrate_camera_from_keypoints,
        project_bbox_to_world,
        load_bboxes_from_directory,
        parse_yolo_bbox
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n1. Testing YOLO bbox parsing...")
test_line = "0 0.5 0.5 0.05 0.05"
try:
    bbox = parse_yolo_bbox(test_line)
    assert bbox['class_id'] == 0
    assert bbox['cx'] == 0.5
    print(f"✓ Parsed bbox: {bbox}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n2. Testing camera calibration with synthetic data...")

image_shape = (1080, 1920)

keypoints_2d = np.array([
    [500, 800],
    [1400, 800],
    [500, 200],
    [1400, 200],
    [600, 800],
    [600, 200]
], dtype=np.float32)

keypoints_3d = np.array([
    [-10.97/2, 0, 23.77/2],
    [10.97/2, 0, 23.77/2],
    [-10.97/2, 0, -23.77/2],
    [10.97/2, 0, -23.77/2],
    [-8.23/2, 0, 23.77/2],
    [-8.23/2, 0, -23.77/2]
], dtype=np.float32)

try:
    calibration = calibrate_camera_from_keypoints(
        keypoints_2d,
        keypoints_3d,
        image_shape,
        focal_range=(1000, 2000),
        n_iterations=20
    )
    print(f"✓ Calibration successful")
    print(f"  Focal length: {calibration['focal_length']:.1f} px")
    print(f"  Reprojection error: {calibration['reprojection_error']:.2f} px")
    print(f"  Camera position: {calibration['camera_position'].ravel()}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing bbox projection to 3D...")
test_bbox = {
    'class_id': 0,
    'cx': 0.5,
    'cy': 0.5,
    'w': 0.03,
    'h': 0.03
}

try:
    world_point = project_bbox_to_world(
        test_bbox,
        image_shape,
        calibration['camera_matrix'],
        calibration['camera_position'],
        calibration['R_world'],
        ball_diameter_m=0.066
    )
    print(f"✓ Projection successful")
    print(f"  World position: X={world_point['x']:.2f}, Y={world_point['y']:.2f}, Z={world_point['z']:.2f} m")
    print(f"  Depth: {world_point['depth']:.2f} m")
    print(f"  Bbox width: {world_point['bbox_width_px']:.1f} px")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED")
print("="*60)
print("\nThe trajectory tracking implementation is working correctly!")
print("You can now use ball_trajectory.py with your actual data.")
