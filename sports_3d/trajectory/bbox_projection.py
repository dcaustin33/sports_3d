import numpy as np
from typing import Union, Tuple


def estimate_ball_depth(
    bbox_width_px: float,
    focal_px: float,
    ball_diameter_m: float = 0.066
) -> float:
    """
    Estimate ball distance from camera using pinhole camera model.

    Assumes bbox width approximates ball diameter projection.

    Args:
        bbox_width_px: Bounding box width in pixels
        focal_px: Camera focal length in pixels
        ball_diameter_m: Tennis ball diameter in meters (ITF regulation: 66mm)

    Returns:
        Estimated depth Z in camera frame (meters)
    """
    return ball_diameter_m * focal_px / bbox_width_px


def project_bbox_to_camera_frame(
    bbox_center_px: Tuple[float, float],
    bbox_width_px: float,
    focal_px: float,
    principal_point: Tuple[float, float],
    ball_diameter_m: float = 0.066
) -> np.ndarray:
    """
    Back-project 2D bounding box to 3D point in camera coordinate frame.

    Args:
        bbox_center_px: (cx, cy) bounding box center in pixels
        bbox_width_px: Bounding box width in pixels (for depth estimation)
        focal_px: Camera focal length in pixels
        principal_point: (ppx, ppy) principal point in pixels
        ball_diameter_m: Ball diameter in meters

    Returns:
        3x1 array [X_cam, Y_cam, Z_cam] in meters
    """
    cx, cy = bbox_center_px
    ppx, ppy = principal_point

    Z = estimate_ball_depth(bbox_width_px, focal_px, ball_diameter_m)
    X = (cx - ppx) * Z / focal_px
    Y = (cy - ppy) * Z / focal_px

    return np.array([[X], [Y], [Z]], dtype=np.float32)


def camera_frame_to_world(
    point_camera: np.ndarray,
    camera_position: np.ndarray,
    R_world: np.ndarray
) -> np.ndarray:
    """
    Transform point from camera frame to world frame.

    Args:
        point_camera: 3x1 point in camera coordinates
        camera_position: 3x1 camera position in world coordinates
        R_world: 3x3 rotation matrix (camera-to-world)

    Returns:
        3x1 point in world coordinates
    """
    point_camera = point_camera.reshape(3, 1)
    camera_position = camera_position.reshape(3, 1)

    return camera_position + R_world @ point_camera


def project_bbox_to_world(
    bbox: Union[list, dict],
    image_shape: Tuple[int, int],
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    R_world: np.ndarray,
    ball_diameter_m: float = 0.066
) -> dict:
    """
    Project 2D bounding box to 3D world coordinates.

    Unified function combining all projection steps.

    Args:
        bbox: YOLO format [class_id, cx_norm, cy_norm, w_norm, h_norm] OR
              dict with keys {cx, cy, w, h} (normalized [0,1])
        image_shape: (height, width) tuple
        camera_matrix: 3x3 intrinsic matrix K
        camera_position: 3x1 camera position in world frame
        R_world: 3x3 rotation matrix (camera-to-world)
        ball_diameter_m: Ball diameter in meters

    Returns:
        Dictionary with:
            - x, y, z: World coordinates in meters
            - depth: Distance from camera in meters
            - bbox_width_px: Bounding box width in pixels
    """
    height, width = image_shape
    focal_px = camera_matrix[0, 0]
    principal_point = (camera_matrix[0, 2], camera_matrix[1, 2])

    if isinstance(bbox, dict):
        cx_norm = bbox['cx']
        cy_norm = bbox['cy']
        w_norm = bbox['w']
        h_norm = bbox['h']
    else:
        if len(bbox) == 5:
            _, cx_norm, cy_norm, w_norm, h_norm = bbox
        else:
            cx_norm, cy_norm, w_norm, h_norm = bbox

    cx_px = cx_norm * width
    cy_px = cy_norm * height
    w_px = w_norm * width
    h_px = h_norm * height

    point_camera = project_bbox_to_camera_frame(
        bbox_center_px=(cx_px, cy_px),
        bbox_width_px=w_px,
        focal_px=focal_px,
        principal_point=principal_point,
        ball_diameter_m=ball_diameter_m
    )

    point_world = camera_frame_to_world(point_camera, camera_position, R_world)

    return {
        'x': float(point_world[0, 0]),
        'y': float(point_world[1, 0]),
        'z': float(point_world[2, 0]),
        'depth': float(point_camera[2, 0]),
        'bbox_width_px': float(w_px),
        'bbox_height_px': float(h_px)
    }
