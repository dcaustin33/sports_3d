import cv2
import numpy as np
from typing import Tuple, Optional


def estimate_focal_range(image_width: int, image_height: int) -> Tuple[float, float]:
    """
    Estimate conservative focal length search range based on typical FOV assumptions.

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        (min_f, max_f): Focal length range in pixels
    """
    min_f = image_width / (2 * np.tan(np.radians(90) / 2))
    max_f = image_width / (2 * np.tan(np.radians(30) / 2))

    min_f *= 0.8
    max_f *= 1.2

    return min_f, max_f


def get_camera_pose_in_world(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert OpenCV camera pose (world-to-camera) to world-centric representation.

    Args:
        rvec: 3x1 rotation vector (axis-angle)
        tvec: 3x1 translation vector

    Returns:
        camera_position: 3x1 camera position in world coordinates
        R_world: 3x3 rotation matrix (camera-to-world)
    """
    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    R_world = R.T

    return camera_position, R_world


def solve_planar_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray
) -> Tuple[list, list, list]:
    """
    Solve PnP for planar points using IPPE algorithm (returns 2 solutions).

    Args:
        object_points: Nx3 array of 3D world points
        image_points: Nx2 array of corresponding 2D image points
        camera_matrix: 3x3 camera intrinsic matrix

    Returns:
        rvecs: List of rotation vectors (typically 2 solutions)
        tvecs: List of translation vectors
        reprojErrors: List of reprojection errors
    """
    success, rvecs, tvecs, reprojErrors = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        raise ValueError("PnP solve failed")

    return rvecs, tvecs, reprojErrors


def select_valid_solution(
    rvecs: list,
    tvecs: list,
    reprojErrors: list
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Select physically valid camera pose from multiple PnP solutions.

    For tennis court: camera should be behind the court (Z < 0) when viewing.

    Args:
        rvecs: List of rotation vectors
        tvecs: List of translation vectors
        reprojErrors: List of reprojection errors

    Returns:
        rvec: Selected rotation vector
        tvec: Selected translation vector
        error: Reprojection error for selected solution
    """
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        camera_pos, _ = get_camera_pose_in_world(rvec, tvec)

        if camera_pos[2] < 0:
            return rvec, tvec, reprojErrors[i]

    best_idx = np.argmin(reprojErrors)
    return rvecs[best_idx], tvecs[best_idx], reprojErrors[best_idx]


def calibrate_camera_from_keypoints(
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
    image_shape: Tuple[int, int],
    focal_range: Optional[Tuple[float, float]] = None,
    n_iterations: int = 50
) -> dict:
    """
    Calibrate camera from court keypoint correspondences.

    Performs focal length search followed by planar PnP solving with
    physical constraint-based solution selection.

    Args:
        keypoints_2d: Nx2 array of image keypoints in pixels
        keypoints_3d: Nx3 array of corresponding world coordinates in meters
        image_shape: (height, width) tuple
        focal_range: Optional (min_f, max_f) for search, auto-estimated if None
        n_iterations: Number of focal length candidates to test

    Returns:
        Dictionary containing:
            - camera_matrix: 3x3 intrinsic matrix K
            - rvec: 3x1 rotation vector (world-to-camera)
            - tvec: 3x1 translation vector (world-to-camera)
            - camera_position: 3x1 camera position in world frame
            - R_world: 3x3 rotation matrix (camera-to-world)
            - focal_length: Estimated focal length in pixels
            - reprojection_error: Final calibration error in pixels
    """
    height, width = image_shape
    principal_point = (width / 2, height / 2)

    if focal_range is None:
        focal_range = estimate_focal_range(width, height)

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)

    best_f = None
    best_error = float('inf')
    best_pose = None

    for f in np.linspace(focal_range[0], focal_range[1], n_iterations):
        camera_matrix = np.array(
            [[f, 0, principal_point[0]],
             [0, f, principal_point[1]],
             [0, 0, 1]],
            dtype=np.float32
        )

        success, rvec, tvec = cv2.solvePnP(
            keypoints_3d,
            keypoints_2d,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_IPPE
        )

        if success:
            projected, _ = cv2.projectPoints(
                keypoints_3d, rvec, tvec, camera_matrix, None
            )
            error = np.mean(np.linalg.norm(projected.squeeze() - keypoints_2d, axis=1))

            if error < best_error:
                best_error = error
                best_f = f
                best_pose = (rvec, tvec)

    if best_f is None:
        raise ValueError("Camera calibration failed: no valid focal length found")

    camera_matrix = np.array(
        [[best_f, 0, principal_point[0]],
         [0, best_f, principal_point[1]],
         [0, 0, 1]],
        dtype=np.float32
    )

    rvecs, tvecs, errors = solve_planar_pnp(keypoints_3d, keypoints_2d, camera_matrix)
    rvec, tvec, error = select_valid_solution(rvecs, tvecs, errors)

    camera_position, R_world = get_camera_pose_in_world(rvec, tvec)

    return {
        'camera_matrix': camera_matrix,
        'rvec': rvec,
        'tvec': tvec,
        'camera_position': camera_position,
        'R_world': R_world,
        'focal_length': best_f,
        'reprojection_error': float(error)
    }
