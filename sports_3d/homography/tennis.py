import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sports_3d.utils.labeling_utils import read_txt_file

# origin is center of the court
three_d_keypoints = [
    (-10.97 / 2, 0, 23.77 / 2),
    (10.97 / 2, 0, 23.77 / 2),
    (-10.97 / 2, 0, -23.77 / 2),
    (10.97 / 2, 0, -23.77 / 2),
    (-8.23 / 2, 0, 23.77 / 2),
    (-8.23 / 2, 0, -23.77 / 2),
    (8.23 / 2, 0, 23.77 / 2),
    (8.23 / 2, 0, -23.77 / 2),
    (-8.23 / 2, 0, 6.4),
    (8.23 / 2, 0, 6.4),
    (-8.23 / 2, 0, -6.4),
    (8.23 / 2, 0, -6.4),
    (0, 0, 6.4),
    (0, 0, -6.4),
    (0, 0.914, 0)
]


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=14):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return F.sigmoid(x)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def postprocess(heatmap, scale=2, low_thresh=250):
    x_pred, y_pred = None, None
    _, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=5,
        minRadius=10,
    )
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred


def get_keypoints(preds: torch.Tensor):
    points = []
    for kps_num in range(14):
        heatmap = (preds[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, scale=1, low_thresh=170)
        points.append((x_pred, y_pred))
    return points


class tracknet_transform(torch.nn.Module):
    def __init__(
        self,
        size: tuple,
    ) -> None:
        super().__init__()
        self.img_size = size

    def __call__(self, img_path: str) -> torch.Tensor:
        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        image = torch.from_numpy(np.array(image) / 255.0).float()
        image = image.permute(2, 0, 1)
        return image


def refine_keypoints(keypoints: list, image: np.ndarray, window_size=20):
    """
    Refines keypoints by finding the centroid of line intersections.
    Robust to thick/pixelated lines.

    Args:
        keypoints: List of (x, y) tuples from initial detection
        image: Original image as numpy array (H, W, 3)
        window_size: Size of the window to search around each keypoint

    Returns:
        List of refined (x, y) tuples
    """
    refined_keypoints = []

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    for x_pred, y_pred in keypoints[:]:
        if x_pred is None or y_pred is None:
            refined_keypoints.append((None, None))
            continue

        x_pred, y_pred = int(x_pred), int(y_pred)

        x_min = max(0, x_pred - window_size)
        x_max = min(w, x_pred + window_size)
        y_min = max(0, y_pred - window_size)
        y_max = min(h, y_pred + window_size)

        window = gray[y_min:y_max, x_min:x_max]

        if window.size == 0:
            refined_keypoints.append((x_pred, y_pred))
            continue

        _, binary = cv2.threshold(window * 255, 200, 255, cv2.THRESH_BINARY)
        skeleton = binary

        corners = cv2.goodFeaturesToTrack(
            skeleton, maxCorners=5, qualityLevel=0.01, minDistance=5, blockSize=3
        )

        center_x, center_y = window.shape[1] // 2, window.shape[0] // 2

        if corners is not None and len(corners) > 0:
            corners = corners.reshape(-1, 2)
            closest = min(
                corners, key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2
            )
            refined_x = closest[0] + x_min
            refined_y = closest[1] + y_min
            refined_keypoints.append((refined_x, refined_y))
        else:
            y_coords, x_coords = np.where(binary > 0)

            if len(x_coords) > 0:
                distances = np.sqrt(
                    (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                )
                weights = np.exp(-distances / 10)

                centroid_x = np.average(x_coords, weights=weights)
                centroid_y = np.average(y_coords, weights=weights)

                refined_x = centroid_x + x_min
                refined_y = centroid_y + y_min
                refined_keypoints.append((refined_x, refined_y))
            else:
                refined_keypoints.append((x_pred, y_pred))
    return refined_keypoints


def estimate_focal_range(image_width, image_height):
    """Conservative search range"""

    # Assume FOV between 30° and 90°
    min_f = image_width / (2 * np.tan(np.radians(90) / 2))  # Wide
    max_f = image_width / (2 * np.tan(np.radians(30) / 2))  # Narrow

    # Add 20% margin
    min_f *= 0.8
    max_f *= 1.2

    return min_f, max_f


def solve_pnp_with_focal_search(
    object_points, image_points, focal_range=(500, 2000), principal_point=None
):
    """
    object_points: Nx3 array of 3D points
    image_points: Nx2 array of 2D image points
    focal_range: (min_f, max_f) to search
    """
    if principal_point is None:
        principal_point = (image_points[:, 0].mean(), image_points[:, 1].mean())

    best_f = None
    best_error = float("inf")
    best_pose = None

    for f in np.linspace(focal_range[0], focal_range[1], 50):
        camera_matrix = np.array(
            [[f, 0, principal_point[0]], [0, f, principal_point[1]], [0, 0, 1]],
            dtype=np.float32,
        )

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points.astype(np.float32),
                image_points.astype(np.float32),
                camera_matrix,
                None,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            continue

        if success:
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, None
            )
            error = np.mean(np.linalg.norm(projected.squeeze() - image_points, axis=1))

            if error < best_error:
                best_error = error
                best_f = f
                best_pose = (rvec, tvec)
    return best_f, best_pose, best_error


def get_points(keypoints: list):
    points_2d = []
    points_3d = []
    for idx, keypoint in enumerate(keypoints):
        if keypoint[0] is not None and keypoint[1] is not None:
            points_2d.append(keypoint)
            points_3d.append(three_d_keypoints[idx])
    return np.array(points_2d), np.array(points_3d)


def plot_keypoints(keypoints: list, image: np.ndarray):
    for idx, keypoint in enumerate(keypoints):
        if keypoint[0] is not None and keypoint[1] is not None:
            # plot text
            cv2.putText(
                image,
                str(idx),
                (int(keypoint[0]), int(keypoint[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 25, (0, 0, 255), -1)
    return image


def get_camera_pose_in_world(rvec, tvec):
    """Get camera position in world coordinates"""
    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    R_world = R.T
    return camera_position, R_world


def extract_camera_transform(extrinsic_matrix):
    """Extract camera position and rotation from extrinsic matrix"""
    R = extrinsic_matrix[:, :3]
    t = extrinsic_matrix[:, 3]
    camera_pos = -R.T @ t
    R_world = R.T
    return camera_pos, R_world


def build_intrinsic_matrix(
    focal_length: float,
    principal_point: tuple[float, float] = None,
    image_width: int = None,
    image_height: int = None,
) -> np.ndarray:
    """
    Build a 3x3 camera intrinsic matrix.

    Args:
        focal_length: Focal length in pixels (assumes fx = fy)
        principal_point: (cx, cy) principal point in pixels.
                        If None, uses image center (requires image_width/height)
        image_width: Image width in pixels (required if principal_point is None)
        image_height: Image height in pixels (required if principal_point is None)

    Returns:
        3x3 intrinsic matrix K:
            [[f,  0, cx],
             [0,  f, cy],
             [0,  0,  1]]
    """
    if principal_point is None:
        if image_width is None or image_height is None:
            raise ValueError(
                "Must provide either principal_point or image_width/image_height"
            )
        cx = image_width / 2
        cy = image_height / 2
    else:
        cx, cy = principal_point

    return np.array(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32
    )


def rvec_tvec_to_extrinsic(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    extrinsic_3x4 = np.hstack([R, tvec.reshape(3, 1)])

    return extrinsic_3x4


def get_2d_and_3d_keypoints(file_path: str):
    keypoints = read_txt_file(file_path)
    keypoints = [tuple(map(float, line.split())) for line in keypoints]
    keypoints = sorted(keypoints, key=lambda x: x[0])

    indices = []
    points_2d = []
    points_3d = []

    for keypoint in keypoints:
        idx = int(keypoint[0])
        indices.append(idx)
        points_2d.append((keypoint[1], keypoint[2]))
        points_3d.append(three_d_keypoints[idx])

    return np.array(indices), np.array(points_2d), np.array(points_3d)


def estimate_depth_in_camera_plane(
    object_width_px: float, focal_px: float, object_width_m: float
):
    return object_width_m * focal_px / object_width_px


def estimate_camera_plane_coordinates(
    image_width_px: float,
    image_height_px: float,
    x_coord_px: float,
    y_coord_px: float,
    object_width_m: float,
    object_width_px: float,
    focal_px: float,
) -> np.ndarray:
    depth = estimate_depth_in_camera_plane(object_width_px, focal_px, object_width_m)
    x_coord_m = (x_coord_px - image_width_px / 2) * depth / focal_px
    y_coord_m = (y_coord_px - image_height_px / 2) * depth / focal_px
    return np.array([x_coord_m, y_coord_m, depth])


def camera_plane_to_world(
    camera_pos: np.ndarray,
    R_world: np.ndarray,
    camera_plane_coordinates: np.ndarray,
) -> np.ndarray:
    return camera_pos + R_world @ camera_plane_coordinates


def pixel_to_court_plane_point(
    pixel_x: float,
    pixel_y: float,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray
) -> np.ndarray | None:
    """
    Project 2D pixel coordinate to 3D world coordinate on court plane (y=0).

    Uses ray-plane intersection to find where the ray from camera through pixel
    intersects the ground plane at y=0.

    Args:
        pixel_x: Pixel x-coordinate
        pixel_y: Pixel y-coordinate
        intrinsic_matrix: 3x3 camera intrinsic matrix K
        extrinsic_matrix: 3x4 camera extrinsic matrix [R | t]

    Returns:
        3D world coordinates [x, 0, z] on court plane, or None if no valid intersection

    Raises:
        ValueError: If ray is parallel to ground plane or intersection is behind camera
    """
    K_inv = np.linalg.inv(intrinsic_matrix)
    ray_cam = K_inv @ np.array([pixel_x, pixel_y, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    camera_pos, R_world = extract_camera_transform(extrinsic_matrix)

    ray_world = R_world @ ray_cam
    ray_world = ray_world / np.linalg.norm(ray_world)

    if abs(ray_world[1]) < 1e-6:
        raise ValueError(f"Ray parallel to ground plane (ray_y={ray_world[1]:.2e})")

    t_intersect = -camera_pos[1] / ray_world[1]

    if t_intersect < 0:
        raise ValueError(f"Intersection behind camera (t={t_intersect:.3f})")

    P_world = camera_pos + t_intersect * ray_world
    P_world[1] = 0.0

    return P_world


def pixel_to_court_plane_depth(
    pixel_x: float,
    pixel_y: float,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray
) -> float | None:
    """
    Project 2D pixel to court plane and return only z-depth.

    Wrapper around pixel_to_court_plane_point() that returns only the z-coordinate.

    Args:
        pixel_x: Pixel x-coordinate
        pixel_y: Pixel y-coordinate
        intrinsic_matrix: 3x3 camera intrinsic matrix K
        extrinsic_matrix: 3x4 camera extrinsic matrix [R | t]

    Returns:
        Z-coordinate (depth) in world space, or None if projection fails

    Raises:
        ValueError: If ray is parallel to ground plane or intersection is behind camera
    """
    P_world = pixel_to_court_plane_point(pixel_x, pixel_y, intrinsic_matrix, extrinsic_matrix)
    if P_world is None:
        return None
    return P_world[2]


def bbox_to_world_coordinates(
    bbox_yolo: tuple[float, float, float, float],
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    image_width: int,
    image_height: int,
    object_width_m: float = 0.066,
    flip_y_axis: bool = False,
) -> np.ndarray:
    """
    Convert bounding box position to 3D world coordinates.

    Args:
        bbox_yolo: YOLO format (cx_norm, cy_norm, w_norm, h_norm)
        intrinsic_matrix: 3x3 camera intrinsic matrix K
        extrinsic_matrix: 3x4 camera extrinsic matrix [R | t]
        image_width: Image width in pixels
        image_height: Image height in pixels
        object_width_m: Real-world object width in meters (default: 0.066 for tennis ball)

    Returns:
        3D world coordinates as np.ndarray of shape (3, 1): [X, Y, Z]
    """
    f = intrinsic_matrix[0, 0]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    camera_pos, R_world = extract_camera_transform(extrinsic_matrix)

    cx_px = bbox_yolo[0] * image_width
    cy_px = bbox_yolo[1] * image_height
    w_px = bbox_yolo[2] * image_width

    depth = object_width_m * f / w_px

    x_cam = (cx_px - cx) * depth / f
    y_cam = (cy_px - cy) * depth / f
    z_cam = depth

    P_cam = np.array([x_cam, y_cam, z_cam])
    P_world = camera_pos + R_world @ P_cam
    
    if flip_y_axis:
        P_world[1] = -P_world[1]

    return P_world.reshape(3, 1)
