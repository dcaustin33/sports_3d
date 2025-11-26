from typing import Optional
import os

import cv2
import numpy as np


def _detect_ball_hough(roi: np.ndarray, params: dict) -> Optional[tuple]:
    """
    Detect tennis ball using circular Hough transform.

    Args:
        roi: Region of interest in BGR format
        params: Detection parameters (minRadius, maxRadius)

    Returns:
        (center_x, center_y, radius) in ROI coordinates, or None if detection fails
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    roi_h, roi_w = roi.shape[:2]
    min_dist = max(roi_w, roi_h) // 2

    # extract yellow and green
    lower_green_yellow = np.array([25, 50, 50])
    upper_green_yellow = np.array([85, 255, 255])
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi_hsv, lower_green_yellow, upper_green_yellow)

    circles_gray = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=20,
        param2=15,  # Slightly higher threshold for stronger circle evidence
        minRadius=params["minRadius"],
        maxRadius=params["maxRadius"],
    )

    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=20,
        param2=15,  # Slightly higher threshold for stronger circle evidence
        minRadius=params["minRadius"],
        maxRadius=params["maxRadius"],
    )

    if circles is not None:
        circles = circles + circles_gray if circles_gray is not None else circles
    else:
        circles = circles_gray if circles_gray is not None else None
    if circles is not None and len(circles[0]) > 0:
        roi_center = (roi_w / 2, roi_h / 2)
        best_circle = None
        min_dist_to_center = float("inf")

        for circle in circles[0]:
            x, y, r = circle
            dist = np.sqrt((x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_circle = (float(x), float(y), float(r))

        return best_circle

    return None


def _detect_ball_color(roi: np.ndarray, params: dict) -> Optional[tuple]:
    """
    Detect tennis ball using HSV color segmentation.

    Args:
        roi: Region of interest in BGR format
        params: Detection parameters (minRadius, maxRadius)

    Returns:
        (center_x, center_y, radius) in ROI coordinates, or None if detection fails
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Hue 25-85 covers yellow-green range typical of tennis balls
    lower_green_yellow = np.array([25, 50, 50])
    upper_green_yellow = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green_yellow, upper_green_yellow)

    # Fill holes from black text with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_center = (roi_w / 2, roi_h / 2)
    best_contour = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if not (params["minRadius"] <= radius <= params["maxRadius"]):
            continue

        # Use convex hull for circularity to handle irregular contours from text
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)

        if hull_perimeter == 0:
            continue
        circularity = 4 * np.pi * hull_area / (hull_perimeter**2)

        center_dist = np.sqrt((x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2)
        score = circularity * area / (center_dist + 1)

        if score > best_score:
            best_score = score
            best_contour = (float(x), float(y), float(radius))

    return best_contour if best_score > 0 else None


def refine_with_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Combine all edge points
        all_points = np.vstack(contours)
        (x, y), radius = cv2.minEnclosingCircle(all_points)
        x, y, radius = int(x), int(y), int(radius)
        return (x - radius, y - radius, 2 * radius, 2 * radius)

    return None


def refined_tennis_ball_box(
    box: list[tuple[int, int]],
    image: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Refines a bounding box to tightly fit a tennis ball using multi-method detection.

    Args:
        box: Bounding box [(x1, y1), (x2, y2)] in pixel coordinates
        image: Image array in BGR format (OpenCV standard)

    Returns:
        Refined box [(x1, y1), (x2, y2)] or original box if all refinement methods fail
    """
    (x1, y1), (x2, y2) = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return box

    roi = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]

    min_dim = min(roi_w, roi_h)
    min_radius = int(min_dim * 0.01)
    max_radius = int(min_dim * 1.2)

    params = {"minRadius": min_radius, "maxRadius": max_radius}

    result = _detect_ball_hough(roi, params)

    if result is None:
        result = _detect_ball_color(roi, params)

    if result is None:
        edge_box = refine_with_edges(roi)
        if edge_box is not None:
            edge_x1, edge_y1, edge_w, edge_h = edge_box
            refined_x1 = int(max(x1, x1 + edge_x1))
            refined_y1 = int(max(y1, y1 + edge_y1))
            refined_x2 = int(min(x2, x1 + edge_x1 + edge_w))
            refined_y2 = int(min(y2, y1 + edge_y1 + edge_h))

            original_area = (x2 - x1) * (y2 - y1)
            refined_area = (refined_x2 - refined_x1) * (refined_y2 - refined_y1)

            if (
                refined_area <= original_area * 1.05
                and refined_area >= original_area * 0.3
            ):
                return [(refined_x1, refined_y1), (refined_x2, refined_y2)]

        return box

    cx_roi, cy_roi, radius = result
    cx_img = x1 + cx_roi
    cy_img = y1 + cy_roi

    margin = 1.03
    refined_radius = radius * margin

    refined_x1 = int(max(x1, cx_img - refined_radius))
    refined_y1 = int(max(y1, cy_img - refined_radius))
    refined_x2 = int(min(x2, cx_img + refined_radius))
    refined_y2 = int(min(y2, cy_img + refined_radius))

    original_area = (x2 - x1) * (y2 - y1)
    refined_area = (refined_x2 - refined_x1) * (refined_y2 - refined_y1)

    if refined_area > original_area * 1.05:
        return box

    if refined_area < original_area * 0.05:
        return box

    return [(refined_x1, refined_y1), (refined_x2, refined_y2)]


def read_txt_file(file_path: str):
    with open(file_path, "r") as f:
        return f.readlines()


def read_yolo_box(file_path: str):
    lines = read_txt_file(file_path)
    return [list(map(float, line.split()[1:])) for line in lines]


def yolo_to_x1x2y1y2(yolo_box: list[float], image_width: int, image_height: int):
    cx, cy, w, h = yolo_box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return (
        int(x1 * image_width),
        int(y1 * image_height),
        int(x2 * image_width),
        int(y2 * image_height),
    )
