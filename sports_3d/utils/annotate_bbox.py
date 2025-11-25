import argparse
from typing import Optional

import cv2
import numpy as np

from sports_3d.utils.annotation_base import BaseAnnotator


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

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    roi_h, roi_w = roi.shape[:2]
    min_dist = max(roi_w, roi_h) // 2

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=20,
        minRadius=params['minRadius'],
        maxRadius=params['maxRadius']
    )

    if circles is not None and len(circles[0]) > 0:
        roi_center = (roi_w / 2, roi_h / 2)
        best_circle = None
        min_dist_to_center = float('inf')

        for circle in circles[0]:
            x, y, r = circle
            dist = np.sqrt((x - roi_center[0])**2 + (y - roi_center[1])**2)
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

    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_center = (roi_w / 2, roi_h / 2)
    best_contour = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if not (params['minRadius'] <= radius <= params['maxRadius']):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        center_dist = np.sqrt((x - roi_center[0])**2 + (y - roi_center[1])**2)
        score = circularity * area / (center_dist + 1)

        if score > best_score:
            best_score = score
            best_contour = (float(x), float(y), float(radius))

    return best_contour if best_score > 0 else None


def refined_tennis_ball_box(
    box: list[tuple[int, int]],
    image: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Refines a bounding box to tightly fit a tennis ball using multi-method detection.

    Args:
        box: Bounding box [(x1, y1), (x2, y2)] in pixel coordinates
        image: Image array in BGR format (OpenCV standard)
        min_ball_size: Minimum expected ball radius in pixels
        max_ball_size: Maximum expected ball radius in pixels

    Returns:
        Refined box [(x1, y1), (x2, y2)] or original box if refinement fails
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
    min_radius = int(min_dim * 0.4)
    max_radius = int(min_dim * 1.2)

    params = {
        'minRadius': min_radius,
        'maxRadius': max_radius
    }

    result = _detect_ball_hough(roi, params)
    print("result", result)

    if result is None:
        result = _detect_ball_color(roi, params)

    if result is None:
        return box

    cx_roi, cy_roi, radius = result
    cx_img = x1 + cx_roi
    cy_img = y1 + cy_roi

    margin = 1.03
    refined_radius = radius * margin

    refined_x1 = int(max(0, cx_img - refined_radius))
    refined_y1 = int(max(0, cy_img - refined_radius))
    refined_x2 = int(min(w, cx_img + refined_radius))
    refined_y2 = int(min(h, cy_img + refined_radius))

    original_area = (x2 - x1) * (y2 - y1)
    refined_area = (refined_x2 - refined_x1) * (refined_y2 - refined_y1)

    if refined_area > original_area * 1.05:
        return box

    if refined_area < original_area * 0.3:
        return box

    return [(refined_x1, refined_y1), (refined_x2, refined_y2)]


class BBoxAnnotator(BaseAnnotator):
    def __init__(self, image_path: str, class_id: int = 0, output_dir: str = None, refine_tennis_ball: bool = False):
        super().__init__(image_path, output_dir)
        self.class_id = class_id
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.refine_tennis_ball = refine_tennis_ball

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y), (x, y)]
            self.needs_redraw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_box:
                self.current_box[1] = (x, y)
                self.needs_redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_box:
                self.drawing = False
                x1, y1 = self.current_box[0]
                x2, y2 = self.current_box[1]
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    box_to_add = self.current_box

                    if self.refine_tennis_ball:
                        print(f"Refining tennis ball detection...")
                        print(self.current_box)
                        refined_box = refined_tennis_ball_box(self.current_box, self.original_image)
                        print(refined_box)
                        if refined_box != self.current_box:
                            print(f"Box refined successfully")
                            box_to_add = refined_box
                        else:
                            print(f"Could not refine box, using original")

                    self.boxes.append(box_to_add)
                    print(f"Box added. Total boxes: {len(self.boxes)}")
                self.current_box = None
                self.needs_redraw = True

    def draw_display(self):
        self.display_image = self.original_image.copy()

        for box in self.boxes:
            self.draw_box(box, (0, 255, 0), 2)

        if self.current_box:
            self.draw_box(self.current_box, (0, 255, 255), 2)

        mode = "Tennis Ball (Auto-Refine)" if self.refine_tennis_ball else "Standard"
        instructions = [
            f"Boxes: {len(self.boxes)} | Class ID: {self.class_id} | Mode: {mode}",
            "Drag: Draw box | D: Delete last | S: Save | Q: Quit"
        ]
        self.draw_instructions_overlay(instructions)

    def draw_box(self, box, color, thickness):
        (x1, y1), (x2, y2) = box
        cv2.rectangle(self.display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(self.display_image, (int(cx), int(cy)), 3, color, -1)

    def box_to_yolo(self, box):
        (x1, y1), (x2, y2) = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = max(0, min(x1, self.img_width))
        y1 = max(0, min(y1, self.img_height))
        x2 = max(0, min(x2, self.img_width))
        y2 = max(0, min(y2, self.img_height))

        cx = (x1 + x2) / 2 / self.img_width
        cy = (y1 + y2) / 2 / self.img_height
        w = (x2 - x1) / self.img_width
        h = (y2 - y1) / self.img_height

        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def save_annotations(self):
        return self.save_to_file(self.boxes, "_bbox.txt", self.box_to_yolo, "box")

    def delete_last(self):
        self.delete_last_item(self.boxes, "box")

    def run(self):
        self.setup_window("BBox Annotator", self.mouse_callback)

        print(f"\nAnnotating image: {self.image_path}")
        if self.refine_tennis_ball:
            print("Mode: Tennis Ball Auto-Refinement ENABLED")
            print("Draw rough boxes around balls - they'll be automatically refined using Hough + color detection")
        else:
            print("Mode: Standard bounding box annotation")
        print("Drag to draw bounding boxes")
        print("Press 'D' to delete last, 'S' to save, 'Q' to quit\n")

        while True:
            if self.needs_redraw:
                self.draw_display()
                self.show_image()
                self.needs_redraw = False

            key = cv2.waitKey(20) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("Quitting without saving")
                break
            elif key == ord('s') or key == ord('S'):
                if self.save_annotations():
                    break
            elif key == ord('d') or key == ord('D'):
                self.delete_last()

        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Bounding box annotation tool")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID for YOLO format (default: 0)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--refine_tennis_ball", action="store_true",
                       help="Enable automatic tennis ball detection and refinement after drawing boxes")

    args = parser.parse_args()

    annotator = BBoxAnnotator(args.image_path, args.class_id, args.output_dir, args.refine_tennis_ball)
    annotator.run()


if __name__ == "__main__":
    main()
