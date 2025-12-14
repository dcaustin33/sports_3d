import cv2
import numpy as np
from typing import Tuple, Union


def plot_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Plot a bounding box on an image in xyxy format.

    Args:
        image: Input image (BGR format for OpenCV).
        bbox: Bounding box coordinates as (x1, y1, x2, y2).
        color: BGR color tuple for the box.
        thickness: Line thickness in pixels.
        label: Optional text label to draw above the box.
        font_scale: Font scale for the label text.

    Returns:
        Image with bounding box drawn (modified in place).
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 4),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return image
