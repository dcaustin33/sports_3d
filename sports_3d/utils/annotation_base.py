from pathlib import Path

import cv2


class BaseAnnotator:
    def __init__(self, image_path: str, output_dir: str = None):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.output_dir = Path(output_dir) if output_dir else self.image_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.original_image = self.image.copy()
        self.display_image = self.image.copy()
        self.img_height, self.img_width = self.image.shape[:2]

        self.needs_redraw = True
        self.window_name = "Annotator"

    def draw_text_with_outline(self, text: str, position: tuple, font_size: float = 1.0,
                              outline_color: tuple = (0, 0, 0), text_color: tuple = (255, 255, 255),
                              outline_thickness: int = 4, text_thickness: int = 2):
        """Draw text with an outline for better visibility."""
        cv2.putText(
            self.display_image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            outline_color,
            outline_thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            self.display_image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )

    def draw_instructions_overlay(self, instructions: list, y_offset: int = 30, spacing: int = 25):
        """Draw instruction text overlay on the image."""
        for i, text in enumerate(instructions):
            position = (10, y_offset + i * spacing)
            self.draw_text_with_outline(text, position)

    def delete_last_item(self, items_list: list, item_name: str = "item"):
        """Delete the last item from a list and trigger redraw."""
        if items_list:
            items_list.pop()
            print(f"Deleted last {item_name}. Remaining: {len(items_list)}")
            self.needs_redraw = True
        else:
            print(f"No {item_name}s to delete")

    def save_to_file(self, items: list, suffix: str, formatter_func, item_name: str = "item"):
        """
        Generic save method.

        Args:
            items: List of items to save
            suffix: File suffix (e.g., '_bbox.txt', '_keypoints.txt')
            formatter_func: Function that formats each item for saving
            item_name: Name of item type for messages

        Returns:
            True if saved successfully, False otherwise
        """
        if not items:
            print(f"No {item_name}s to save")
            return False

        output_path = self.output_dir / f"{self.image_path.stem}{suffix}"
        with open(output_path, 'w') as f:
            for item in items:
                f.write(formatter_func(item) + '\n')

        print(f"Saved {len(items)} {item_name}s to {output_path}")
        return True

    def setup_window(self, window_name: str, mouse_callback):
        """Create window and set up mouse callback."""
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_callback)

    def show_image(self):
        """Display the current image."""
        cv2.imshow(self.window_name, self.display_image)

    def cleanup(self):
        """Clean up CV2 windows."""
        cv2.destroyAllWindows()
