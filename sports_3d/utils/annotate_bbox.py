import argparse
from pathlib import Path

import cv2
import numpy as np

from sports_3d.utils.annotation_base import BaseAnnotator
from sports_3d.utils.labeling_utils import refined_tennis_ball_box


class BBoxAnnotator(BaseAnnotator):
    @staticmethod
    def _discover_images(path: Path) -> list[Path]:
        if path.is_file():
            return [path]

        if path.is_dir():
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            images = []
            for ext in extensions:
                images.extend(path.glob(ext))
            return sorted(images)

        raise ValueError(f"Path does not exist: {path}")

    @staticmethod
    def _is_already_annotated(image_path: Path, output_dir: Path) -> bool:
        annotation_file = output_dir / f"{image_path.stem}_bbox.txt"
        return annotation_file.exists()

    def __init__(self, image_path: str, class_id: int = 0, output_dir: str = None, refine_tennis_ball: bool = False):
        input_path = Path(image_path)

        self.output_dir = Path(output_dir) if output_dir else (input_path if input_path.is_dir() else input_path.parent)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_queue = self._discover_images(input_path)
        self.batch_mode = len(self.image_queue) > 1
        self.total_images = len(self.image_queue)
        self.current_index = 0
        self.processed_count = 0
        self.skipped_count = 0
        self.already_annotated_count = 0
        self.is_directory = input_path.is_dir()

        if not self.image_queue:
            raise ValueError(f"No images found in {input_path}")

        for i, img_path in enumerate(self.image_queue):
            if not self._is_already_annotated(img_path, self.output_dir):
                self.current_index = i
                break
        else:
            if self.is_directory:
                print("All images already annotated")
                raise SystemExit(0)

        self.already_annotated_count = sum(
            1 for img in self.image_queue if self._is_already_annotated(img, self.output_dir)
        )

        first_image = str(self.image_queue[self.current_index])
        super().__init__(first_image, str(self.output_dir))

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

    def load_image(self, image_path: Path):
        self.image_path = image_path
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.original_image = self.image.copy()
        self.display_image = self.image.copy()
        self.img_height, self.img_width = self.image.shape[:2]

        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.needs_redraw = True

        if self.batch_mode:
            print(f"\nAnnotating image {self.current_index + 1}/{self.total_images}: {self.image_path.name}")

    def load_next_image(self) -> bool:
        for i in range(self.current_index + 1, self.total_images):
            if not self._is_already_annotated(self.image_queue[i], self.output_dir):
                self.current_index = i
                self.load_image(self.image_queue[i])
                return True
        return False

    def load_previous_image(self) -> bool:
        for i in range(self.current_index - 1, -1, -1):
            if not self._is_already_annotated(self.image_queue[i], self.output_dir):
                self.current_index = i
                self.load_image(self.image_queue[i])
                return True
        print("Already at first unannotated image")
        return False

    def draw_display(self):
        self.display_image = self.original_image.copy()

        for box in self.boxes:
            self.draw_box(box, (0, 255, 0), 2)

        if self.current_box:
            self.draw_box(self.current_box, (0, 255, 255), 2)

        mode = "Tennis Ball (Auto-Refine)" if self.refine_tennis_ball else "Standard"
        if self.batch_mode:
            instructions = [
                f"Image {self.current_index + 1}/{self.total_images} | Boxes: {len(self.boxes)} | Class: {self.class_id} | Mode: {mode}",
                f"Processed: {self.processed_count} | Skipped: {self.skipped_count}",
                "Drag: Box | D: Delete | S: Save+Next | N: Skip | P: Previous | Q: Quit"
            ]
        else:
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

    def print_batch_summary(self):
        remaining = self.total_images - self.already_annotated_count - self.processed_count - self.skipped_count
        print("\n" + "="*50)
        print("Batch annotation session complete")
        print("="*50)
        print(f"Total images found: {self.total_images}")
        print(f"Already annotated: {self.already_annotated_count}")
        print(f"Newly annotated: {self.processed_count}")
        print(f"Skipped: {self.skipped_count}")
        print(f"Remaining: {remaining}")
        print("="*50)

    def run(self):
        self.setup_window("BBox Annotator", self.mouse_callback)

        if self.batch_mode:
            print(f"\nBatch mode: Found {self.total_images} images")
            print(f"Already annotated: {self.already_annotated_count} (auto-skipped)")
            print(f"To process: {self.total_images - self.already_annotated_count} images\n")
            print(f"Annotating image {self.current_index + 1}/{self.total_images}: {self.image_path.name}")
        else:
            print(f"\nAnnotating image: {self.image_path}")

        if self.refine_tennis_ball:
            print("Mode: Tennis Ball Auto-Refinement ENABLED")
            print("Draw rough boxes around balls - they'll be automatically refined using Hough + color detection")
        else:
            print("Mode: Standard bounding box annotation")
        print("Drag to draw bounding boxes")

        if self.batch_mode:
            print("Press 'S' to save+next, 'N' to skip, 'P' for previous, 'D' to delete last, 'Q' to quit\n")
        else:
            print("Press 'D' to delete last, 'S' to save, 'Q' to quit\n")

        while True:
            if self.needs_redraw:
                self.draw_display()
                self.show_image()
                self.needs_redraw = False

            key = cv2.waitKey(20) & 0xFF

            if key == ord('q') or key == ord('Q'):
                if self.batch_mode:
                    self.print_batch_summary()
                else:
                    print("Quitting without saving")
                break
            elif key == ord('s') or key == ord('S'):
                if self.save_annotations():
                    if self.batch_mode:
                        self.processed_count += 1
                        if not self.load_next_image():
                            self.print_batch_summary()
                            break
                    else:
                        break
            elif key == ord('n') or key == ord('N'):
                if self.batch_mode:
                    self.skipped_count += 1
                    if not self.load_next_image():
                        self.print_batch_summary()
                        break
            elif key == ord('p') or key == ord('P'):
                if self.batch_mode:
                    self.load_previous_image()
            elif key == ord('d') or key == ord('D'):
                self.delete_last()

        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Bounding box annotation tool")
    parser.add_argument("image_path", type=str, help="Path to image file or directory")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID for YOLO format (default: 0)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--refine_tennis_ball", action="store_true",
                       help="Enable automatic tennis ball detection and refinement after drawing boxes")

    args = parser.parse_args()

    annotator = BBoxAnnotator(args.image_path, args.class_id, args.output_dir, args.refine_tennis_ball)
    annotator.run()


if __name__ == "__main__":
    main()
