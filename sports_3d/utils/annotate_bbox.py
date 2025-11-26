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

        self.zoom_level = 1.0
        self.zoom_center = (self.img_width / 2, self.img_height / 2)

    def get_crop_region(self):
        crop_width = self.img_width / self.zoom_level
        crop_height = self.img_height / self.zoom_level
        crop_x1 = self.zoom_center[0] - crop_width / 2
        crop_y1 = self.zoom_center[1] - crop_height / 2

        crop_x1 = max(0, min(crop_x1, self.img_width - crop_width))
        crop_y1 = max(0, min(crop_y1, self.img_height - crop_height))

        return crop_x1, crop_y1, crop_width, crop_height

    def display_to_original(self, x, y):
        if self.zoom_level == 1.0:
            return x, y

        crop_x1, crop_y1, _, _ = self.get_crop_region()
        orig_x = crop_x1 + (x / self.zoom_level)
        orig_y = crop_y1 + (y / self.zoom_level)

        return orig_x, orig_y

    def original_to_display(self, x, y):
        if self.zoom_level == 1.0:
            return x, y

        crop_x1, crop_y1, _, _ = self.get_crop_region()
        disp_x = (x - crop_x1) * self.zoom_level
        disp_y = (y - crop_y1) * self.zoom_level

        return disp_x, disp_y

    def is_box_visible(self, box):
        if self.zoom_level == 1.0:
            return True

        crop_x1, crop_y1, crop_width, crop_height = self.get_crop_region()
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        (x1, y1), (x2, y2) = box
        box_x1, box_x2 = min(x1, x2), max(x1, x2)
        box_y1, box_y2 = min(y1, y2), max(y1, y2)

        return not (box_x2 < crop_x1 or box_x1 > crop_x2 or box_y2 < crop_y1 or box_y1 > crop_y2)

    def mouse_callback(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                orig_x, orig_y = self.display_to_original(x, y)
                self.zoom_center = (orig_x, orig_y)

                zoom_levels = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
                current_idx = zoom_levels.index(self.zoom_level) if self.zoom_level in zoom_levels else 0
                next_idx = (current_idx + 1) % len(zoom_levels)
                self.zoom_level = zoom_levels[next_idx]

                print(f"Zoom: {self.zoom_level}x at ({int(orig_x)}, {int(orig_y)})")
                self.needs_redraw = True
            else:
                self.drawing = True
                orig_x, orig_y = self.display_to_original(x, y)
                self.current_box = [(orig_x, orig_y), (orig_x, orig_y)]
                self.needs_redraw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_box:
                orig_x, orig_y = self.display_to_original(x, y)
                self.current_box[1] = (orig_x, orig_y)
                self.needs_redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_box:
                self.drawing = False
                x1, y1 = self.current_box[0]
                x2, y2 = self.current_box[1]
                disp_x1, disp_y1 = self.original_to_display(x1, y1)
                disp_x2, disp_y2 = self.original_to_display(x2, y2)
                if abs(disp_x2 - disp_x1) > 5 and abs(disp_y2 - disp_y1) > 5:
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

        self.zoom_level = 1.0
        self.zoom_center = (self.img_width / 2, self.img_height / 2)

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
        if self.zoom_level > 1.0:
            crop_x1, crop_y1, crop_width, crop_height = self.get_crop_region()
            x1, y1 = int(crop_x1), int(crop_y1)
            x2, y2 = int(crop_x1 + crop_width), int(crop_y1 + crop_height)

            cropped = self.original_image[y1:y2, x1:x2]
            self.display_image = cv2.resize(cropped, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        else:
            self.display_image = self.original_image.copy()

        for box in self.boxes:
            if self.is_box_visible(box):
                self.draw_box(box, (0, 255, 0), 2)

        if self.current_box:
            if self.is_box_visible(self.current_box):
                self.draw_box(self.current_box, (0, 255, 255), 2)

        mode = "Tennis Ball (Auto-Refine)" if self.refine_tennis_ball else "Standard"
        zoom_info = f" | Zoom: {self.zoom_level}x" if self.zoom_level > 1.0 else ""
        if self.batch_mode:
            instructions = [
                f"Image {self.current_index + 1}/{self.total_images} | Boxes: {len(self.boxes)} | Class: {self.class_id} | Mode: {mode}{zoom_info}",
                f"Processed: {self.processed_count} | Skipped: {self.skipped_count}",
                "Drag: Box | D: Delete | S: Save+Next | N: Skip | P: Previous | R: Reset Zoom | Q: Quit"
            ]
        else:
            instructions = [
                f"Boxes: {len(self.boxes)} | Class ID: {self.class_id} | Mode: {mode}{zoom_info}",
                "Drag: Draw box | D: Delete last | S: Save | R: Reset Zoom | Q: Quit"
            ]
        self.draw_instructions_overlay(instructions)

    def draw_box(self, box, color, thickness):
        (x1, y1), (x2, y2) = box

        disp_x1, disp_y1 = self.original_to_display(x1, y1)
        disp_x2, disp_y2 = self.original_to_display(x2, y2)

        cv2.rectangle(self.display_image, (int(disp_x1), int(disp_y1)), (int(disp_x2), int(disp_y2)), color, thickness)

        cx = (disp_x1 + disp_x2) / 2
        cy = (disp_y1 + disp_y2) / 2
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
        print("Ctrl+Click to zoom (progressive 2x/4x/8x/12x/16x), 'R' to reset zoom")

        if self.batch_mode:
            print("Press 'S' to save+next, 'N' to skip, 'P' for previous, 'D' to delete last, 'R' to reset zoom, 'Q' to quit\n")
        else:
            print("Press 'D' to delete last, 'S' to save, 'R' to reset zoom, 'Q' to quit\n")

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
            elif key == ord('r') or key == ord('R'):
                if self.zoom_level != 1.0:
                    self.zoom_level = 1.0
                    self.zoom_center = (self.img_width / 2, self.img_height / 2)
                    print("Zoom reset to 1.0x")
                    self.needs_redraw = True

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
