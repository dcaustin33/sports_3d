from pathlib import Path

import cv2

from sports_3d.utils.annotation_base import BaseAnnotator


class BaseBBoxAnnotator(BaseAnnotator):
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

    def __init__(self, image_path: str, class_id: int = 0, output_dir: str = None, enable_refinement: bool = False, refine_current_boxes: bool = False):
        input_path = Path(image_path)

        self.output_dir = Path(output_dir) if output_dir else (input_path if input_path.is_dir() else input_path.parent)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.refine_current_boxes = refine_current_boxes

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

        if self.refine_current_boxes:
            for i, img_path in enumerate(self.image_queue):
                if self._is_already_annotated(img_path, self.output_dir):
                    self.current_index = i
                    break
            else:
                if self.is_directory:
                    print("No annotated images found to refine")
                    raise SystemExit(0)
        else:
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
        self.enable_refinement = enable_refinement

        self.zoom_level = 1.0
        self.zoom_center = (self.img_width / 2, self.img_height / 2)

        self.in_refine_mode = False
        self.refine_temp_box = None
        self.refine_original_zoom = None
        self.refine_original_zoom_center = None

        self.refining_existing_boxes = False
        self.box_refine_queue = []
        self.current_refine_index = 0

        self.load_image(self.image_queue[self.current_index])

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

    def yolo_to_box(self, yolo_line: str):
        parts = yolo_line.strip().split()
        if len(parts) != 5:
            return None

        _, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        cx_px = cx * self.img_width
        cy_px = cy * self.img_height
        w_px = w * self.img_width
        h_px = h * self.img_height

        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2

        return [(x1, y1), (x2, y2)]

    def load_existing_annotations(self, image_path: Path):
        annotation_file = self.output_dir / f"{image_path.stem}_bbox.txt"
        if not annotation_file.exists():
            return []

        boxes = []
        with open(annotation_file, 'r') as f:
            for line in f:
                box = self.yolo_to_box(line)
                if box:
                    boxes.append(box)

        return boxes

    def zoom_to_box(self, box, expansion_factor=1.3):
        (x1, y1), (x2, y2) = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        box_width = (x2 - x1) * expansion_factor
        box_height = (y2 - y1) * expansion_factor

        zoom_x = self.img_width / box_width
        zoom_y = self.img_height / box_height
        zoom_level = min(zoom_x, zoom_y)

        zoom_level = max(1.0, min(zoom_level, 16.0))

        return zoom_level, (cx, cy)

    def refine_box(self, box: list[tuple]) -> list[tuple]:
        """
        Hook: Override to implement automatic refinement logic.

        Called when:
        - User finishes drawing a box (if enable_refinement=True)
        - Iterating through existing boxes for refinement

        Args:
            box: Original box coordinates [(x1,y1), (x2,y2)]

        Returns:
            Refined box (can return original if no refinement possible)

        Default: Returns box unchanged (no-op)
        """
        return box

    def on_enter_refine_mode(self, box: list[tuple]):
        """
        Hook: Called when entering refine mode.
        Use for custom logging or UI updates.
        Default: pass

        Args:
            box: The box being refined
        """
        pass

    def on_exit_refine_mode(self, accept: bool):
        """
        Hook: Called when exiting refine mode.

        Args:
            accept: True if user accepted refined box
        """
        _ = accept  # Unused in base class, available for subclass hooks

    def get_refinement_mode_name(self) -> str:
        """
        Hook: Return human-readable name for refinement mode.
        Used in UI instructions overlay.
        Default: "Auto-Refine" if enable_refinement else "Standard"
        """
        return "Auto-Refine" if self.enable_refinement else "Standard"

    def enter_refine_mode(self, box):
        self.refine_original_zoom = self.zoom_level
        self.refine_original_zoom_center = self.zoom_center
        self.refine_temp_box = box
        self.in_refine_mode = True

        new_zoom, new_center = self.zoom_to_box(box, expansion_factor=1.3)
        self.zoom_level = new_zoom
        self.zoom_center = new_center

        self.on_enter_refine_mode(box)
        print(f"Refine mode: Draw new box or press 'S' to accept, 'Q' to discard")

    def exit_refine_mode(self, accept=False):
        if accept and self.refine_temp_box:
            self.boxes.append(self.refine_temp_box)
            print(f"Box accepted. Total boxes: {len(self.boxes)}")
        else:
            if self.refining_existing_boxes:
                original_box = self.box_refine_queue[self.current_refine_index]
                self.boxes.append(original_box)
                print("Box discarded, kept original")
            else:
                print("Box discarded")

        self.zoom_level = self.refine_original_zoom
        self.zoom_center = self.refine_original_zoom_center

        self.refine_temp_box = None
        self.in_refine_mode = False
        self.refine_original_zoom = None
        self.refine_original_zoom_center = None

        self.needs_redraw = True

        self.on_exit_refine_mode(accept)

        if self.refining_existing_boxes:
            self.current_refine_index += 1
            if not self.refine_next_box_in_queue():
                pass

    def start_refining_existing_boxes(self):
        if not self.boxes:
            print("No boxes to refine")
            return False

        self.refining_existing_boxes = True
        self.box_refine_queue = self.boxes.copy()
        self.current_refine_index = 0
        self.boxes = []

        return self.refine_next_box_in_queue()

    def refine_next_box_in_queue(self):
        if self.current_refine_index >= len(self.box_refine_queue):
            self.refining_existing_boxes = False
            print(f"All {len(self.box_refine_queue)} boxes refined")
            return False

        box = self.box_refine_queue[self.current_refine_index]

        box = self.refine_box(box)

        self.enter_refine_mode(box)
        print(f"Refining box {self.current_refine_index + 1}/{len(self.box_refine_queue)}")
        return True

    def mouse_callback(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.in_refine_mode:
                    print("Manual zoom disabled in refine mode")
                    return
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

                    if self.in_refine_mode:
                        self.refine_temp_box = box_to_add
                        print(f"Box replaced in refine mode (manual correction, no auto-refinement)")
                    else:
                        if self.enable_refinement:
                            box_to_add = self.refine_box(self.current_box)
                            self.enter_refine_mode(box_to_add)
                        else:
                            self.boxes.append(box_to_add)
                            print(f"Box added. Total boxes: {len(self.boxes)}")
                self.current_box = None
                self.needs_redraw = True

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

    def draw_display(self):
        if self.zoom_level > 1.0:
            crop_x1, crop_y1, crop_width, crop_height = self.get_crop_region()
            x1, y1 = int(crop_x1), int(crop_y1)
            x2, y2 = int(crop_x1 + crop_width), int(crop_y1 + crop_height)

            cropped = self.original_image[y1:y2, x1:x2]
            self.display_image = cv2.resize(cropped, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        else:
            self.display_image = self.original_image.copy()

        if self.in_refine_mode:
            if self.refine_temp_box and self.is_box_visible(self.refine_temp_box):
                self.draw_box(self.refine_temp_box, (0, 255, 255), 3)

            if self.current_box and self.is_box_visible(self.current_box):
                self.draw_box(self.current_box, (255, 255, 0), 2)

            if self.refining_existing_boxes:
                instructions = [
                    f"REFINE MODE | Box {self.current_refine_index + 1}/{len(self.box_refine_queue)} | Zoom: {self.zoom_level:.1f}x",
                    "Drag: Redraw box | S: Accept & Next | Q: Keep original & Next"
                ]
            else:
                instructions = [
                    f"REFINE MODE | Zoom: {self.zoom_level:.1f}x",
                    "Drag: Redraw box | S: Accept | Q: Discard"
                ]
        else:
            for box in self.boxes:
                if self.is_box_visible(box):
                    self.draw_box(box, (0, 255, 0), 2)

            if self.current_box:
                if self.is_box_visible(self.current_box):
                    self.draw_box(self.current_box, (0, 255, 255), 2)

            mode = self.get_refinement_mode_name()
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

    def load_image(self, image_path: Path):
        self.image_path = image_path
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.original_image = self.image.copy()
        self.display_image = self.image.copy()
        self.img_height, self.img_width = self.image.shape[:2]

        if self.refine_current_boxes:
            self.boxes = self.load_existing_annotations(image_path)
            print(f"Loaded {len(self.boxes)} existing boxes")
        else:
            self.boxes = []

        self.current_box = None
        self.drawing = False
        self.needs_redraw = True

        self.zoom_level = 1.0
        self.zoom_center = (self.img_width / 2, self.img_height / 2)

        self.in_refine_mode = False
        self.refine_temp_box = None
        self.refine_original_zoom = None
        self.refine_original_zoom_center = None

        if self.batch_mode:
            print(f"\nAnnotating image {self.current_index + 1}/{self.total_images}: {self.image_path.name}")

        if self.refine_current_boxes and self.boxes:
            self.start_refining_existing_boxes()

    def load_next_image(self) -> bool:
        for i in range(self.current_index + 1, self.total_images):
            if self.refine_current_boxes:
                if self._is_already_annotated(self.image_queue[i], self.output_dir):
                    self.current_index = i
                    self.load_image(self.image_queue[i])
                    return True
            else:
                if not self._is_already_annotated(self.image_queue[i], self.output_dir):
                    self.current_index = i
                    self.load_image(self.image_queue[i])
                    return True
        return False

    def load_previous_image(self) -> bool:
        for i in range(self.current_index - 1, -1, -1):
            if self.refine_current_boxes:
                if self._is_already_annotated(self.image_queue[i], self.output_dir):
                    self.current_index = i
                    self.load_image(self.image_queue[i])
                    return True
            else:
                if not self._is_already_annotated(self.image_queue[i], self.output_dir):
                    self.current_index = i
                    self.load_image(self.image_queue[i])
                    return True
        print("Already at first unannotated image" if not self.refine_current_boxes else "Already at first annotated image")
        return False

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

        if self.refine_current_boxes:
            print("Mode: REFINE EXISTING BOXES")
            print("  - Each box will be shown sequentially for review/refinement")

        if not self.refine_current_boxes:
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
                if self.in_refine_mode:
                    self.exit_refine_mode(accept=False)
                else:
                    if self.batch_mode:
                        self.print_batch_summary()
                    else:
                        print("Quitting without saving")
                    break
            elif key == ord('s') or key == ord('S'):
                if self.in_refine_mode:
                    self.exit_refine_mode(accept=True)
                else:
                    if self.save_annotations():
                        if self.batch_mode:
                            self.processed_count += 1
                            if not self.load_next_image():
                                self.print_batch_summary()
                                break
                        else:
                            break
            elif key == ord('n') or key == ord('N'):
                if self.in_refine_mode:
                    print("Skip disabled in refine mode")
                elif self.batch_mode:
                    self.skipped_count += 1
                    if not self.load_next_image():
                        self.print_batch_summary()
                        break
            elif key == ord('p') or key == ord('P'):
                if self.in_refine_mode:
                    print("Previous disabled in refine mode")
                elif self.batch_mode:
                    self.load_previous_image()
            elif key == ord('d') or key == ord('D'):
                if self.in_refine_mode:
                    print("Delete disabled in refine mode")
                else:
                    self.delete_last()
            elif key == ord('r') or key == ord('R'):
                if self.in_refine_mode:
                    print("Reset zoom disabled in refine mode")
                elif self.zoom_level != 1.0:
                    self.zoom_level = 1.0
                    self.zoom_center = (self.img_width / 2, self.img_height / 2)
                    print("Zoom reset to 1.0x")
                    self.needs_redraw = True

        self.cleanup()
