import argparse
from pathlib import Path

import cv2
import numpy as np


class BBoxAnnotator:
    def __init__(self, image_path: str, class_id: int = 0):
        self.image_path = Path(image_path)
        self.class_id = class_id
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.original_image = self.image.copy()
        self.img_height, self.img_width = self.image.shape[:2]

        self.window_width = 1200
        self.window_height = 800

        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.panning = False
        self.pan_start = None

        self.window_name = "BBox Annotator - Press 'H' for help"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom_level
            # zoom_factor = 1.15 if flags > 0 else 1/1.15
            zoom_factor = 1.15
            self.zoom_level *= zoom_factor
            self.zoom_level = max(0.5, min(self.zoom_level, 50.0))

            mouse_img_x = self.offset_x + x / old_zoom
            mouse_img_y = self.offset_y + y / old_zoom

            self.offset_x = mouse_img_x - x / self.zoom_level
            self.offset_y = mouse_img_y - y / self.zoom_level

        elif event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.panning = True
                self.pan_start = (x, y)
            else:
                self.drawing = True
                img_x = self.offset_x + x / self.zoom_level
                img_y = self.offset_y + y / self.zoom_level
                self.current_box = [(img_x, img_y), (img_x, img_y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.panning and self.pan_start:
                dx = (self.pan_start[0] - x) / self.zoom_level
                dy = (self.pan_start[1] - y) / self.zoom_level
                self.offset_x += dx
                self.offset_y += dy
                self.pan_start = (x, y)
            elif self.drawing and self.current_box:
                img_x = self.offset_x + x / self.zoom_level
                img_y = self.offset_y + y / self.zoom_level
                self.current_box[1] = (img_x, img_y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.panning:
                self.panning = False
                self.pan_start = None
            elif self.drawing and self.current_box:
                self.drawing = False
                x1, y1 = self.current_box[0]
                x2, y2 = self.current_box[1]
                if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
                    self.boxes.append(self.current_box)
                    print(f"Box added. Total boxes: {len(self.boxes)}")
                self.current_box = None

    def draw_display(self):
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        view_width = self.window_width / self.zoom_level
        view_height = self.window_height / self.zoom_level

        x1 = max(0, int(self.offset_x))
        y1 = max(0, int(self.offset_y))
        x2 = min(self.img_width, int(self.offset_x + view_width))
        y2 = min(self.img_height, int(self.offset_y + view_height))

        if x2 > x1 and y2 > y1:
            view = self.original_image[y1:y2, x1:x2]

            display_width = int((x2 - x1) * self.zoom_level)
            display_height = int((y2 - y1) * self.zoom_level)

            if display_width > 0 and display_height > 0:
                view_resized = cv2.resize(view, (display_width, display_height),
                                         interpolation=cv2.INTER_LINEAR if self.zoom_level > 1 else cv2.INTER_AREA)

                dest_x = int((x1 - self.offset_x) * self.zoom_level)
                dest_y = int((y1 - self.offset_y) * self.zoom_level)

                if dest_x < self.window_width and dest_y < self.window_height:
                    dest_x2 = min(self.window_width, dest_x + display_width)
                    dest_y2 = min(self.window_height, dest_y + display_height)
                    src_w = dest_x2 - max(0, dest_x)
                    src_h = dest_y2 - max(0, dest_y)

                    if src_w > 0 and src_h > 0:
                        canvas[max(0, dest_y):dest_y2, max(0, dest_x):dest_x2] = view_resized[:src_h, :src_w]

        for box in self.boxes:
            self.draw_box(canvas, box, (0, 255, 0), 2)

        if self.current_box:
            self.draw_box(canvas, self.current_box, (0, 255, 255), 2)

        info_text = [
            f"Boxes: {len(self.boxes)} | Zoom: {self.zoom_level:.2f}x",
            "Scroll: Zoom | Shift+Drag: Pan | Drag: Draw | D: Delete | S: Save | Q: Quit"
        ]
        for i, text in enumerate(info_text):
            cv2.putText(canvas, text, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(canvas, text, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return canvas

    def draw_box(self, canvas, box, color, thickness):
        (x1, y1), (x2, y2) = box

        screen_x1 = int((x1 - self.offset_x) * self.zoom_level)
        screen_y1 = int((y1 - self.offset_y) * self.zoom_level)
        screen_x2 = int((x2 - self.offset_x) * self.zoom_level)
        screen_y2 = int((y2 - self.offset_y) * self.zoom_level)

        cv2.rectangle(canvas, (screen_x1, screen_y1), (screen_x2, screen_y2), color, thickness)

        cx = (screen_x1 + screen_x2) // 2
        cy = (screen_y1 + screen_y2) // 2
        cv2.circle(canvas, (cx, cy), 3, color, -1)

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
        if not self.boxes:
            print("No boxes to save")
            return False

        output_path = self.image_path.parent / f"{self.image_path.stem}_bbox.txt"
        with open(output_path, 'w') as f:
            for box in self.boxes:
                f.write(self.box_to_yolo(box) + '\n')

        print(f"Saved {len(self.boxes)} boxes to {output_path}")
        return True

    def show_help(self):
        help_text = [
            "CONTROLS:",
            "  Drag mouse: Draw bounding box",
            "  Scroll wheel: Zoom in/out (centers on mouse)",
            "  Shift + Drag: Pan view",
            "  D: Delete last box",
            "  S: Save all boxes and exit",
            "  Q: Quit without saving",
            "  R: Reset view",
            "  H: Show this help",
        ]
        print("\n" + "\n".join(help_text))

    def run(self):
        self.show_help()

        while True:
            display = self.draw_display()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit without saving")
                break
            elif key == ord('s'):
                if self.save_annotations():
                    break
            elif key == ord('d'):
                if self.boxes:
                    self.boxes.pop()
                    print(f"Deleted last box. Remaining: {len(self.boxes)}")
            elif key == ord('r'):
                self.zoom_level = 1.0
                self.offset_x = 0
                self.offset_y = 0
                print("View reset")
            elif key == ord('h'):
                self.show_help()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Bounding box annotation tool with zoom support')
    parser.add_argument('image_path', help='Path to the image to annotate')
    parser.add_argument('--class-id', type=int, default=0, help='Class ID for YOLO format (default: 0)')

    args = parser.parse_args()

    try:
        annotator = BBoxAnnotator(args.image_path, args.class_id)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
