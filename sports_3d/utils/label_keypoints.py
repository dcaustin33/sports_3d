import argparse

import cv2

from sports_3d.utils.annotation_base import BaseAnnotator


class KeypointLabeler(BaseAnnotator):
    def __init__(self, image_path: str, output_dir: str = None):
        super().__init__(image_path, output_dir)
        self.keypoints = []
        self.pending_point = None
        self.input_buffer = ""
        self.waiting_for_input = False

    def draw_keypoints(self):
        self.display_image = self.original_image.copy()

        for idx, x, y in self.keypoints:
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self.display_image,
                str(idx),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if self.pending_point:
            x, y = self.pending_point
            cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)

        instructions = [
            "Click: Mark keypoint",
            "D: Delete last | S: Save | Q: Quit"
        ]
        self.draw_instructions_overlay(instructions)

        if self.waiting_for_input:
            self.draw_input_prompt()

    def draw_input_prompt(self):
        h, w = self.display_image.shape[:2]
        overlay = self.display_image.copy()

        box_height = 100
        box_width = 400
        box_x = (w - box_width) // 2
        box_y = (h - box_height) // 2

        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
        self.display_image = cv2.addWeighted(overlay, 0.8, self.display_image, 0.2, 0)

        cv2.rectangle(self.display_image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)

        prompt_text = "Enter index (0-13):"
        input_text = f"> {self.input_buffer}_"

        cv2.putText(
            self.display_image,
            prompt_text,
            (box_x + 20, box_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(
            self.display_image,
            input_text,
            (box_x + 20, box_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.waiting_for_input:
            self.pending_point = (x, y)
            self.waiting_for_input = True
            self.input_buffer = ""
            print(f"\nClicked point at ({x}, {y})")
            self.draw_keypoints()
            self.show_image()

    def delete_last(self):
        if self.keypoints:
            removed = self.keypoints.pop()
            print(f"Deleted keypoint {removed[0]} at ({removed[1]}, {removed[2]})")
            self.draw_keypoints()
            self.show_image()
        else:
            print("No keypoints to delete")

    def format_keypoint(self, keypoint):
        idx, x, y = keypoint
        return f"{idx} {x} {y}"

    def save(self):
        if not self.keypoints:
            print("No keypoints to save")
            return False
        return self.save_to_file(self.keypoints, "_keypoints.txt", self.format_keypoint, "keypoint")

    def handle_input(self, key):
        if self.waiting_for_input:
            if key == 13:
                if self.input_buffer:
                    try:
                        idx = int(self.input_buffer)
                        if 0 <= idx <= 13:
                            x, y = self.pending_point
                            self.keypoints.append((idx, x, y))
                            print(f"Added keypoint {idx} at ({x}, {y})")
                            self.pending_point = None
                            self.waiting_for_input = False
                            self.input_buffer = ""
                        else:
                            print("Invalid index. Must be between 0 and 13.")
                            self.input_buffer = ""
                    except ValueError:
                        print("Invalid input. Enter a number between 0 and 13.")
                        self.input_buffer = ""
            elif key == 27:
                self.pending_point = None
                self.waiting_for_input = False
                self.input_buffer = ""
                print("Cancelled")
            elif key == 8 or key == 127:
                self.input_buffer = self.input_buffer[:-1]
            elif 48 <= key <= 57:
                if len(self.input_buffer) < 2:
                    self.input_buffer += chr(key)
        else:
            if key == ord('q') or key == ord('Q'):
                print("Quitting without saving")
                return False
            elif key == ord('d') or key == ord('D'):
                self.delete_last()
            elif key == ord('s') or key == ord('S'):
                if self.save():
                    return False
        return True

    def run(self):
        self.setup_window("Keypoint Labeler", self.mouse_callback)

        self.draw_keypoints()
        self.show_image()

        print(f"\nLabeling image: {self.image_path}")
        print("Click on keypoints and enter their index (0-13)")
        print("Press 'D' to delete last, 'S' to save, 'Q' to quit\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key != 255:
                if not self.handle_input(key):
                    break

            self.draw_keypoints()
            self.show_image()

        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Label tennis court keypoints on an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")

    args = parser.parse_args()

    labeler = KeypointLabeler(args.image_path, args.output_dir)
    labeler.run()


if __name__ == "__main__":
    main()
