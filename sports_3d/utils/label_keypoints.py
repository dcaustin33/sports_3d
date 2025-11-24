import argparse
from pathlib import Path

import cv2
import numpy as np


class KeypointLabeler:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.display_image = self.image.copy()
        self.keypoints = []
        self.window_name = "Keypoint Labeler"
        self.pending_point = None

    def draw_keypoints(self):
        self.display_image = self.image.copy()

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

        self.draw_instructions()

    def draw_instructions(self):
        instructions = [
            "Click: Mark keypoint",
            "D: Delete last | S: Save | Q: Quit"
        ]
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display_image,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                self.display_image,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_point = (x, y)
            self.draw_keypoints()
            cv2.imshow(self.window_name, self.display_image)

            while True:
                print(f"\nClicked point at ({x}, {y})")
                idx_input = input("Enter keypoint index (0-13) or 'c' to cancel: ").strip()

                if idx_input.lower() == 'c':
                    self.pending_point = None
                    break

                try:
                    idx = int(idx_input)
                    if 0 <= idx <= 13:
                        self.keypoints.append((idx, x, y))
                        print(f"Added keypoint {idx} at ({x}, {y})")
                        self.pending_point = None
                        break
                    else:
                        print("Invalid index. Must be between 0 and 13.")
                except ValueError:
                    print("Invalid input. Enter a number between 0 and 13.")

            self.draw_keypoints()
            cv2.imshow(self.window_name, self.display_image)

    def delete_last(self):
        if self.keypoints:
            removed = self.keypoints.pop()
            print(f"Deleted keypoint {removed[0]} at ({removed[1]}, {removed[2]})")
            self.draw_keypoints()
            cv2.imshow(self.window_name, self.display_image)
        else:
            print("No keypoints to delete")

    def save(self):
        output_path = self.image_path.parent / f"{self.image_path.stem}_keypoints.txt"

        with open(output_path, 'w') as f:
            for idx, x, y in self.keypoints:
                f.write(f"{idx} {x} {y}\n")

        print(f"\nSaved {len(self.keypoints)} keypoints to {output_path}")
        return output_path

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.draw_keypoints()
        cv2.imshow(self.window_name, self.display_image)

        print(f"\nLabeling image: {self.image_path}")
        print("Click on keypoints and enter their index (0-13)")
        print("Press 'D' to delete last, 'S' to save, 'Q' to quit\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("Quitting without saving")
                break
            elif key == ord('d') or key == ord('D'):
                self.delete_last()
            elif key == ord('s') or key == ord('S'):
                self.save()
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Label tennis court keypoints on an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    labeler = KeypointLabeler(args.image_path)
    labeler.run()


if __name__ == "__main__":
    main()
