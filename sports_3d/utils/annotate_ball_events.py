import argparse
from pathlib import Path

import cv2

from sports_3d.utils.annotation_base import BaseAnnotator


class BallEventAnnotator(BaseAnnotator):
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
        annotation_file = output_dir / f"{image_path.stem}_events.txt"
        return annotation_file.exists()

    def __init__(self, image_path: str, output_dir: str = None):
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

        self.events = []
        self.pending_event = None
        self.waiting_for_event_type = False
        self.waiting_for_player_pos = False
        self.mouse_x = 0
        self.mouse_y = 0

        self.COLOR_GROUND = (255, 0, 0)
        self.COLOR_RACQUET = (0, 255, 0)
        self.COLOR_PLAYER = (0, 165, 255)
        self.COLOR_PENDING = (0, 255, 255)

        self.load_image(self.image_queue[self.current_index])

    def load_existing_annotations(self, image_path: Path):
        annotation_file = self.output_dir / f"{image_path.stem}_events.txt"
        if not annotation_file.exists():
            return []

        events = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0] == 'ground':
                    events.append({
                        'type': 'ground',
                        'ball_x': int(parts[1]),
                        'ball_y': int(parts[2])
                    })
                elif len(parts) == 5 and parts[0] == 'racquet':
                    events.append({
                        'type': 'racquet',
                        'ball_x': int(parts[1]),
                        'ball_y': int(parts[2]),
                        'player_x': int(parts[3]),
                        'player_y': int(parts[4])
                    })
        return events

    def load_image(self, image_path: Path):
        self.image_path = image_path
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.original_image = self.image.copy()
        self.display_image = self.image.copy()
        self.img_height, self.img_width = self.image.shape[:2]

        self.events = []
        self.pending_event = None
        self.waiting_for_event_type = False
        self.waiting_for_player_pos = False
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

    def draw_event_type_prompt(self):
        h, w = self.display_image.shape[:2]
        overlay = self.display_image.copy()

        box_height = 120
        box_width = 500
        box_x = (w - box_width) // 2
        box_y = (h - box_height) // 2

        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
        self.display_image = cv2.addWeighted(overlay, 0.8, self.display_image, 0.2, 0)

        cv2.rectangle(self.display_image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)

        prompt_text = "Select event type:"
        options_text = "G = Ground | R = Racquet | ESC = Cancel"

        cv2.putText(
            self.display_image,
            prompt_text,
            (box_x + 20, box_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            self.display_image,
            options_text,
            (box_x + 20, box_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    def draw_player_prompt(self):
        h, w = self.display_image.shape[:2]
        overlay = self.display_image.copy()

        box_height = 100
        box_width = 450
        box_x = (w - box_width) // 2
        box_y = (h - box_height) // 2

        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
        self.display_image = cv2.addWeighted(overlay, 0.8, self.display_image, 0.2, 0)

        cv2.rectangle(self.display_image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)

        prompt_text = "Click player position (ESC = Cancel)"

        cv2.putText(
            self.display_image,
            prompt_text,
            (box_x + 20, box_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    def draw_crosshair(self, x, y, color, size=20):
        cv2.line(self.display_image, (x - size, y), (x + size, y), color, 2)
        cv2.line(self.display_image, (x, y - size), (x, y + size), color, 2)

    def draw_events(self):
        self.display_image = self.original_image.copy()

        for event in self.events:
            ball_x, ball_y = event['ball_x'], event['ball_y']

            if event['type'] == 'ground':
                cv2.circle(self.display_image, (ball_x, ball_y), 8, self.COLOR_GROUND, -1)
                cv2.putText(
                    self.display_image,
                    "G",
                    (ball_x + 12, ball_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLOR_GROUND,
                    2
                )
            else:
                player_x, player_y = event['player_x'], event['player_y']
                cv2.circle(self.display_image, (ball_x, ball_y), 8, self.COLOR_RACQUET, -1)
                cv2.circle(self.display_image, (player_x, player_y), 8, self.COLOR_PLAYER, -1)
                cv2.line(self.display_image, (ball_x, ball_y), (player_x, player_y), self.COLOR_PLAYER, 2)
                cv2.putText(
                    self.display_image,
                    "R",
                    (ball_x + 12, ball_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLOR_RACQUET,
                    2
                )

        if self.pending_event:
            ball_x, ball_y = self.pending_event['ball_x'], self.pending_event['ball_y']
            cv2.circle(self.display_image, (ball_x, ball_y), 8, self.COLOR_PENDING, -1)

        if self.waiting_for_player_pos:
            self.draw_crosshair(self.mouse_x, self.mouse_y, self.COLOR_PLAYER)

        if self.batch_mode:
            instructions = [
                f"Image {self.current_index + 1}/{self.total_images} | Events: {len(self.events)} | Processed: {self.processed_count} | Skipped: {self.skipped_count}",
                "Click: Ball position | D: Delete last | S: Save+Next | N: Skip | P: Previous | Q: Quit"
            ]
        else:
            instructions = [
                f"Events: {len(self.events)}",
                "Click: Ball position | D: Delete last | S: Save | Q: Quit"
            ]
        self.draw_instructions_overlay(instructions)

        if self.waiting_for_event_type:
            self.draw_event_type_prompt()
        elif self.waiting_for_player_pos:
            self.draw_player_prompt()

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            if self.waiting_for_player_pos:
                self.needs_redraw = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.waiting_for_player_pos:
                self.pending_event['player_x'] = x
                self.pending_event['player_y'] = y
                self.events.append(self.pending_event)
                print(f"Added racquet event at ball ({self.pending_event['ball_x']}, {self.pending_event['ball_y']}) with player at ({x}, {y})")
                self.pending_event = None
                self.waiting_for_player_pos = False
                self.needs_redraw = True

            elif not self.waiting_for_event_type:
                self.pending_event = {'ball_x': x, 'ball_y': y}
                self.waiting_for_event_type = True
                print(f"Clicked ball at ({x}, {y})")
                self.needs_redraw = True

    def delete_last(self):
        if self.events:
            removed = self.events.pop()
            event_type = removed['type']
            print(f"Deleted last {event_type} event. Remaining: {len(self.events)}")
            self.needs_redraw = True
        else:
            print("No events to delete")

    def format_event(self, event):
        if event['type'] == 'ground':
            return f"ground {event['ball_x']} {event['ball_y']}"
        else:
            return f"racquet {event['ball_x']} {event['ball_y']} {event['player_x']} {event['player_y']}"

    def save_annotations(self):
        return self.save_to_file(self.events, "_events.txt", self.format_event, "event")

    def handle_input(self, key):
        if self.waiting_for_event_type:
            if key == ord('g') or key == ord('G'):
                self.pending_event['type'] = 'ground'
                self.events.append(self.pending_event)
                print(f"Added ground event at ({self.pending_event['ball_x']}, {self.pending_event['ball_y']})")
                self.pending_event = None
                self.waiting_for_event_type = False
                self.needs_redraw = True

            elif key == ord('r') or key == ord('R'):
                self.pending_event['type'] = 'racquet'
                self.waiting_for_event_type = False
                self.waiting_for_player_pos = True
                print("Racquet event selected. Click player position...")
                self.needs_redraw = True

            elif key == 27:
                self.pending_event = None
                self.waiting_for_event_type = False
                print("Cancelled")
                self.needs_redraw = True

        elif self.waiting_for_player_pos:
            if key == 27:
                self.pending_event = None
                self.waiting_for_player_pos = False
                print("Cancelled racquet event")
                self.needs_redraw = True

        else:
            if key == ord('q') or key == ord('Q'):
                if self.batch_mode:
                    self.print_batch_summary()
                else:
                    print("Quitting without saving")
                return False

            elif key == ord('d') or key == ord('D'):
                self.delete_last()

            elif key == ord('s') or key == ord('S'):
                if self.save_annotations():
                    if self.batch_mode:
                        self.processed_count += 1
                        if not self.load_next_image():
                            self.print_batch_summary()
                            return False
                    else:
                        return False

            elif key == ord('n') or key == ord('N'):
                if self.batch_mode:
                    self.skipped_count += 1
                    if not self.load_next_image():
                        self.print_batch_summary()
                        return False

            elif key == ord('p') or key == ord('P'):
                if self.batch_mode:
                    self.load_previous_image()

        return True

    def run(self):
        self.setup_window("Ball Event Annotator", self.mouse_callback)

        if self.batch_mode:
            print(f"\nBatch mode: Found {self.total_images} images")
            print(f"Already annotated: {self.already_annotated_count} (auto-skipped)")
            print(f"To process: {self.total_images - self.already_annotated_count} images\n")
            print(f"Annotating image {self.current_index + 1}/{self.total_images}: {self.image_path.name}")
        else:
            print(f"\nAnnotating image: {self.image_path}")

        print("\nClick on ball location, then:")
        print("  G = Ground contact")
        print("  R = Racquet contact (then click player position)")
        print("\nKeyboard shortcuts:")
        if self.batch_mode:
            print("  S = Save and next | N = Skip | P = Previous | D = Delete last | Q = Quit\n")
        else:
            print("  S = Save | D = Delete last | Q = Quit\n")

        while True:
            if self.needs_redraw:
                self.draw_events()
                self.show_image()
                self.needs_redraw = False

            key = cv2.waitKey(20) & 0xFF

            if key != 255:
                if not self.handle_input(key):
                    break

        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Tennis ball event annotation tool")
    parser.add_argument("image_path", type=str, help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")

    args = parser.parse_args()

    annotator = BallEventAnnotator(args.image_path, args.output_dir)
    annotator.run()


if __name__ == "__main__":
    main()
