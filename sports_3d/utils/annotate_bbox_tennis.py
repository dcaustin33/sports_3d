import argparse

from sports_3d.utils.bbox_annotator_base import BaseBBoxAnnotator
from sports_3d.utils.labeling_utils import refined_tennis_ball_box


class TennisBBoxAnnotator(BaseBBoxAnnotator):
    def __init__(
        self,
        image_path: str,
        class_id: int = 0,
        output_dir: str = None,
        refine_tennis_ball: bool = False,
        refine_current_boxes: bool = False
    ):
        super().__init__(
            image_path=image_path,
            class_id=class_id,
            output_dir=output_dir,
            enable_refinement=refine_tennis_ball,
            refine_current_boxes=refine_current_boxes
        )
        self.refine_tennis_ball = refine_tennis_ball

    def refine_box(self, box: list[tuple]) -> list[tuple]:
        """Apply tennis ball Hough+color detection."""
        if not self.refine_tennis_ball:
            return box

        print("Auto-refining tennis ball box...")
        refined = refined_tennis_ball_box(box, self.original_image)

        if refined != box:
            print("Box auto-refined successfully")
        else:
            print("Auto-refinement unchanged, using original")

        return refined

    def get_refinement_mode_name(self) -> str:
        """Tennis-specific mode name."""
        if self.refine_tennis_ball:
            return "Tennis Ball (Auto-Refine)"
        return "Standard"


def main():
    parser = argparse.ArgumentParser(description="Tennis ball bounding box annotation tool")
    parser.add_argument("image_path", type=str, help="Path to image file or directory")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID for YOLO format (default: 0)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--refine_tennis_ball", action="store_true",
                       help="Enable automatic tennis ball detection and refinement after drawing boxes")
    parser.add_argument("--refine_current_boxes", action="store_true",
                       help="Refine existing annotated boxes instead of skipping annotated images. "
                            "Loads each box sequentially in zoom/refine mode. Can be combined with "
                            "--refine_tennis_ball for automatic refinement.")

    args = parser.parse_args()

    annotator = TennisBBoxAnnotator(
        args.image_path, args.class_id, args.output_dir,
        args.refine_tennis_ball, args.refine_current_boxes
    )
    annotator.run()


if __name__ == "__main__":
    main()
