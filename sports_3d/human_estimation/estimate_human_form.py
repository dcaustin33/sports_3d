import pickle
from sports_3d.human_estimation.utils import (
    load_sam_3d_body,
    visualize_sample_together,
    setup_sam_3d_body,
)
import cv2
import numpy as np


if __name__ == "__main__":
    device = "cpu"
    model_path = "/Users/derek/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/11aaa346c7204874a1cbafe3d39a979080b2c55a/model.ckpt"
    mhr_path = "/Users/derek/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/11aaa346c7204874a1cbafe3d39a979080b2c55a/assets/mhr_model.pt"

    estimator = setup_sam_3d_body(
        hf_repo_id="facebook/sam-3d-body-dinov3", device=device
    )
    # Load and process image
    img_bgr = cv2.imread(
        "/Users/derek/Desktop/sports_3d/data/sinner_ruud_frames/frame_004140_t69.000s.png"
    )

    bboxes = np.array(
        [
            [3.41126160e02, 6.23878296e02, 1.21459619e03, 2.11426514e03],
            [2.43352441e03, 1.99211868e02, 2.62849878e03, 5.31113159e02],
        ]
    )

    outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bboxes=bboxes)
    with open("outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
