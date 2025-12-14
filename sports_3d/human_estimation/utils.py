# your_repo/utils/sam3d.py
import sys
from pathlib import Path

_SAM3D_PATH = Path(__file__).resolve().parent.parent.parent / "external" / "sam-3d-body"
sys.path.insert(0, str(_SAM3D_PATH))

# Re-export what you need
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

__all__ = [
    "setup_sam_3d_body",
    "visualize_sample_together",
    "load_sam_3d_body",
    "SAM3DBodyEstimator",
]
