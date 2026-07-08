"""Monocular depth estimation via Depth Anything V2 (metric, indoor).

The metric-indoor checkpoint outputs depth directly in metres, which is a
better starting point than the relative model because the per-frame scale is
already roughly consistent. We still align each frame to COLMAP's sparse
depths during fusion (COLMAP's absolute scale is arbitrary), but starting from
metric depth makes that alignment more stable.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Alternatives if this 404s or is too heavy for 8 GB VRAM:
#   depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf   (lightest)
#   depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf   (best, heaviest)
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


class DepthEstimator:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[depth] loading {model_id} on {self.device} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def estimate(self, image_path: str) -> np.ndarray:
        """Return a HxW float32 metric depth map (metres), matching the input
        image's resolution.
        """
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # predicted_depth is (1, H', W') at the model's internal resolution;
        # resize back to the original image size.
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return depth.detach().cpu().numpy().astype(np.float32)
