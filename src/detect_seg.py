"""Registration driven by learned instance masks.

A fine-tuned segmenter replaces the geometric pile-splitting: each
predicted mask is registered exactly like an oracle mask. The geometric
verification stack stays -- learned masks propose, depth physics disposes.
"""

import cv2
import numpy as np

from .detect import MIN_CONFIDENCE, nms
from .register import PoseEstimator
from .scene_io import Scene, backproject

#: Boundary peel for predicted masks, pixels (mask edges carry blended depth).
ERODE_PX = 2

#: Predicted masks smaller than this cannot hold a registrable instance.
MIN_MASK_PX = 1200


def detect_from_masks(scene: Scene, estimator: PoseEstimator,
                      masks: list) -> list:
    """Register one scene's learned instance masks.

    Args:
        scene: The capture.
        estimator: Prepared registration stack for this scene.
        masks: [(bool_mask, seg_confidence)], any order.

    Returns:
        Verified pose estimates, deduplicated.
    """
    found = []
    for mask, _ in sorted(masks, key=lambda m: -m[1]):
        if mask.sum() < MIN_MASK_PX:
            continue
        slim = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=ERODE_PX).astype(bool)
        if slim.sum() < MIN_MASK_PX // 2:
            slim = mask
        points = backproject(scene.depth, scene.K, slim)
        if len(points) < 300:
            continue
        anchor = points[np.argmin(points[:, 2])]
        est = estimator.estimate(points, attempts=3, anchor=anchor)
        if est is None or est.confidence < MIN_CONFIDENCE:
            continue
        found.append(est)
    return nms(found)


def masks_from_model(model, bgr: np.ndarray, conf: float = 0.4) -> list:
    """Run an ultralytics segmentation model; returns [(bool_mask, conf)]."""
    H, W = bgr.shape[:2]
    result = model(bgr, imgsz=960, conf=conf, verbose=False,
                   retina_masks=True)[0]
    out = []
    if result.masks is None:
        return out
    for m, c in zip(result.masks.data.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()):
        mask = cv2.resize(m.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST) > 0
        out.append((mask, float(c)))
    return out
