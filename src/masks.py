"""Which pixels can belong to a part at all: the two foreground gates.

Colour is the cheap gate this dataset allows -- the part is saturated
orange-red on grey or white, so hue and saturation alone find nearly every
part pixel (98 % recall on train). Depth is the colour-blind alternative:
everything standing off the support plane. Neither splits touching parts
into instances; that is the detectors' job (`detect.py`, `detect_seg.py`).
Both are used everywhere a stage needs "part or not": the proposal gate,
the hole polish, the geometric detector, the RGB hole cue.
"""

import cv2
import numpy as np

#: Part pixels are saturated orange-red; cardboard is duller and browner.
SATURATION_MIN = 90
HUE_RED_BELOW, HUE_RED_ABOVE = 15, 170


def part_pixel_mask(bgr: np.ndarray, erode_px: int = 0) -> np.ndarray:
    """Colour gate: every pixel that plausibly belongs to some part.

    ``erode_px`` peels the boundary; leave it at 0 when the true mask edge
    is the point (e.g. for hole-based pose refinement).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    mask = (sat >= SATURATION_MIN) & (
        (hue <= HUE_RED_BELOW) | (hue >= HUE_RED_ABOVE))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    if erode_px:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_px)
    return mask > 0


def foreground_depth_mask(depth: np.ndarray, K: np.ndarray,
                          min_height: float = 0.006) -> np.ndarray:
    """Colour-free foreground: everything standing off the support plane.

    RANSAC-fits the dominant plane of the depth map (the table or tray
    floor) and keeps pixels at least ``min_height`` above it. Works for a
    part of any colour under any lighting; the price is that tray walls
    also survive, which registration's verification then rejects.
    """
    from .scene_io import backproject_pixels

    valid = depth > 0
    points, rows, cols = backproject_pixels(depth, K, valid)
    if len(points) < 1000:
        return np.zeros_like(valid)
    rng = np.random.default_rng(0)
    best_inliers = 0
    best = None
    for _ in range(120):
        trio = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        dist = np.abs((points - trio[0]) @ normal)
        inliers = int((dist < 0.003).sum())
        if inliers > best_inliers:
            best_inliers, best = inliers, (normal, trio[0])
    if best is None:
        return np.zeros_like(valid)
    normal, origin = best
    if normal @ origin > 0:          # orient the normal towards the camera
        normal = -normal
    height = (points - origin) @ normal
    mask = np.zeros_like(valid)
    mask[rows[height > min_height], cols[height > min_height]] = True
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    return mask > 0
