"""Scene loading and depth back-projection.

Every function here follows the dataset conventions in ASSIGNMENT.md: poses are
T_camera_object, the camera frame is OpenCV (+X right, +Y down, +Z forward),
depth is registered to the colour image and given in millimetre ticks scaled
by ``camera.json["depth_scale"]``.
"""

import glob
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Scene:
    """One RGB-D capture and its calibration."""

    scene_id: str
    rgb: np.ndarray            # (H, W, 3) BGR uint8
    depth: np.ndarray          # (H, W) float64 metres, 0 = no measurement
    K: np.ndarray              # (3, 3) intrinsics
    mask_paths: list = field(default_factory=list)   # GT masks, when present


def list_scenes(root: str, split: str) -> list:
    """Sorted scene ids under ``root/split``."""
    split_dir = os.path.join(root, split)
    return sorted(s for s in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, s)))


def load_scene(root: str, split: str, scene_id: str) -> Scene:
    """Load one scene's image, depth (in metres) and intrinsics."""
    d = os.path.join(root, split, scene_id)
    rgb = cv2.imread(os.path.join(d, "rgb.png"))
    if rgb is None:
        raise FileNotFoundError(os.path.join(d, "rgb.png"))
    with open(os.path.join(d, "camera.json")) as fh:
        cam = json.load(fh)
    depth_raw = cv2.imread(os.path.join(d, "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(os.path.join(d, "depth.png"))
    depth = depth_raw.astype(np.float64) * cam["depth_scale"]
    return Scene(
        scene_id=scene_id,
        rgb=rgb,
        depth=depth,
        K=np.asarray(cam["K"], dtype=np.float64),
        mask_paths=sorted(glob.glob(os.path.join(d, "masks", "*.png"))),
    )


def backproject(depth: np.ndarray, K: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Lift masked depth pixels into camera-frame 3D points.

    Args:
        depth: (H, W) metres, 0 = no measurement.
        K: (3, 3) pinhole intrinsics; images are rectified, distortion is zero.
        mask: (H, W) bool, pixels to lift.

    Returns:
        (N, 3) float64 points; pixels without a depth measurement are dropped.
    """
    points, _, _ = backproject_pixels(depth, K, mask)
    return points


def backproject_pixels(depth: np.ndarray, K: np.ndarray, mask: np.ndarray):
    """Like :func:`backproject`, also returning each point's pixel row/col.

    Returns:
        ``(points, rows, cols)`` -- callers that carve up an organised cloud
        (per-pixel bookkeeping) need to map points back to pixels.
    """
    ys, xs = np.nonzero(mask & (depth > 0))
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.column_stack([x, y, z]), ys, xs
