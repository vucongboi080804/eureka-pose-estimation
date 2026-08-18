"""Run the pipeline on frames from a live camera instead of a release folder.

The pipeline only ever sees a :class:`src.scene_io.Scene` (BGR image, depth
in metres, intrinsics), so a camera SDK plugs in at exactly one point: hand
:func:`scene_from_arrays` the registered colour frame, the raw depth frame
and the calibration it reports, then call :func:`estimate_scene` -- the same
constructor and detector calls scripts/run_pipeline.py makes per scene.

    from deploy.pose.adapter import load_models, scene_from_arrays, estimate_scene
    model_cloud, seg, extra = load_models("model/3d_model.ply",
                                          "weights/part-seg.pt",
                                          "weights/part-seg-synthetic.pt")
    # per cycle: rgb_bgr (H, W, 3) uint8, depth_raw (H, W) uint16, from the SDK
    scene = scene_from_arrays(rgb_bgr, depth_raw, K, depth_scale)
    poses = estimate_scene(scene, model_cloud, seg, extra, pick=True)

Requirements on the camera stream match ASSIGNMENT.md: depth registered to the
colour image (same resolution, same pixel grid), pinhole intrinsics ``K``
of that colour image, no lens distortion, ``depth_scale`` turning raw depth
ticks into metres (0 = no measurement). Poses come back as T_camera_object
in the submission's conventions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.masks import part_pixel_mask
from src.detect_seg import detect_scene_hybrid
from src.model_cloud import load_model_cloud
from src.register import PoseEstimator
from src.scene_io import Scene


def load_models(ply_path: str, seg_weights: str, extra_weights: str | None = None):
    """Load once per process: CAD clouds/features and the segmenter(s)."""
    from ultralytics import YOLO
    extra = YOLO(extra_weights) if extra_weights else None
    return load_model_cloud(ply_path), YOLO(seg_weights), extra


def scene_from_arrays(rgb_bgr: np.ndarray, depth_raw_u16: np.ndarray,
                      K, depth_scale: float, scene_id: str = "live") -> Scene:
    """Wrap one registered RGB-D frame the way load_scene wraps a folder.

    Args:
        rgb_bgr: (H, W, 3) uint8 colour image, OpenCV channel order.
        depth_raw_u16: (H, W) raw depth ticks (uint16 as most SDKs deliver
            it; any integer or float array works), 0 = no measurement.
        K: (3, 3) intrinsics of the colour image.
        depth_scale: metres per depth tick (camera.json["depth_scale"]).
        scene_id: label used in logs and output files.
    """
    rgb_bgr = np.ascontiguousarray(rgb_bgr, dtype=np.uint8)
    depth_raw_u16 = np.asarray(depth_raw_u16)
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError("rgb_bgr must be (H, W, 3), got %r" % (rgb_bgr.shape,))
    if depth_raw_u16.shape != rgb_bgr.shape[:2]:
        raise ValueError("depth %r is not registered to rgb %r"
                         % (depth_raw_u16.shape, rgb_bgr.shape[:2]))
    return Scene(scene_id=scene_id, rgb=rgb_bgr,
                 depth=depth_raw_u16.astype(np.float64) * float(depth_scale),
                 K=np.asarray(K, dtype=np.float64).reshape(3, 3))


def estimate_scene(scene: Scene, model_cloud, seg, extra=None,
                   pick: bool = False) -> list:
    """Detect and register every instance in one frame.

    Mirrors scripts/run_pipeline.py: the estimator gets the class-level
    colour mask so winning poses receive the hole-centre polish, and the
    hybrid detector pools both segmenters with the geometric safety net.
    ``pick=True`` returns as soon as one pose clears the pick score --
    grab it, then rescan (a bin changes after every pick).

    Returns:
        [{"R": (3, 3) list, "t": (3,) list, "score": float}], best first;
        ``score`` = segmenter confidence x depth verification, the value to
        gate on before commanding a robot.
    """
    estimator = PoseEstimator(model_cloud, scene.depth, scene.K,
                              part_mask=part_pixel_mask(scene.rgb))
    kwargs = {"pick": True} if pick else {}
    found = detect_scene_hybrid(scene, estimator, seg, extra_model=extra,
                                **kwargs)
    return [{"R": e.R.tolist(), "t": e.t.tolist(),
             "score": round(e.submission_score, 4)} for e in found]
