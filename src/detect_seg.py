"""Registration driven by learned instance masks.

A fine-tuned segmenter replaces the geometric pile-splitting: each
predicted mask is registered exactly like an oracle mask. The geometric
verification stack stays -- learned masks propose, depth physics disposes.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import (EXPLAINED_DIST, MIN_CONFIDENCE, MIN_OWN_CONSUMED,
                     detect_scene, nms, part_pixel_mask)
from .register import PoseEstimator
from .scene_io import Scene, backproject

#: Boundary peel for predicted masks, pixels (mask edges carry blended depth).
ERODE_PX = 2

#: Predicted masks smaller than this cannot hold a registrable instance.
MIN_MASK_PX = 1200

#: A proposal must be at least this part-coloured to be registered. The
#: synthetic-only segmenter is trained with randomised part colours, so it
#: also fires on plain light background (tray rims, white table); a flat
#: patch of background can still pass depth verification because ICP sinks
#: the flat CAD plate flush into the support plane. Real proposals of either
#: segmenter are >= 0.7 part-coloured, background ones <= 0.16 -- the gate
#: sits in the empty middle. It reuses the HSV gate the hole polisher
#: already relies on, so it adds no new colour assumption to the pipeline.
MIN_PART_COLOUR_FRACTION = 0.3

#: A verified pose must also explain the mask that proposed it: at least
#: this fraction of the mask's points (capped at the geometric detector's
#: MIN_OWN_CONSUMED) within EXPLAINED_DIST of the posed model surface. This
#: mirrors the geometric detector's progress invariant -- a pose that
#: verifies somewhere else does not deserve the proposal it came from. The
#: motivating case is a mask cut off by the image edge whose registration
#: drifted out of frame and settled the flat CAD plate on the tray floor,
#: where free-space verification cannot object (nothing is in front of the
#: floor) but almost none of the mask's own points lie on the model.
OWN_MASK_MIN_FRACTION = 0.3

#: Pick mode: a robot cell needs one confident pick per cycle, then rescans,
#: so the sweep may stop at the first pose scoring at least this. Proposals
#: are registered in segmenter-confidence order, so the first hit is a good
#: pick; the submission uses the full sweep.
PICK_SCORE = 0.8

#: Domain-shift guard: when fewer than this many detections verify at least
#: this well, the segmenter is assumed to be out of its depth (new lighting,
#: background, or part appearance) and the training-free geometric detector
#: joins in.
FALLBACK_MIN_STRONG = 2
FALLBACK_STRONG_CONF = 0.5


def detect_from_masks(scene: Scene, estimator: PoseEstimator,
                      masks: list, attempts: int = 3,
                      colour_gate: bool = True, own_mask_check: bool = True,
                      stop_at: float | None = None) -> list:
    """Register one scene's learned instance masks.

    Args:
        scene: The capture.
        estimator: Prepared registration stack for this scene.
        masks: [(bool_mask, seg_confidence)], any order.
        attempts: RANSAC restart budget per mask.
        colour_gate: Ablation switch for the part-colour proposal gate.
        own_mask_check: Ablation switch for the own-mask explanation check.
        stop_at: Pick mode -- return as soon as a pose scores at least
            this (None: register every proposal).

    Returns:
        Verified pose estimates, deduplicated.
    """
    part_colour = part_pixel_mask(scene.rgb)
    found = []
    for mask, seg_conf in sorted(masks, key=lambda m: -m[1]):
        if mask.sum() < MIN_MASK_PX:
            continue
        if colour_gate and part_colour[mask].mean() < MIN_PART_COLOUR_FRACTION:
            continue
        slim = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=ERODE_PX).astype(bool)
        if slim.sum() < MIN_MASK_PX // 2:
            slim = mask
        points = backproject(scene.depth, scene.K, slim)
        if len(points) < 300:
            continue
        anchor = points[np.argmin(points[:, 2])]
        est = estimator.estimate(points, attempts=attempts, anchor=anchor)
        if est is None or est.confidence < MIN_CONFIDENCE:
            continue
        if own_mask_check:
            explained = int((estimator.distance_to_model(points, est)
                             < EXPLAINED_DIST).sum())
            if explained < min(MIN_OWN_CONSUMED,
                               OWN_MASK_MIN_FRACTION * len(points)):
                continue
        # Submission rank = joint belief: how much the segmenter trusted
        # the proposal times how well the pose verifies. Low-confidence
        # proposals may still land true poses (good for recall), but they
        # must not outrank solid ones -- ranking decides top-1.
        est.seg_conf = seg_conf
        found.append(est)
        if stop_at is not None and est.submission_score >= stop_at:
            break
    return nms(found)


def detect_scene_hybrid(scene: Scene, estimator: PoseEstimator,
                        seg_model, extra_model=None,
                        conf: float = 0.25, pick: bool = False) -> list:
    """Learned masks first; geometric detector as a domain-shift safety net.

    ``extra_model`` widens the proposal pool (e.g. the synthetic-only
    segmenter): proposals cost only registration time, wrong ones die in
    verification, and the joint submission score keeps weak proposals
    from outranking solid ones.

    The verifier is environment-agnostic — a wrong pose fails the
    free-space check no matter where its mask came from — so a collapse of
    verified detections is a reliable sign the segmenters left their
    training domain. In that case the colour-and-geometry detector (which
    needs no training) sweeps the scene too, and the union is deduplicated.

    ``pick`` stops at the first pose scoring at least PICK_SCORE (one
    confident pick per robot cycle); the safety net only runs when no
    pick was found.
    """
    masks = masks_from_model(seg_model, scene.rgb, conf=conf)
    if extra_model is not None:
        masks += masks_from_model(extra_model, scene.rgb, conf=conf)
    found = detect_from_masks(scene, estimator, masks,
                              stop_at=PICK_SCORE if pick else None)
    if pick and any(e.submission_score >= PICK_SCORE for e in found):
        return found
    strong = [e for e in found if e.confidence >= FALLBACK_STRONG_CONF]
    if len(strong) >= FALLBACK_MIN_STRONG:
        return found
    return nms(found + detect_scene(scene, estimator))


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
