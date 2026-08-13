"""Judge a pose hypothesis against the observed depth map.

Registration fitness alone cannot tell a correct pose from a near-symmetric
flip: both explain the visible surface almost equally well. The depth map
can. A flipped pose puts model surface in front of surfaces the camera saw
behind it -- pixels where the camera measured *through* space the model
claims to occupy. Correct poses never do that, occluded poses sit behind the
observed depth, and only impossible poses sit in front of it.

Two sources of false "violation" are excluded before judging:

* Back-surface points -- only points whose normal faces the camera are
  projected, since only the front surface is comparable to a depth map.
* Silhouette-boundary pixels -- a projected edge point can land one pixel
  outside the part, where the measurement legitimately belongs to the
  background. The comparison runs on the silhouette's interior only.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Verdict:
    """How a posed model relates to the observed depth, per interior pixel.

    Fractions are over the model's interior silhouette pixels that carry a
    depth measurement; they sum to 1.
    """

    support: float      # model depth agrees with observed depth
    violation: float    # model in front of observed surface: impossible
    occluded: float     # model behind observed surface: hidden by something
    n_pixels: int       # interior pixels with a valid measurement

    @property
    def visible_fraction(self) -> float:
        """Estimate of the instance's visible fraction, comparable to the
        dataset's mask-area / amodal-area definition."""
        return self.support

    @property
    def confidence(self) -> float:
        """Single figure for ranking hypotheses: reward agreement, punish
        impossibility. Occlusion is left neutral -- being hidden is not
        evidence against a pose."""
        return max(0.0, self.support - 2.0 * self.violation)


def depth_slope(depth: np.ndarray) -> np.ndarray:
    """Per-pixel depth change, metres per pixel, for slope-aware margins.

    On a steep surface one pixel of projection error is worth several
    millimetres of depth, so a fixed comparison margin misreads steep faces
    as violations. Precompute once per scene and hand to
    :func:`verify_pose`.
    """
    dv, du = np.gradient(depth)
    return np.hypot(du, dv)


def verify_pose(R: np.ndarray, t: np.ndarray,
                model_points: np.ndarray, model_normals: np.ndarray,
                depth: np.ndarray, K: np.ndarray,
                margin: float = 0.003,
                slope: np.ndarray | None = None) -> Verdict:
    """Compare the posed model's front surface with the depth map.

    Args:
        R, t: Pose, object -> camera.
        model_points: (N, 3) dense sample of the CAD surface, object frame.
        model_normals: (N, 3) outward normals of that sample.
        depth: (H, W) observed depth, metres, 0 = no measurement.
        K: (3, 3) intrinsics.
        margin: Agreement tolerance on flat ground, metres. Wider than the
            sensor noise so only genuine disagreement counts.
        slope: Optional precomputed :func:`depth_slope`; the margin widens
            with it so steep faces are judged as leniently as flat ones.

    Returns:
        Per-pixel fractions of agreement, violation and occlusion.
    """
    H, W = depth.shape
    pc = model_points @ R.T + t
    # Front-facing: normal (rotated to camera frame) against the view ray.
    facing = np.einsum("ij,ij->i", model_normals @ R.T, pc) < 0
    pc = pc[facing & (pc[:, 2] > 1e-6)]
    if not len(pc):
        return Verdict(0.0, 1.0, 0.0, 0)

    u = np.rint(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return Verdict(0.0, 1.0, 0.0, 0)
    u, v, z = u[inside], v[inside], pc[inside, 2]

    # Poor man's z-buffer over the silhouette's bounding box: the model's
    # visible depth at a pixel is the nearest point landing there.
    u0, v0 = u.min(), v.min()
    w, h = u.max() - u0 + 1, v.max() - v0 + 1
    lin = (v - v0) * w + (u - u0)
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, lin, z)
    zbuf = zbuf.reshape(h, w)

    # Interior pixels only: close point-splat pinholes, then peel the rim.
    sil = (zbuf < np.inf).astype(np.uint8)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    interior = cv2.erode(sil, np.ones((3, 3), np.uint8), iterations=2) > 0
    interior &= zbuf < np.inf

    roi = np.s_[v0:v0 + h, u0:u0 + w]
    observed = depth[roi][interior]
    measured = observed > 0
    n = int(measured.sum())
    if not n:
        return Verdict(0.0, 1.0, 0.0, 0)

    tolerance = np.full(n, margin)
    if slope is not None:
        # ~1.5 px of projection/rasterisation error, expressed in depth.
        tolerance = margin + np.minimum(1.5 * slope[roi][interior][measured],
                                        0.009)

    diff = zbuf[interior][measured] - observed[measured]
    support = float((np.abs(diff) <= tolerance).sum() / n)
    violation = float((diff < -tolerance).sum() / n)
    return Verdict(support, violation, 1.0 - support - violation, n)
