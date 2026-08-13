"""Split a pile into smooth surface patches.

A ball of points around a pile seed mixes several instances, and global
registration cannot be trusted on a contaminated cloud. But one *smooth
face* always belongs to one instance: surfaces break -- in depth or in
normal direction -- exactly where one part ends and another begins. Cutting
the component at those breaks yields clean per-face point sets to register,
at the cost of a face being smaller than the whole visible instance (the
registration's verification stage judges the full model against the full
depth map, so a face is enough to propose a pose).
"""

import cv2
import numpy as np

#: A depth step between 4-neighbours larger than this separates surfaces,
#: metres.
DEPTH_STEP = 0.003

#: Neighbouring normals disagreeing by more than this separate surfaces.
NORMAL_ANGLE_DEG = 28.0

#: Patches smaller than this carry too little surface to propose a pose.
MIN_PATCH_PX = 700


def normal_map(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-pixel outward surface normals of an organised depth map.

    Depth is bilaterally smoothed first: 1 mm quantisation makes raw
    gradients staircase-noisy, and the filter's range cutoff keeps the
    smoothing from bleeding across real depth steps.
    """
    smooth = cv2.bilateralFilter(depth.astype(np.float32), 7, 0.004, 3.0)
    smooth = smooth.astype(np.float64)
    H, W = depth.shape
    xs = (np.arange(W)[None, :] - K[0, 2]) * smooth / K[0, 0]
    ys = (np.arange(H)[:, None] - K[1, 2]) * smooth / K[1, 1]
    P = np.dstack([xs, ys, smooth])
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    towards_camera = np.einsum("ijk,ijk->ij", n, P) > 0
    n[towards_camera] *= -1.0
    return n


def surface_patches(depth: np.ndarray, K: np.ndarray,
                    component: np.ndarray) -> np.ndarray:
    """Label a component's pixels into smooth surface patches.

    Args:
        depth: (H, W) metres, 0 = no measurement.
        K: (3, 3) intrinsics.
        component: (H, W) bool, the pile to split.

    Returns:
        (H, W) int32 labels; 0 is background/edges, 1..N are patches
        (smallest ones already suppressed).
    """
    valid = component & (depth > 0)
    normals = normal_map(depth, K)

    edge = np.zeros_like(valid)
    for axis in (0, 1):
        d_step = np.abs(np.diff(depth, axis=axis)) > DEPTH_STEP
        cos = np.einsum("ijk,ijk->ij",
                        normals[:-1, :] if axis == 0 else normals[:, :-1],
                        normals[1:, :] if axis == 0 else normals[:, 1:])
        n_break = cos < np.cos(np.deg2rad(NORMAL_ANGLE_DEG))
        breaks = d_step | n_break
        if axis == 0:
            edge[:-1, :] |= breaks
            edge[1:, :] |= breaks
        else:
            edge[:, :-1] |= breaks
            edge[:, 1:] |= breaks

    n_labels, labels = cv2.connectedComponents(
        (valid & ~edge).astype(np.uint8))
    counts = np.bincount(labels.reshape(-1), minlength=n_labels)
    too_small = counts < MIN_PATCH_PX
    too_small[0] = True
    labels[too_small[labels]] = 0
    return labels
