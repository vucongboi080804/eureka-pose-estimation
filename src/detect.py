"""Find every part instance in a scene that ships no masks.

The part is saturated orange on a grey/white background, so colour alone
finds nearly every part pixel (98% recall on train). What colour cannot do
is split touching parts into instances. Geometry does that in two steps:
connected components of the colour mask are *piles*, and each pile is cut
into smooth surface patches (`src/surface_patches.py`) -- a smooth face
always belongs to a single instance, because surfaces break in depth or
normal direction exactly where one part ends and the next begins.

Patches are registered top-of-pile first: the least occluded instances are
both the ones the score requires and the ones whose removal uncovers the
rest. Every accepted pose consumes the pixels it explains; every rejected
patch is buried; either way the sweep shrinks and terminates.

False colour regions (saturated patches of cardboard) survive segmentation
but die in registration: a flat patch of background cannot be explained by
the CAD surface, so its hypotheses never reach the confidence floor.
"""

import cv2
import numpy as np

from .register import PoseEstimator
from .scene_io import Scene, backproject
from .surface_patches import surface_patches

#: Part pixels are saturated orange-red; cardboard is duller and browner.
SATURATION_MIN = 90
HUE_RED_BELOW, HUE_RED_ABOVE = 15, 170

#: Peel segmentation boundaries: edge pixels carry depth blended between the
#: part and whatever lies behind it.
ERODE_PX = 2

#: Components smaller than this cannot hold a scoreable instance
#: (the smallest labelled instance in train covers ~5500 raw pixels).
MIN_COMPONENT_PX = 1500

#: A patch must keep at least this many unconsumed pixels to be worth a
#: registration attempt.
MIN_LIVE_PATCH_PX = 600

#: Hypotheses below this depth-map confidence are discarded. Deliberately
#: permissive: a matched prediction on a barely-visible instance costs
#: nothing, an unmatched one costs precision but never AR, and steep faces
#: cap out at modest support even for perfect poses.
MIN_CONFIDENCE = 0.15

#: On a failed attempt, bury only this ball around the seed and retry the
#: patch elsewhere -- a merged patch can hold several instances.
DEAD_BALL_RADIUS = 0.02

#: Strike budget per patch before burying what is left of it. A whole flat
#: pile can merge into one patch, so the budget grows with its size --
#: four strikes must not erase nine instances.
STRIKES_BASE = 4
STRIKES_PER_PX = 1 / 6000.0

#: Surface registration below this confidence is not trusted as final: the
#: hole-pair proposer also gets a shot and the better verdict wins. On a
#: merged flat patch, surface RANSAC often lands a plausible-looking pose
#: bridging two instances; the holes see through it.
HOLE_TRY_BELOW = 0.65

#: A point is explained by an accepted pose when it lies this close to the
#: posed model surface, metres.
EXPLAINED_DIST = 0.003

#: A pose must explain at least this many points of the patch that
#: proposed it. A hypothesis that landed somewhere else entirely (a false
#: hole pairing can verify decently 70 mm away) would neither shrink the
#: patch -- looping forever -- nor deserve the points it would steal from
#: whatever really lives where it landed.
MIN_OWN_CONSUMED = 150

#: Two detections this close (object centres, metres) AND this aligned
#: (degrees) are duplicates. Both tests: two parts stacked flat sit only a
#: thickness apart, but then their orientations differ.
NMS_DIST = 0.009
NMS_ANGLE_DEG = 30.0


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


def observed_holes_3d(scene: Scene, part_mask: np.ndarray,
                      holes_uv: list) -> list:
    """Lift observed hole centres to camera-frame 3D points.

    A hole's own pixels see the surface *behind* the part, so its depth is
    read off a ring just outside the rim -- the part surface around it.

    Returns:
        [(centre_xyz, radius_metres)].
    """
    fx, fy = scene.K[0, 0], scene.K[1, 1]
    cx, cy = scene.K[0, 2], scene.K[1, 2]
    H, W = scene.depth.shape
    out = []
    for u, v, area in holes_uv:
        r_px = np.sqrt(area / np.pi)
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        ring_r = r_px + 3.0
        us = np.rint(u + ring_r * np.cos(angles)).astype(np.int64)
        vs = np.rint(v + ring_r * np.sin(angles)).astype(np.int64)
        ok = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        us, vs = us[ok], vs[ok]
        good = part_mask[vs, us] & (scene.depth[vs, us] > 0)
        if good.sum() < 6:
            continue
        z = float(np.median(scene.depth[vs[good], us[good]]))
        centre = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        out.append((centre, r_px * z / fx))
    return out


def _patch_plane_normal(points: np.ndarray) -> np.ndarray:
    """Outward (towards-camera) normal of a roughly planar point set."""
    centred = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    normal = vectors[:, 0]
    return normal if normal @ points.mean(axis=0) < 0 else -normal


def extract_instances(scene: Scene, component: np.ndarray,
                      estimator: PoseEstimator, holes_3d: list) -> list:
    """Register one pile's instances, one surface patch at a time."""
    labels = surface_patches(scene.depth, scene.K, component)
    component_points = backproject(scene.depth, scene.K, component)

    patches = {}
    strikes = {}
    for pid in np.unique(labels):
        if pid == 0:
            continue
        pts = backproject(scene.depth, scene.K, labels == pid)
        patches[pid] = (pts, np.ones(len(pts), bool))
        strikes[pid] = 0

    found = []
    while len(found) < 40:
        # Topmost live patch first.
        best_pid, best_z = None, np.inf
        for pid, (pts, live) in patches.items():
            if live.sum() < MIN_LIVE_PATCH_PX:
                continue
            z = pts[live, 2].min()
            if z < best_z:
                best_pid, best_z = pid, z
        if best_pid is None:
            break

        pts, live = patches[best_pid]
        sub = pts[live]
        anchor = sub[np.argmin(sub[:, 2])]
        est = estimator.estimate(sub, attempts=3, anchor=anchor)

        if est is None or est.confidence < HOLE_TRY_BELOW:
            # Flat coplanar neighbours defeat surface features; the
            # through-holes around the seed still pin the pose. Everything
            # is local to the anchor: a merged patch spans several
            # instances, so its centre and mean normal describe nobody.
            near = [h for h in holes_3d
                    if np.linalg.norm(h[0] - anchor) < 0.08]
            local = sub[np.linalg.norm(sub - anchor, axis=1) < 0.025]
            if len(near) >= 2 and len(local) >= 60:
                by_holes = estimator.estimate_from_holes(
                    near, _patch_plane_normal(local), sub)
                if by_holes is not None and (
                        est is None
                        or by_holes.confidence > est.confidence):
                    est = by_holes

        if est is not None and est.confidence >= MIN_CONFIDENCE:
            est = estimator.refine_local(component_points, est)

        claims = {}
        own_claim = 0
        if est is not None and est.confidence >= MIN_CONFIDENCE:
            for pid2, (p2, live2) in patches.items():
                if not live2.any():
                    continue
                idx = np.nonzero(live2)[0]
                near_model = estimator.distance_to_model(
                    p2[idx], est) < EXPLAINED_DIST
                claims[pid2] = idx[near_model]
                if pid2 == best_pid:
                    own_claim = int(near_model.sum())

        if (est is None or est.confidence < MIN_CONFIDENCE
                or own_claim < MIN_OWN_CONSUMED):
            # A merged patch can hold several instances: bury only the
            # failed seed's neighbourhood, unless the patch keeps failing.
            strikes[best_pid] += 1
            budget = STRIKES_BASE + int(len(pts) * STRIKES_PER_PX)
            if strikes[best_pid] >= budget:
                live[:] = False
            else:
                idx = np.nonzero(live)[0]
                dead = np.linalg.norm(pts[idx] - anchor,
                                      axis=1) < DEAD_BALL_RADIUS
                live[idx[dead]] = False
            continue

        found.append(est)
        for pid2, idx in claims.items():
            patches[pid2][1][idx] = False

    found.extend(_cleanup_pass(patches, estimator, component_points))
    return found


def _cleanup_pass(patches, estimator, component_points,
                  max_extra: int = 6) -> list:
    """Sweep the pooled leftovers for instances the greedy pass damaged.

    Early acceptances can nibble or fragment a neighbouring instance's
    patch below the per-patch threshold; pooled together, those leftovers
    are often still one registrable instance.
    """
    leftovers = [pts[live] for pts, live in patches.values() if live.any()]
    if not leftovers:
        return []
    pool = np.vstack(leftovers)
    extra = []
    while len(pool) >= 2500 and len(extra) < max_extra:
        anchor = pool[np.argmin(pool[:, 2])]
        est = estimator.estimate(pool, attempts=2, anchor=anchor)
        if est is not None and est.confidence >= MIN_CONFIDENCE:
            est = estimator.refine_local(component_points, est)
        claimed = None
        if est is not None and est.confidence >= MIN_CONFIDENCE:
            claimed = estimator.distance_to_model(pool, est) < EXPLAINED_DIST
        if claimed is None or claimed.sum() < MIN_OWN_CONSUMED:
            dead = np.linalg.norm(pool - anchor, axis=1) < DEAD_BALL_RADIUS
            if not dead.any():
                break
            pool = pool[~dead]
            continue
        extra.append(est)
        pool = pool[~claimed]
    return extra


def nms(estimates: list) -> list:
    """Drop the lower-confidence member of any pair that is really the same
    detection: nearly the same position *and* orientation."""
    kept = []
    for est in sorted(estimates, key=lambda e: -e.confidence):
        duplicate = False
        for k in kept:
            if np.linalg.norm(est.t - k.t) > NMS_DIST:
                continue
            cos = (np.trace(est.R.T @ k.R) - 1.0) / 2.0
            angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            if angle < NMS_ANGLE_DEG:
                duplicate = True
                break
        if not duplicate:
            kept.append(est)
    return kept


def detect_scene(scene: Scene, estimator: PoseEstimator,
                 passes: int = 1, foreground: str = "colour") -> list:
    """All instances of one scene, best-verified first.

    ``passes`` reruns the whole sweep and unions the results: RANSAC makes
    the greedy extraction path-dependent, so independent sweeps drop
    *different* instances, and their union (deduplicated by NMS) recovers
    most of what any single sweep loses.

    ``foreground`` selects how part pixels are found: "colour" (the HSV
    gate, sharpest when the part keeps its colour) or "depth" (support-
    plane removal — colour-blind, for parts or lighting the gate has
    never seen).
    """
    if foreground == "depth":
        clean_mask = foreground_depth_mask(scene.depth, scene.K)
        mask = cv2.erode(clean_mask.astype(np.uint8),
                         np.ones((3, 3), np.uint8), iterations=ERODE_PX) > 0
    else:
        clean_mask = part_pixel_mask(scene.rgb)
        mask = part_pixel_mask(scene.rgb, erode_px=ERODE_PX)
    holes_3d = observed_holes_3d(scene, clean_mask,
                                 _mask_holes(clean_mask))
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    found = []
    for _ in range(passes):
        for label in range(1, n_labels):
            component = labels == label
            if component.sum() < MIN_COMPONENT_PX:
                continue
            found.extend(extract_instances(scene, component, estimator,
                                           holes_3d))
    return nms(found)


def _mask_holes(part_mask: np.ndarray) -> list:
    """(u, v, area) of every enclosed background region of the part mask."""
    contours, hierarchy = cv2.findContours(part_mask.astype(np.uint8),
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    out = []
    for contour, info in zip(contours, hierarchy[0]):
        if info[3] == -1:
            continue
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        m = cv2.moments(contour)
        if m["m00"] <= 0:
            continue
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    return out
