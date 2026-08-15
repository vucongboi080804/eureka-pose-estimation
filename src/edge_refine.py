"""Final pose polish: depth Gauss-Newton plus image hole-centre alignment.

Depth quantised to 1 mm cannot pin a pose better than ~2 mm in-plane: whole
neighbourhoods of poses explain the staircase equally well, and pure depth
ICP stalls there. The colour image can -- but not via silhouette chamfer:
in a pile every neighbour shares the part's colour, so a rim point happily
matches the neighbour's edge and the chamfer minimum sits pixels away from
the truth.

The part's through-holes are the reliable image feature. A hole belongs to
one instance alone, it reads as an enclosed background region in both the
predicted silhouette and the observed part mask, and its centroid averages
the whole rim into sub-pixel precision. Matching predicted to observed hole
centres yields exact in-plane constraints; depth handles the rest:

* a Gauss-Newton **depth** pass (point-to-plane against the CAD mesh, with
  a deadzone of half the quantisation step) holds z and the two tilts;
* a **hole** pass solves the in-plane shift and view-ray roll from matched
  hole centres, moving the pose only as far as the matches demand.
"""

import cv2
import numpy as np
import open3d as o3d

#: Depth residuals inside the deadzone carry no gradient: the 1 mm
#: quantisation step erases structure at this scale, and a deadzone
#: narrower than the step lets the staircase bias the solve.
DEPTH_DEADZONE = 0.0009

#: Tukey cutoff for the depth residuals, metres.
TUKEY_K = 0.0015

#: Predicted and observed hole centres pair up only within this radius.
MATCH_RADIUS_PX = 12.0

#: ...and only when their areas agree within this factor.
MATCH_AREA_RATIO = 3.0

#: In-plane rotation needs a stable baseline; with fewer pairs than this
#: only the shift is solved. Two pairs were tried and measured WORSE
#: (train AR 0.838 -> 0.824): partially occluded holes shift their
#: observed centroids by more than the theoretical sub-pixel noise, and a
#: 2-point baseline amplifies that straight into roll.
MIN_PAIRS_FOR_ROLL = 3

#: Matched pairs whose residual after the first solve exceeds this are
#: dropped (a partly occluded hole reports a shifted centroid) and the
#: transform is re-solved from the rest.
PAIR_RESIDUAL_PX = 2.0

#: Holes smaller than this are noise, in pixels.
MIN_HOLE_AREA_PX = 30


class PosePolisher:
    """Polishes object->camera poses of one scene."""

    def __init__(self, mesh_path: str, model_points: np.ndarray,
                 model_normals: np.ndarray):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self._raycast = o3d.t.geometry.RaycastingScene()
        self._raycast.add_triangles(tmesh)
        mesh.compute_triangle_normals()
        self._face_normals = np.asarray(mesh.triangle_normals)
        self._points = model_points
        self._normals = model_normals

    def set_scene(self, depth: np.ndarray, K: np.ndarray,
                  part_mask: np.ndarray) -> None:
        """Give the polisher this scene's depth and observed part mask."""
        self.depth = depth
        self.K = K
        self._observed_holes = _holes_of(part_mask.astype(np.uint8))

    def polish(self, scene_points: np.ndarray, R: np.ndarray, t: np.ndarray):
        """Refine an object->camera pose against depth and hole centres.

        Args:
            scene_points: (N, 3) camera-frame points of this instance.
            R, t: Starting pose.

        Returns:
            (R, t) polished.
        """
        R, t = self._depth_pass(scene_points, R, t, restrict=False)
        moved, R, t = self._hole_pass(R, t)
        if moved:
            # In-plane moved; let depth re-settle the freedoms it owns
            # (pivoted at the part centre, these cannot undo the shift),
            # then snap the holes once more.
            R, t = self._depth_pass(scene_points, R, t, restrict=True)
            _, R, t = self._hole_pass(R, t)
        return R, t

    # ----------------------------------------------------------------- #
    # Depth pass
    # ----------------------------------------------------------------- #

    #: Restricted freedoms: tilt about the part's own centre plus z. With
    #: the rotation pivoted at the part centre these are exactly the
    #: freedoms depth measures well, and none of them moves the silhouette
    #: sideways.
    _TILT_AND_Z = np.array([0, 1, 5])

    def _depth_pass(self, scene_points, R, t, restrict: bool,
                    iterations: int = 6):
        """Deadzoned point-to-plane Gauss-Newton against the CAD mesh.

        The perturbation rotates about the part centroid, not the camera
        origin: X' = X + omega x (X - X0) + tau. Otherwise a small "tilt"
        about the camera origin drags the part sideways by ~0.7 m x omega
        and the restricted solve would leak into the in-plane freedoms.
        """
        centroid_obj = self._points.mean(axis=0)
        for _ in range(iterations):
            obj = (scene_points - t) @ R
            hit = self._raycast.compute_closest_points(
                o3d.core.Tensor(obj.astype(np.float32)))
            s_c = hit["points"].numpy().astype(np.float64) @ R.T + t
            n_c = self._face_normals[hit["primitive_ids"].numpy()] @ R.T
            X0 = R @ centroid_obj + t

            r = np.einsum("ij,ij->i", scene_points - s_c, n_c)
            r = np.sign(r) * np.maximum(0.0, np.abs(r) - DEPTH_DEADZONE)
            w = np.square(1.0 - np.square(np.clip(r / TUKEY_K, -1.0, 1.0)))
            # r(d) = r0 - ((s-X0) x n).omega - n.tau
            J = -np.hstack([np.cross(s_c - X0, n_c), n_c])
            if restrict:
                J = J[:, self._TILT_AND_Z]
            A = (J * w[:, None]).T @ J
            b = (J * w[:, None]).T @ r
            try:
                solved = np.linalg.solve(A, -b)
            except np.linalg.LinAlgError:
                break
            delta = np.zeros(6)
            if restrict:
                delta[self._TILT_AND_Z] = solved
            else:
                delta = solved
            omega, tau = delta[:3], delta[3:]
            R_step = _rodrigues(omega)
            R = R_step @ R
            t = R_step @ (t - X0) + X0 + tau
        return R, t

    # ----------------------------------------------------------------- #
    # Hole pass
    # ----------------------------------------------------------------- #

    def _hole_pass(self, R, t):
        """Snap in-plane shift and view-ray roll to matched hole centres.

        Returns:
            ``(moved, R, t)`` -- whether a correction was applied.
        """
        predicted = self._predicted_holes(R, t)
        if len(predicted) < 2 or len(self._observed_holes) == 0:
            return False, R, t
        pairs = _match_holes(predicted, self._observed_holes)
        if len(pairs) < 2:
            return False, R, t

        src = np.array([p[:2] for p, _ in pairs])
        dst = np.array([o[:2] for _, o in pairs])
        area = np.array([min(p[2], o[2]) for p, o in pairs])
        theta, shift = _rigid_2d(src, dst, area)
        residual = np.linalg.norm(_apply_2d(src, theta, shift, src.mean(0))
                                  - dst, axis=1)
        keep = residual < PAIR_RESIDUAL_PX
        if keep.sum() >= 2 and not keep.all():
            src, dst, area = src[keep], dst[keep], area[keep]
            theta, shift = _rigid_2d(src, dst, area)

        # Pixel correction -> camera frame, at the holes' depth.
        z = self._hole_depth(R, t)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        t = t + np.array([shift[0] * z / fx, shift[1] * z / fy, 0.0])
        centre = src.mean(axis=0)
        axis = np.array([(centre[0] - cx) / fx, (centre[1] - cy) / fy, 1.0])
        axis /= np.linalg.norm(axis)
        pivot = axis * z / axis[2]
        R_roll = _rodrigues(axis * theta)
        return True, R_roll @ R, R_roll @ (t - pivot) + pivot

    def _hole_depth(self, R, t) -> float:
        """Mean camera-frame depth of the part's surface."""
        return float((self._points @ R.T + t)[:, 2].mean())

    def _predicted_holes(self, R, t) -> list:
        """Hole centres of the posed model's silhouette, as (u, v, area)."""
        H, W = self.depth.shape
        pc = self._points @ R.T + t
        keep = pc[:, 2] > 1e-6
        pc = pc[keep]
        if not len(pc):
            return []
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        u = np.rint(fx * pc[:, 0] / pc[:, 2] + cx).astype(np.int64)
        v = np.rint(fy * pc[:, 1] / pc[:, 2] + cy).astype(np.int64)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[ok], v[ok]
        if not len(u):
            return []
        u0, v0 = u.min(), v.min()
        sil = np.zeros((v.max() - v0 + 1, u.max() - u0 + 1), np.uint8)
        sil[v - v0, u - u0] = 1
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return [(cu + u0, cv_ + v0, area)
                for cu, cv_, area in _holes_of(sil)]


def _holes_of(mask: np.ndarray) -> list:
    """Enclosed background regions of a binary mask: (u, v, area) each.

    ``RETR_CCOMP`` arranges contours in two levels; every child contour is
    the rim of a hole.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    out = []
    for contour, info in zip(contours, hierarchy[0]):
        if info[3] == -1:            # top level: an outer boundary
            continue
        area = cv2.contourArea(contour)
        if area < MIN_HOLE_AREA_PX:
            continue
        m = cv2.moments(contour)
        if m["m00"] <= 0:
            continue
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    return out


def _match_holes(predicted: list, observed: list) -> list:
    """Mutual-nearest pairs of hole centres, gated by radius and area."""
    pairs = []
    for p in predicted:
        best, best_d = None, MATCH_RADIUS_PX
        for o in observed:
            d = float(np.hypot(p[0] - o[0], p[1] - o[1]))
            if d < best_d:
                best, best_d = o, d
        if best is None:
            continue
        ratio = max(p[2], best[2]) / max(min(p[2], best[2]), 1.0)
        if ratio > MATCH_AREA_RATIO:
            continue
        back = min(predicted,
                   key=lambda q: np.hypot(q[0] - best[0], q[1] - best[1]))
        if back is p:
            pairs.append((p, best))
    return pairs


def _rigid_2d(src: np.ndarray, dst: np.ndarray, weight: np.ndarray):
    """Weighted least-squares rotation (about the src centroid) plus shift.

    A rotation estimated from a 2-point baseline amplifies centroid noise,
    so with fewer than MIN_PAIRS_FOR_ROLL pairs only the shift is solved.
    """
    w = weight / weight.sum()
    c_src = (src * w[:, None]).sum(axis=0)
    c_dst = (dst * w[:, None]).sum(axis=0)
    theta = 0.0
    if len(src) >= MIN_PAIRS_FOR_ROLL:
        a, b = src - c_src, dst - c_dst
        theta = float(np.arctan2(
            (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum(),
            (w * (a * b).sum(axis=1)).sum()))
    return theta, c_dst - c_src


def _apply_2d(pts, theta, shift, pivot):
    c, s = np.cos(theta), np.sin(theta)
    return (pts - pivot) @ np.array([[c, -s], [s, c]]).T + pivot + shift


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(omega)
    K = np.array([[0, -omega[2], omega[1]],
                  [omega[2], 0, -omega[0]],
                  [-omega[1], omega[0], 0]])
    if angle < 1e-8:
        return np.eye(3) + K
    return (np.eye(3) + np.sin(angle) / angle * K
            + (1 - np.cos(angle)) / angle**2 * K @ K)
