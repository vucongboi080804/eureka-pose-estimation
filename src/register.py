"""Estimate T_camera_object from a segmented depth region.

The scene cloud is the visible surface only, so registration runs
scene -> model (every scene point has a counterpart on the full CAD surface)
and the result is inverted at the end to give the object-to-camera pose the
submission format wants.

Global initialisation is FPFH feature matching under RANSAC, refined by
robust point-to-plane ICP against a dense CAD sample. Two failure modes are
handled explicitly:

* RANSAC is stochastic -- it is restarted until the refined pose verifies
  well against the depth map or the attempt budget runs out.
* The part is nearly 180-degree symmetric about its own axes, so ICP happily
  converges onto a flipped pose. Every refined pose therefore spawns three
  flipped candidates (pi about X, Y, Z through the model centroid), each is
  re-refined, and the depth-map verdict -- not ICP fitness -- picks the
  winner, because a flip explains the visible surface but pokes through
  free space the camera saw behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .edge_refine import PosePolisher
from .model_cloud import ModelCloud
from .verify import Verdict, depth_slope, verify_pose

#: ICP correspondence distances on the coarse scene cloud, metres.
ICP_SCHEDULE = (0.006, 0.0025)

#: Final ICP distances on the full-resolution scene cloud, metres.
ICP_FINE_SCHEDULE = (0.0015, 0.001)

#: Voxel for the full-resolution refinement cloud, metres.
FINE_VOXEL = 0.001

#: Inlier distance used to judge a finished registration, metres.
FITNESS_DIST = 0.0015

#: RANSAC restarts: stop early once a pose verifies this well...
GOOD_CONFIDENCE = 0.85

#: ...or give up after this many restarts and keep the best seen.
MAX_ATTEMPTS = 5

#: Candidates whose confidence is within this of the best are treated as
#: equally plausible; the tighter ICP fitness picks among them.
CONFIDENCE_TIE = 0.1

#: When feature matching leaves nothing verifying at least this well, fall
#: back to brute force over a rotation grid.
FALLBACK_TRIGGER = 0.5

#: Rotation grid: this many viewing directions, each with this many rolls.
GRID_DIRECTIONS = 12
GRID_ROLLS = 5


@dataclass
class PoseEstimate:
    """One registration result, in submission conventions."""

    R: np.ndarray          # (3, 3) object -> camera rotation
    t: np.ndarray          # (3,) metres
    fitness: float         # inlier fraction of the scene cloud at FITNESS_DIST
    rmse: float            # inlier RMSE at FITNESS_DIST, metres
    verdict: Verdict       # depth-map verification of the winning pose
    n_points: int          # scene points used
    seg_conf: float = 1.0  # confidence of the proposing segmenter, if any

    @property
    def confidence(self) -> float:
        return self.verdict.confidence

    @property
    def submission_score(self) -> float:
        """Joint belief for ranking: pose verification times the proposing
        segmenter's confidence (1.0 when no segmenter was involved)."""
        return self.confidence * self.seg_conf


def make_scene_cloud(points: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    """Downsampled scene cloud with camera-facing normals.

    The camera sits at the origin of its own frame, so orienting normals
    towards the origin makes them point out of the surface the camera saw.
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    return cloud


def _frame_between(model_dir: np.ndarray, model_normal: np.ndarray,
                   obs_dir: np.ndarray, obs_normal: np.ndarray):
    """Rotation mapping (model_dir, model_normal) onto (obs_dir, obs_normal).

    Directions are projected into their normal's plane first; degenerate
    inputs return None.
    """
    def frame(direction, normal):
        n = normal / np.linalg.norm(normal)
        d = direction - (direction @ n) * n
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            return None
        d = d / norm
        return np.column_stack([d, np.cross(n, d), n])

    F_model = frame(model_dir, model_normal)
    F_obs = frame(obs_dir, obs_normal)
    if F_model is None or F_obs is None:
        return None
    return F_obs @ F_model.T


def grid_rotations(directions: int = GRID_DIRECTIONS,
                   rolls: int = GRID_ROLLS) -> list:
    """A coarse cover of SO(3): Fibonacci-sphere viewing directions, each
    swept through evenly spaced rolls. Every orientation lies within ~35
    degrees of some grid member -- inside coarse ICP's pull-in range."""
    out = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(directions):
        z = 1.0 - 2.0 * (i + 0.5) / directions
        r = np.sqrt(max(0.0, 1.0 - z * z))
        d = np.array([r * np.cos(golden * i), r * np.sin(golden * i), z])
        # Rotation taking +Z to d.
        v = np.cross([0.0, 0.0, 1.0], d)
        s, c = np.linalg.norm(v), d[2]
        if s < 1e-9:
            base = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            base = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
        for k in range(rolls):
            a = 2.0 * np.pi * k / rolls
            roll = np.array([[np.cos(a), -np.sin(a), 0],
                             [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            out.append(base @ roll)
    return out


def flip_transforms(model: ModelCloud) -> list:
    """Pi rotations about X, Y, Z through the model centroid, as 4x4s.

    These are the near-symmetries a flat part offers ICP as false minima;
    composing them with a converged pose enumerates its rivals.
    """
    c = np.asarray(model.fine.get_center())
    out = []
    for axis in np.eye(3):
        R = 2.0 * np.outer(axis, axis) - np.eye(3)   # pi rotation about axis
        S = np.eye(4)
        S[:3, :3] = R
        S[:3, 3] = c - R @ c
        out.append(S)
    return out


class PoseEstimator:
    """Registers instance point clouds of one scene against the CAD model."""

    def __init__(self, model: ModelCloud, depth: np.ndarray, K: np.ndarray,
                 part_mask: np.ndarray | None = None, flips: bool = True,
                 grid: bool = True, polish: bool = True):
        """Args:
            model: Prepared CAD clouds and features.
            depth: (H, W) scene depth, metres.
            K: (3, 3) intrinsics.
            part_mask: Optional class-level part mask (all instances, no
                erosion). When given, winning poses get a final joint
                depth+edge polish -- the stage that recovers the in-plane
                precision quantised depth cannot provide.
            flips, grid, polish: Ablation switches for the flip rivals, the
                rotation-grid fallback and the polish stage. All on in
                production; each off measures what that stage buys.
        """
        self.model = model
        self.depth = depth
        self.K = K
        self.use_flips = flips
        self.use_grid = grid
        self.use_polish = polish
        self._flips = flip_transforms(model)
        self._grid = grid_rotations()
        # Denser grid for the anchored search: the normal gate below prunes
        # most of it, so density is affordable exactly where it helps.
        self._grid_dense = grid_rotations(directions=24, rolls=8)
        self._slope = depth_slope(depth)
        self._model_np = np.asarray(model.fine.points)
        self._normals_np = np.asarray(model.fine.normals)
        # Plain point-to-plane while far, robust once close: a Tukey kernel
        # narrower than the remaining misalignment zeroes every residual's
        # weight and freezes ICP where it starts.
        self._p2l_coarse = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        self._p2l_fine = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=0.0015))
        self._model_tree = cKDTree(self._model_np)
        self._coarse_np = np.asarray(model.coarse.points)
        self._coarse_nrm = np.asarray(model.coarse.normals)
        # Locally averaged normals: the top point of a tilted part is often
        # a corner, where the raw normal is one arbitrary face's; the mean
        # over a neighbourhood is stable enough to gate candidates on.
        nbrs = cKDTree(self._coarse_np).query_ball_point(self._coarse_np,
                                                         r=0.008)
        mean_nrm = np.array([self._coarse_nrm[i].mean(axis=0) for i in nbrs])
        self._coarse_nrm_smooth = mean_nrm / np.maximum(
            np.linalg.norm(mean_nrm, axis=1, keepdims=True), 1e-12)
        self._polisher = None
        if part_mask is not None:
            self._polisher = PosePolisher(model.mesh_path, self._model_np,
                                          self._normals_np)
            self._polisher.set_scene(depth, K, part_mask)

    def _global_init(self, scene, scene_fpfh) -> np.ndarray:
        """One RANSAC feature-matching attempt; returns scene->model, 4x4."""
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            scene, self.model.coarse, scene_fpfh, self.model.fpfh,
            mutual_filter=False,
            max_correspondence_distance=self.model.voxel * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.model.voxel * 1.5),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.9999),
        )
        return result.transformation

    def _refine(self, scene, fine_scene, T_ms: np.ndarray) -> np.ndarray:
        """Robust point-to-plane ICP, coarse to fine, onto the dense model.

        The last stages run on the full-resolution scene cloud: MSSD
        amplifies rotation error by the part's ~43 mm lever arm, and rotation
        is exactly what a few thousand extra correspondences pin down.
        """
        T = T_ms
        for cloud, schedule, method in (
                (scene, ICP_SCHEDULE, self._p2l_coarse),
                (fine_scene, ICP_FINE_SCHEDULE, self._p2l_fine)):
            for dist in schedule:
                icp = o3d.pipelines.registration.registration_icp(
                    cloud, self.model.fine, dist, T, method,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                T = icp.transformation
        return T

    def _judge(self, scene, T_ms: np.ndarray):
        """Fitness on the scene cloud plus the depth-map verdict."""
        fit = o3d.pipelines.registration.evaluate_registration(
            scene, self.model.fine, FITNESS_DIST, T_ms)
        T_co = np.linalg.inv(T_ms)
        verdict = verify_pose(T_co[:3, :3], T_co[:3, 3],
                              self._model_np, self._normals_np,
                              self.depth, self.K, slope=self._slope)
        return fit, verdict

    def estimate(self, points: np.ndarray, attempts: int = MAX_ATTEMPTS,
                 allow_fallback: bool = True,
                 anchor: np.ndarray | None = None) -> PoseEstimate | None:
        """Register one instance's back-projected depth points.

        Args:
            points: (N, 3) camera-frame points of the instance's visible
                surface.
            attempts: RANSAC restart budget.
            allow_fallback: Permit the rotation-grid brute force when
                feature matching verifies poorly. Callers sweeping cluttered
                residue turn it off for regions too small to be worth it.
            anchor: Optional camera-frame point known to lie on the target
                instance's top surface (a pile seed); pins the fallback
                grid's translation.

        Returns:
            The best-verifying pose, or None when the region is too small.
        """
        if len(points) < 50:
            return None
        scene = make_scene_cloud(points, self.model.voxel)
        if len(scene.points) < 20:
            return None
        fine_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        fine_scene = fine_scene.voxel_down_sample(FINE_VOXEL)
        scene_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            scene,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.model.voxel * 5.0,
                                                 max_nn=100))

        judged = []   # (fit, verdict, T_ms)
        for _ in range(attempts):
            T = self._refine(scene, fine_scene,
                             self._global_init(scene, scene_fpfh))
            rivals = [S @ T for S in self._flips] if self.use_flips else []
            for candidate in [T] + rivals:
                T_ref = self._refine(scene, fine_scene, candidate)
                judged.append((*self._judge(scene, T_ref), T_ref))
            if max(v.confidence for _, v, _ in judged) >= GOOD_CONFIDENCE:
                break

        if (allow_fallback and self.use_grid
                and max(v.confidence for _, v, _ in judged) < FALLBACK_TRIGGER):
            # Feature matching found nothing that verifies: brute force.
            # Every grid orientation gets a short coarse alignment; the few
            # that grip the cloud get the full refinement.
            for T0 in self._grid_candidates(scene, anchor=anchor):
                T_ref = self._refine(scene, fine_scene, T0)
                judged.append((*self._judge(scene, T_ref), T_ref))

        return self._select_and_package(judged, scene, fine_scene)

    def estimate_from_holes(self, holes: list, surface_normal: np.ndarray,
                            points: np.ndarray) -> PoseEstimate | None:
        """Register from matched hole pairs instead of surface features.

        On flat-lying parts pressed against coplanar neighbours, neither
        feature matching nor the rotation grid can tell where one instance
        ends -- but the through-holes can: the observed surface normal fixes
        two rotational freedoms, and one matched pair of hole centres fixes
        the remaining three. Each geometrically consistent pairing becomes a
        candidate, and the depth-map verdict picks the survivor.

        Args:
            holes: [(centre_xyz, radius_m)] observed holes near this
                surface, camera frame, centres on the visible surface.
            surface_normal: Unit normal of the surface, toward the camera.
            points: (N, 3) the surface's points.

        Returns:
            The best-verifying pose, or None.
        """
        candidates = self._hole_pair_transforms(holes, surface_normal)
        if not candidates or len(points) < 50:
            return None
        scene = make_scene_cloud(points, self.model.voxel)
        fine_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        fine_scene = fine_scene.voxel_down_sample(FINE_VOXEL)
        judged = []
        for T0 in candidates:
            T_ref = self._refine(scene, fine_scene, T0)
            judged.append((*self._judge(scene, T_ref), T_ref))
        return self._select_and_package(judged, scene, fine_scene)

    def _hole_pair_transforms(self, holes: list,
                              normal: np.ndarray) -> list:
        """Candidate scene->model transforms from hole-pair matches."""
        centres = self.model.hole_centres
        if centres is None or len(centres) < 2 or len(holes) < 2:
            return []
        axis = self.model.plate_axis
        lo, hi = self.model.plate_span
        mid = 0.5 * (lo + hi)

        out = []
        for i in range(len(holes)):
            for j in range(len(holes)):
                if i == j:
                    continue
                (A1, r1), (A2, r2) = holes[i], holes[j]
                d_obs = np.linalg.norm(A2 - A1)
                for p in range(len(centres)):
                    for q in range(len(centres)):
                        if p == q:
                            continue
                        if abs(np.linalg.norm(centres[q] - centres[p])
                               - d_obs) > 0.003:
                            continue
                        if (abs(self.model.hole_radii[p] - r1) > 0.0025
                                or abs(self.model.hole_radii[q] - r2) > 0.0025):
                            continue
                        for sign in (1.0, -1.0):
                            m_axis = sign * axis
                            offset = (hi - mid) if sign > 0 else (lo - mid)
                            V1 = centres[p] + axis * offset
                            V2 = centres[q] + axis * offset
                            R = _frame_between(V2 - V1, m_axis,
                                               A2 - A1, normal)
                            if R is None:
                                continue
                            t = A1 - R @ V1
                            T = np.eye(4)
                            T[:3, :3] = R.T
                            T[:3, 3] = -R.T @ t
                            out.append(T)
        return out

    def _select_and_package(self, judged, scene, fine_scene) -> PoseEstimate:
        # The verdict's 3 mm margin makes confidence decisive against flips
        # but noisy between near-identical poses, so confidence shortlists
        # and fitness -- the tighter measure -- picks the winner.
        top = max(v.confidence for _, v, _ in judged)
        viable = [j for j in judged if j[1].confidence >= top - CONFIDENCE_TIE]
        fit, verdict, T_ms = max(viable, key=lambda j: j[0].fitness)
        T_ms = self._polish(np.asarray(fine_scene.points), T_ms)
        fit, verdict = self._judge(scene, T_ms)
        return self._package(T_ms, fit, verdict, len(scene.points))

    def _grid_candidates(self, scene, anchor: np.ndarray | None = None,
                         keep: int = 6) -> list:
        """Best few grid orientations after coarse ICP.

        Translation per orientation: centroids aligned by default. With an
        ``anchor`` -- the cloud's closest-to-camera point, which under the
        true pose coincides with the posed model's closest-to-camera point
        -- the model's top point is pinned there instead. Anchoring removes
        the lateral ambiguity that centroid alignment suffers when the
        cloud holds more than one instance.
        """
        reg = o3d.pipelines.registration
        c_scene = np.asarray(scene.get_center())
        c_model = self._model_np.mean(axis=0)

        rotations = self._grid
        patch_normal = None
        if anchor is not None:
            rotations = self._grid_dense
            patch_normal = self._patch_normal(scene, anchor)

        scored = []
        for R0 in rotations:
            if anchor is None:
                t_co = c_scene - R0 @ c_model
            else:
                top_idx = int(np.argmin(self._coarse_np @ R0[2]))
                t_co = anchor - R0 @ self._coarse_np[top_idx]
                if patch_normal is not None:
                    # Under the true pose, the model's (locally averaged)
                    # normal at the anchor point matches the observed
                    # surface normal there; skip clear contradictions.
                    agreement = (R0 @ self._coarse_nrm_smooth[top_idx]
                                 ) @ patch_normal
                    if agreement < 0.6:
                        continue
            T = np.eye(4)
            T[:3, :3] = R0.T
            T[:3, 3] = -R0.T @ t_co
            T_icp = T
            for dist in (0.012, 0.006):
                icp = reg.registration_icp(
                    scene, self.model.coarse, dist, T_icp, self._p2l_coarse,
                    reg.ICPConvergenceCriteria(max_iteration=20))
                T_icp = icp.transformation
            # Rank by the depth-map verdict, not scene fitness: when the
            # cloud holds several instances, a wrong pose sprawled across
            # the pile explains more scene points than the right pose on
            # one instance -- but only the right pose survives free-space
            # verification.
            T_co = np.linalg.inv(T_icp)
            verdict = verify_pose(T_co[:3, :3], T_co[:3, 3],
                                  self._model_np, self._normals_np,
                                  self.depth, self.K, slope=self._slope)
            scored.append((verdict.confidence, T_icp))
        scored.sort(key=lambda s: -s[0])
        return [T for _, T in scored[:keep]]

    def _patch_normal(self, scene, anchor: np.ndarray,
                      radius: float = 0.012) -> np.ndarray | None:
        """Outward normal of the surface patch around the anchor point."""
        pts = np.asarray(scene.points)
        near = pts[np.linalg.norm(pts - anchor, axis=1) < radius]
        if len(near) < 10:
            return None
        centred = near - near.mean(axis=0)
        _, vectors = np.linalg.eigh(centred.T @ centred)
        normal = vectors[:, 0]
        # Camera sits at the origin; the visible side faces it.
        return normal if normal @ anchor < 0 else -normal

    def _polish(self, points: np.ndarray, T_ms: np.ndarray) -> np.ndarray:
        """Joint depth+edge polish of the winning pose, when enabled."""
        if self._polisher is None or not self.use_polish:
            return T_ms
        T_co = np.linalg.inv(T_ms)
        R, t = self._polisher.polish(points, T_co[:3, :3], T_co[:3, 3])
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    def _package(self, T_ms, fit, verdict, n_points) -> PoseEstimate:
        T_co = np.linalg.inv(T_ms)
        return PoseEstimate(
            R=T_co[:3, :3].copy(), t=T_co[:3, 3].copy(),
            fitness=float(fit.fitness), rmse=float(fit.inlier_rmse),
            verdict=verdict, n_points=n_points,
        )

    def distance_to_model(self, points: np.ndarray,
                          est: PoseEstimate) -> np.ndarray:
        """Distance of each camera-frame point to the posed model surface."""
        obj = (points - est.t) @ est.R   # camera -> object frame
        d, _ = self._model_tree.query(obj, workers=1)
        return d

    def refine_local(self, points: np.ndarray, est: PoseEstimate,
                     radius: float = 0.004) -> PoseEstimate:
        """Re-refine a pose on exactly the points it claims.

        In a cluttered component, ICP ran against every point that was still
        unclaimed; neighbouring instances contaminate the solve. Cropping to
        the points near the accepted pose and repeating the fine schedule
        removes that contamination.
        """
        near = points[self.distance_to_model(points, est) < radius]
        if len(near) < 100:
            return est
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(near))
        cloud = cloud.voxel_down_sample(FINE_VOXEL)
        T = np.eye(4)
        T[:3, :3] = est.R.T
        T[:3, 3] = -est.R.T @ est.t      # invert object->camera: scene->model
        for dist in ICP_FINE_SCHEDULE:
            icp = o3d.pipelines.registration.registration_icp(
                cloud, self.model.fine, dist, T, self._p2l_fine,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = icp.transformation
        T = self._polish(np.asarray(cloud.points), T)
        fit, verdict = self._judge(cloud, T)
        return self._package(T, fit, verdict, len(cloud.points))
