"""Turn verified object poses into ranked grasp poses a robot can execute.

The pose pipeline answers "where is the part". A cell needs "where do I
put the tool, on which part, first". That is three separate questions and
this module keeps them separate:

* **Which grasp on the part** -- fixed, measured offsets in the CAD frame,
  read from ``grasps.part.json``. Nothing about them depends on the scene.
* **Which instance** -- top of the pile first. The report's whole
  detection strategy registers top-of-pile instances first and carves out
  the points they explain (report.md, Domain-shift safety net); those are
  also the instances that are actually pickable, and every pick thins the
  pile for the next cycle.
* **Whether the tool can get there** -- the approach cylinder must be
  free of observed depth points that do not belong to the target
  instance. This is the only test in the module that touches the scene.

Ranking combines the three: the pose's own ``score`` (segmenter
confidence x depth verification, and at >= 0.7 worth ~0.99 precision at
5 mm on cross-validated data, analysis/score_calibration.md), how close
to the top of the pile the grasp point sits, and how much room the
approach has.

**What this deliberately does not do.** There is no model of the robot
arm, so nothing here knows whether the elbow fits; no bin geometry, so
nothing here knows about the wall the gripper would clip on the way out;
no motion planning, no reachability, no joint limits, no singularities,
no cycle-time optimisation over pick order. Those need the controller's
own kinematics and belong to it. What this module guarantees is narrower
and checkable: the grasp poses it returns sit on the part where the CAD
says they do, and the straight-line approach to each one was free of
measured obstacles at the moment the frame was taken.

Accuracy context: the pipeline's in-plane accuracy is bounded near 2 mm
by the 1 mm depth quantisation (report.md, Limitations), and the
top-scoring pose of every test scene landed within 3.2 mm
(analysis/score_calibration.md). Grasps are chosen with clearances that
survive that, not with clearances that assume the pose is exact.

Run ``python -m deploy.pick.grasp`` for the self-check on the CAD alone.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .frames import (FrameError, FramedTransform, Transform,
                     angle_between_deg)

#: Grasp kinds the planner understands. A file naming anything else is a
#: configuration error, not something to skip quietly.
SUCTION = "suction"
PARALLEL = "parallel"
GRASP_TYPES = (SUCTION, PARALLEL)

#: What the grasp file declares itself as.
GRASPS_FORMAT = "cell-grasps/1.0"

#: Edge length the CAD is subdivided to before it becomes the point cloud
#: the planner tests scene points against, metres. A query point can then
#: be at most 1.2 mm from a model point (measured against a 0.4 mm
#: resample of this CAD), well inside the 3 mm tolerance it is compared
#: with, so the tessellation of the CAD never decides whether a scene
#: point belongs to the part.
MODEL_RESOLUTION_M = 0.001


class GraspConfigError(ValueError):
    """A grasp definition file the planner refuses to run on."""


@dataclass(frozen=True)
class GraspDef:
    """One grasp on the part, in the CAD/object frame.

    See ``grasps.part.json`` for what each field means and where its
    value was measured from.
    """

    name: str
    type: str
    position: np.ndarray                  # (3,) metres, object frame
    R: np.ndarray                         # (3, 3), columns (close, y, approach)
    approach_m: float
    retreat_m: float
    tool_radius_m: float
    patch_radius_m: Optional[float] = None
    opening_width_m: Optional[float] = None
    note: str = ""

    @property
    def transform(self) -> Transform:
        """``T_object_grasp``."""
        return Transform.from_Rt(self.R, self.position)

    @property
    def approach(self) -> np.ndarray:
        """Unit direction the tool travels, object frame: the frame's +Z."""
        return self.R[:, 2].copy()

    @property
    def close_axis(self) -> np.ndarray:
        """Jaw travel direction, object frame: the frame's +X."""
        return self.R[:, 0].copy()


def load_grasps(path: str) -> List[GraspDef]:
    """Read and validate a grasp file. Every failure here is fatal.

    A cell that comes up with a mis-typed grasp file must fail on the
    launch, not on the first pick -- the same rule the pose service's
    configuration follows.
    """
    try:
        with open(path) as handle:
            doc = json.load(handle)
    except IOError as exc:
        raise GraspConfigError("cannot read grasp file %r: %s" % (path, exc))
    except ValueError as exc:
        raise GraspConfigError("grasp file %r is not valid JSON: %s"
                               % (path, exc))
    if not isinstance(doc, dict):
        raise GraspConfigError("grasp file must be a JSON object")
    fmt = str(doc.get("format", ""))
    if not fmt.startswith("cell-grasps/"):
        raise GraspConfigError("unknown grasp file format %r (expected %r)"
                               % (fmt, GRASPS_FORMAT))
    if doc.get("units") != "m":
        raise GraspConfigError("grasp file units must be 'm', got %r"
                               % (doc.get("units"),))
    if doc.get("frame") != "object":
        raise GraspConfigError(
            "grasp poses must be given in the object frame (got %r); a "
            "grasp expressed in any other frame moves with the part"
            % (doc.get("frame"),))
    raw = doc.get("grasps")
    if not isinstance(raw, list) or not raw:
        raise GraspConfigError("grasp file %r defines no grasps" % path)

    grasps, seen = [], set()
    for i, item in enumerate(raw):
        grasps.append(_parse_grasp(item, "%s[%d]" % (path, i)))
        if grasps[-1].name in seen:
            raise GraspConfigError("two grasps are both named %r"
                                   % grasps[-1].name)
        seen.add(grasps[-1].name)
    return grasps


def _parse_grasp(item: Any, where: str) -> GraspDef:
    if not isinstance(item, dict):
        raise GraspConfigError("%s is not an object" % where)
    name = str(item.get("name", ""))
    if not name:
        raise GraspConfigError("%s has no name" % where)
    kind = str(item.get("type", ""))
    if kind not in GRASP_TYPES:
        raise GraspConfigError("%s (%s) has type %r, expected one of %s"
                               % (where, name, kind, GRASP_TYPES))
    try:
        position = np.asarray(item["position"], dtype=np.float64).reshape(3)
        R = np.asarray(item["R"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError) as exc:
        raise GraspConfigError("%s (%s) needs a 3-vector 'position' and a "
                               "3x3 'R': %s" % (where, name, exc))
    try:
        Transform.from_Rt(R, position)      # orthonormality, sane magnitude
    except FrameError as exc:
        raise GraspConfigError("%s (%s): %s" % (where, name, exc))

    # The file states the approach twice -- as a vector and as R's third
    # column -- because a reader needs to see it without reading a matrix.
    # Two statements of one fact have to be checked against each other.
    for key, column in (("approach", 2), ("close_axis", 0)):
        if key not in item:
            continue
        stated = np.asarray(item[key], dtype=np.float64).reshape(3)
        if angle_between_deg(stated, R[:, column]) > 0.1:
            raise GraspConfigError(
                "%s (%s): '%s' %s disagrees with column %d of R %s"
                % (where, name, key, np.round(stated, 4), column,
                   np.round(R[:, column], 4)))

    def positive(key, required):
        value = item.get(key)
        if value is None:
            if required:
                raise GraspConfigError("%s (%s) needs %r" % (where, name, key))
            return None
        value = float(value)
        if not value > 0.0:
            raise GraspConfigError("%s (%s): %s must be positive, got %r"
                                   % (where, name, key, value))
        return value

    grasp = GraspDef(
        name=name, type=kind, position=position, R=R,
        approach_m=positive("approach_m", True),
        retreat_m=positive("retreat_m", True),
        tool_radius_m=positive("tool_radius_m", True),
        patch_radius_m=positive("patch_radius_m", kind == SUCTION),
        opening_width_m=positive("opening_width_m", kind == PARALLEL),
        note=str(item.get("note", "")))
    if kind == PARALLEL and grasp.tool_radius_m < grasp.opening_width_m / 2.0:
        raise GraspConfigError(
            "%s (%s): tool_radius_m %.4f is inside the open jaws "
            "(%.4f/2); the clearance cylinder would miss the fingers"
            % (where, name, grasp.tool_radius_m, grasp.opening_width_m))
    return grasp


@dataclass
class GraspConfig:
    """Planner settings. Every default is a measured or stated number."""

    #: How far the approach may lean off the camera's viewing axis before
    #: the grasp face is taken as turned away from the camera. The camera
    #: is fixed above the bin and the robot works the same volume, so an
    #: approach more than this far off the optical axis is on a face the
    #: camera is seeing edge-on -- a pose whose supporting evidence is
    #: thinnest exactly where the grasp needs it to be strongest.
    max_approach_tilt_deg: float = 60.0

    #: A scene point this close to the posed CAD surface is the target
    #: instance itself, not an obstacle. 3 mm is the pipeline's own
    #: threshold for "this point belongs to this pose" (report.md,
    #: Proposals: the own-mask check), so the planner and the estimator
    #: agree on what the part occupies.
    self_tolerance_m: float = 0.003

    #: Depth range over which top-of-pile preference decays to nothing.
    #: The part is 8 mm thick with a 15 mm boss, so 30 mm is about four
    #: layers: past that an instance is plainly under something.
    pile_span_m: float = 0.030

    #: Score the ranking measures distance *above*, not absolute score.
    #: A pose below the cell's accept gate is not picked at all
    #: (policy.ACCEPT_SCORE), so ranking on the raw 0..1 score would
    #: spend most of its range on poses that will never be commanded.
    #: Stretching the gated band 0.7..1.0 over the full term makes the
    #: three weights below mean what they say.
    score_reference: float = 0.70

    #: Ranking weights over three terms each normalised to 0..1. Score
    #: leads because it is the one quantity with a measured precision
    #: curve (>= 0.7 -> ~0.99 at 5 mm, analysis/score_calibration.md).
    #: Top-of-pile is next because a buried part fails mechanically
    #: however well it was localised -- and it outranks score whenever
    #: the score gap is small, which is the common case among poses that
    #: already cleared the gate. Clearance breaks ties: past about twice
    #: the tool radius, more room does not make the pick more likely to
    #: work.
    weight_score: float = 0.50
    weight_top_of_pile: float = 0.35
    weight_clearance: float = 0.15

    #: Poses below this are not planned at all. Default 0.0: the cell's
    #: accept gate lives in policy.py so there is exactly one place to
    #: change it. Raise this only to plan fewer candidates per frame.
    min_score: float = 0.0

    #: Scene points beyond this from a grasp point cannot reach its
    #: cylinder and are never examined. Kept as a constant so the
    #: neighbour search has a fixed, checkable bound.
    neighbour_margin_m: float = 0.002


@dataclass
class GraspCandidate:
    """One grasp on one instance, with everything the cell decides on.

    Rejected candidates are built too and carry ``reason``; they are what
    an engineer reads when a frame full of good poses yields no pick.
    """

    grasp_name: str
    grasp_type: str
    pose_index: int
    score: float
    #: ``T_camera_grasp`` -- where the tool must be, in camera frame.
    T_camera_grasp: Transform
    #: The same in the robot base frame; None when no hand-eye was given.
    T_base_grasp: Optional[Transform]
    #: The pose this grasp hangs off, ``T_camera_object``.
    T_camera_object: Transform
    #: Unit approach direction in the camera frame (and base, if known).
    approach_camera: np.ndarray
    approach_base: Optional[np.ndarray]
    #: Depth of the grasp point, metres. Smaller is nearer the top.
    depth_m: float
    #: Smallest distance from the approach axis to a measured point that
    #: is not the target instance, metres; inf when the cylinder is
    #: empty. Compare against ``tool_radius_m``.
    clearance_m: float
    tool_radius_m: float
    approach_tilt_deg: float
    rank: float
    accepted: bool
    reason: str = ""
    #: How far to lift back along -approach once the grasp closes; the
    #: grasp definition's retreat_m, carried so the controller does not
    #: have to reopen the file.
    retreat_m: float = 0.0

    @property
    def retreat_camera(self) -> np.ndarray:
        """Where to lift to after the grasp closes, camera frame."""
        return self.T_camera_grasp.t - self.approach_camera * self.retreat_m

    def to_dict(self) -> Dict[str, Any]:
        """Standard-library types only, the way schema.py sends a pose."""
        base = self.T_base_grasp
        return {"grasp": self.grasp_name, "type": self.grasp_type,
                "pose_index": self.pose_index, "score": self.score,
                "T_camera_grasp": self.T_camera_grasp.to_list(),
                "T_base_grasp": None if base is None else base.to_list(),
                "quaternion_xyzw_camera":
                    self.T_camera_grasp.quaternion_xyzw(),
                "quaternion_xyzw_base":
                    None if base is None else base.quaternion_xyzw(),
                "approach_camera": [float(v) for v in self.approach_camera],
                "depth_m": self.depth_m,
                "clearance_m": (None if not np.isfinite(self.clearance_m)
                                else float(self.clearance_m)),
                "tool_radius_m": self.tool_radius_m,
                "approach_tilt_deg": self.approach_tilt_deg,
                "retreat_m": self.retreat_m, "rank": self.rank,
                "accepted": self.accepted, "reason": self.reason}


class GraspPlanner:
    """Plans grasps for one part, reused across frames.

    Everything that does not change between frames -- the CAD sample, its
    KD-tree, the grasp definitions -- is built once here. Per frame the
    planner builds one KD-tree of the scene and does a bounded neighbour
    query per candidate, which is what keeps it inside a 4-8 s cycle.
    """

    def __init__(self, model_path: str, grasps_path: str,
                 config: Optional[GraspConfig] = None):
        """Args:
            model_path: The CAD, metres, object frame (model/3d_model.ply).
            grasps_path: A ``cell-grasps/1.x`` file, normally the
                ``grasps.part.json`` beside this module.
            config: Planner settings; defaults are the shipped ones.

        Raises:
            GraspConfigError: the CAD or the grasp file cannot be used.
        """
        self.config = config or GraspConfig()
        self.grasps = load_grasps(grasps_path)
        try:
            mesh = trimesh.load(model_path, force="mesh")
        except Exception as exc:                      # trimesh is broad
            raise GraspConfigError("cannot read CAD %r: %s"
                                   % (model_path, exc))
        if mesh.vertices is None or len(mesh.vertices) == 0:
            raise GraspConfigError("CAD %r has no vertices" % model_path)

        # A deterministic surface cloud, not a random sample: two runs of
        # the cell must agree on what the part occupies. The raw mesh
        # tessellates its large flat faces into a few big triangles, so
        # its vertices alone leave gaps of a couple of millimetres in
        # exactly the places a suction cup lands; subdividing to a fixed
        # edge length fixes the resolution at MODEL_RESOLUTION_M
        # everywhere, independent of how the CAD was triangulated.
        vertices, faces = trimesh.remesh.subdivide_to_size(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces), max_edge=MODEL_RESOLUTION_M)
        # Subdivision leaves many near-duplicate points on the flat
        # faces; one representative per grid cell keeps the resolution
        # and drops the count by an order of magnitude.
        vertices = np.asarray(vertices, dtype=np.float64)
        cells = np.round(vertices / MODEL_RESOLUTION_M).astype(np.int64)
        _, keep = np.unique(cells, axis=0, return_index=True)
        self._model_points = vertices[np.sort(keep)]
        self._model_tree = cKDTree(self._model_points)
        self._model_radius = float(np.linalg.norm(self._model_points,
                                                  axis=1).max())
        self._check_grasps_on_surface()
        #: Why each candidate of the last frame was dropped. The cell logs
        #: this when a frame yields no pick.
        self.last_rejections = []               # type: List[str]

    def _check_grasps_on_surface(self) -> None:
        """A grasp point that is not on the part is a typo, and a typo
        here drives a tool into a bin.

        A suction contact must sit on the surface, within twice the model
        cloud's own resolution -- tighter than that and the check would
        be reporting how the CAD was tessellated. A parallel grasp sits
        inside the part by construction (mid plate thickness, 4 mm in on
        this part), so it is only required to be within the material;
        12 mm catches a coordinate typed with the wrong sign or unit
        without objecting to a legitimate interior point.
        """
        for grasp in self.grasps:
            distance = float(self._model_tree.query(grasp.position)[0])
            limit = (2.0 * MODEL_RESOLUTION_M if grasp.type == SUCTION
                     else 0.012)
            if distance > limit:
                raise GraspConfigError(
                    "grasp %r sits %.1f mm from the CAD surface (limit "
                    "%.1f mm for a %s grasp) -- check its position against "
                    "the model" % (grasp.name, distance * 1000.0,
                                   limit * 1000.0, grasp.type))

    # -- planning --------------------------------------------------------

    def plan(self, poses: Sequence[Any], scene: np.ndarray,
             T_base_camera: Optional[FramedTransform] = None,
             K: Optional[np.ndarray] = None,
             keep_rejected: bool = False) -> List[GraspCandidate]:
        """Rank the grasps available in one frame.

        Args:
            poses: ``T_camera_object`` estimates, best first. Anything
                carrying ``R``, ``t`` and ``score`` works: a
                ``PoseEstimateDTO`` from the pose service, a
                ``PoseEstimate`` from the estimator, or a plain dict as
                submission.json stores them.
            scene: The frame's measured geometry, either an (N, 3)
                camera-frame point cloud or the (H, W) depth map in
                metres, in which case ``K`` is required.
            T_base_camera: The hand-eye transform, named. When given,
                every candidate also carries its pose in the robot base
                frame. When None the candidates are camera-frame only --
                which is all the ranking needs.
            K: (3, 3) intrinsics, required when ``scene`` is a depth map.
            keep_rejected: Also return the rejected candidates, after the
                accepted ones, each with its ``reason``.

        Returns:
            Accepted candidates, best first, then rejected ones if asked
            for. An empty list means no viable grasp this frame; the
            reasons are in :attr:`last_rejections`. Never raises on frame
            data -- a bad frame is a frame with no pick, not a crash.
        """
        self.last_rejections = []
        try:
            points = self._scene_points(scene, K)
        except (ValueError, TypeError) as exc:
            self.last_rejections.append("unusable scene: %s" % exc)
            return []
        if T_base_camera is not None:
            try:
                _check_hand_eye(T_base_camera)
            except FrameError as exc:
                self.last_rejections.append("unusable hand-eye: %s" % exc)
                return []

        tree = cKDTree(points) if points.size else None
        candidates = []                     # type: List[GraspCandidate]
        for index, pose in enumerate(poses):
            try:
                T_camera_object, score = _pose_fields(pose)
            except (FrameError, KeyError, TypeError, ValueError) as exc:
                self.last_rejections.append("pose %d unusable: %s"
                                            % (index, exc))
                continue
            if score < self.config.min_score:
                self.last_rejections.append(
                    "pose %d: score %.3f below %.3f"
                    % (index, score, self.config.min_score))
                continue
            for grasp in self.grasps:
                candidates.append(self._evaluate(grasp, index, score,
                                                 T_camera_object, points,
                                                 tree, T_base_camera))
        self._rank(candidates)
        for candidate in candidates:
            if not candidate.accepted:
                self.last_rejections.append(
                    "pose %d / %s: %s" % (candidate.pose_index,
                                          candidate.grasp_name,
                                          candidate.reason))
        accepted = sorted([c for c in candidates if c.accepted],
                          key=lambda c: -c.rank)
        if not keep_rejected:
            return accepted
        return accepted + sorted([c for c in candidates if not c.accepted],
                                 key=lambda c: -c.score)

    def _evaluate(self, grasp: GraspDef, index: int, score: float,
                  T_camera_object: Transform, points: np.ndarray,
                  tree: Optional[cKDTree],
                  T_base_camera: Optional[FramedTransform]) -> GraspCandidate:
        T_camera_grasp = T_camera_object @ grasp.transform
        approach_cam = T_camera_grasp.R[:, 2]
        contact = T_camera_grasp.t

        T_base_grasp, approach_base = None, None
        if T_base_camera is not None:
            T_base_grasp = T_base_camera.transform @ T_camera_grasp
            approach_base = T_base_grasp.R[:, 2]

        # The camera looks along +Z (OpenCV). An approach that leans far
        # off that axis is aimed at a face the camera barely saw.
        tilt = angle_between_deg(approach_cam, (0.0, 0.0, 1.0))

        candidate = GraspCandidate(
            grasp_name=grasp.name, grasp_type=grasp.type, pose_index=index,
            score=score, T_camera_grasp=T_camera_grasp,
            T_base_grasp=T_base_grasp, T_camera_object=T_camera_object,
            approach_camera=approach_cam, approach_base=approach_base,
            depth_m=float(contact[2]), clearance_m=float("inf"),
            tool_radius_m=grasp.tool_radius_m, approach_tilt_deg=tilt,
            rank=0.0, accepted=False, retreat_m=grasp.retreat_m)

        if tilt > self.config.max_approach_tilt_deg:
            candidate.reason = ("approach %.1f deg off the camera axis "
                                "(limit %.1f): this face is turned away"
                                % (tilt, self.config.max_approach_tilt_deg))
            return candidate

        candidate.clearance_m = self._clearance(grasp, T_camera_object,
                                                contact, approach_cam,
                                                points, tree)
        if candidate.clearance_m < grasp.tool_radius_m:
            candidate.reason = ("approach blocked: a measured point %.1f mm "
                                "off the axis, inside the %.1f mm tool "
                                "radius" % (candidate.clearance_m * 1000.0,
                                            grasp.tool_radius_m * 1000.0))
            return candidate

        candidate.accepted = True
        return candidate

    def _clearance(self, grasp: GraspDef, T_camera_object: Transform,
                   contact: np.ndarray, approach: np.ndarray,
                   points: np.ndarray, tree: Optional[cKDTree]) -> float:
        """Smallest radial distance from the approach axis to a measured
        point that is not the target instance.

        The tool sweeps a cylinder of radius ``tool_radius_m`` reaching
        ``approach_m`` back along the approach from the contact point.
        Points of the target instance itself are excluded -- the tool is
        allowed to touch the part it is picking -- by lifting the local
        points into the object frame and asking the CAD tree how far they
        are from the surface.

        Returns ``inf`` when nothing else is in the cylinder.
        """
        if tree is None:
            return float("inf")
        length = grasp.approach_m
        radius = grasp.tool_radius_m + self.config.neighbour_margin_m
        # One ball query around the cylinder's bounding sphere: the whole
        # per-candidate cost, and it is what keeps a 300k-point frame off
        # the critical path.
        centre = contact - approach * (length / 2.0)
        ball = float(np.sqrt((length / 2.0) ** 2 + radius ** 2))
        local = tree.query_ball_point(centre, ball)
        if not local:
            return float("inf")

        q = points[local] - contact
        along = -(q @ approach)                       # >0 on the tool side
        band = (along > 1e-4) & (along < length)
        if not band.any():
            return float("inf")
        q = q[band]
        radial = np.linalg.norm(q - np.outer(-along[band], approach), axis=1)

        # Drop the target instance's own surface.
        world = q + contact
        in_object = T_camera_object.inverse().apply(world)
        near_part = np.linalg.norm(in_object, axis=1) <= \
            self._model_radius + self.config.self_tolerance_m
        if near_part.any():
            distance = self._model_tree.query(in_object[near_part])[0]
            keep = np.ones(len(in_object), dtype=bool)
            keep[np.nonzero(near_part)[0]] = distance > \
                self.config.self_tolerance_m
            radial = radial[keep]
        if radial.size == 0:
            return float("inf")
        return float(radial.min())

    def _rank(self, candidates: Sequence[GraspCandidate]) -> None:
        """Score every accepted candidate on the three axes that matter."""
        accepted = [c for c in candidates if c.accepted]
        if not accepted:
            return
        top = min(c.depth_m for c in accepted)
        cfg = self.config
        span = max(1e-6, 1.0 - cfg.score_reference)
        for c in accepted:
            belief = min(1.0, max(0.0, (c.score - cfg.score_reference) / span))
            height = 1.0 - min(1.0, max(0.0, (c.depth_m - top)
                                        / cfg.pile_span_m))
            room = (1.0 if not np.isfinite(c.clearance_m)
                    else min(1.0, c.clearance_m / (2.0 * c.tool_radius_m)))
            c.rank = (cfg.weight_score * belief
                      + cfg.weight_top_of_pile * height
                      + cfg.weight_clearance * room)

    # -- input normalisation ---------------------------------------------

    def _scene_points(self, scene: np.ndarray,
                      K: Optional[np.ndarray]) -> np.ndarray:
        """Accept a point cloud or a depth map; return (N, 3) metres."""
        array = np.asarray(scene)
        if array.ndim == 2 and array.shape[1] == 3 and array.shape[0] != 3:
            points = np.ascontiguousarray(array, dtype=np.float64)
        elif array.ndim == 2:
            if K is None:
                raise ValueError("a depth map needs K to be back-projected")
            points = backproject(array.astype(np.float64),
                                 np.asarray(K, dtype=np.float64), array > 0)
        else:
            raise ValueError("scene must be (N, 3) points or an (H, W) "
                             "depth map, got %r" % (array.shape,))
        # Some SDKs deliver unmeasured pixels as NaN or inf rather than
        # zero. A KD-tree refuses to build on them, and a frame is not
        # worth losing over a handful of bad pixels.
        finite = np.isfinite(points).all(axis=1)
        return points if finite.all() else points[finite]


def _pose_fields(pose: Any) -> Tuple[Transform, float]:
    """Pull ``T_camera_object`` and ``score`` out of whatever was passed."""
    if isinstance(pose, dict):
        R, t = pose["R"], pose["t"]
        score = float(pose.get("score", pose.get("submission_score", 1.0)))
    else:
        R, t = pose.R, pose.t
        score = float(getattr(pose, "score",
                              getattr(pose, "submission_score", 1.0)))
    return Transform.from_Rt(np.asarray(R, dtype=np.float64),
                             np.asarray(t, dtype=np.float64)), score


def _check_hand_eye(T_base_camera: FramedTransform) -> None:
    if not isinstance(T_base_camera, FramedTransform):
        raise FrameError("T_base_camera must be a FramedTransform so the "
                         "composition can be checked; got %r"
                         % type(T_base_camera).__name__)
    if (T_base_camera.parent, T_base_camera.child) != ("base", "camera"):
        raise FrameError("expected T_base_camera, got %s"
                         % T_base_camera.name)


# The pipeline's own back-projection, so the planner and the estimator
# lift the same pixels into the same points.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from src.scene_io import backproject                          # noqa: E402


def default_grasps_path() -> str:
    """The grasp file shipped beside this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "grasps.part.json")


def _self_check() -> int:
    """Geometry checks that need no scene, then a synthetic pile.

    The real evidence for this module is the run on the test scenes (see
    the cell-integration report); what a self-check can prove on its own
    is that the grasp file agrees with the CAD, that the frames compose
    the way the file says, and that the clearance test rejects exactly
    the approach a neighbour blocks and no other.
    """
    failures = 0
    here = os.path.dirname(os.path.abspath(__file__))
    cad = os.path.join(os.path.dirname(os.path.dirname(here)),
                       "model", "3d_model.ply")
    print("grasp.py self-check")

    def check(name, ok, detail=""):
        nonlocal failures
        print("  %-52s %s%s" % (name, "ok" if ok else "FAIL",
                                "" if ok else "  " + detail))
        if not ok:
            failures += 1

    planner = GraspPlanner(cad, default_grasps_path())
    print("  %d grasps, model cloud %d points, radius %.1f mm"
          % (len(planner.grasps), len(planner._model_points),
             planner._model_radius * 1000.0))
    for grasp in planner.grasps:
        distance = float(planner._model_tree.query(grasp.position)[0]) * 1000.0
        print("    %-28s %-8s contact %.2f mm from the CAD surface, "
              "tool r %.1f mm" % (grasp.name, grasp.type, distance,
                                  grasp.tool_radius_m * 1000.0))

    # A pose that presents the plate to the camera: the object's +Y (the
    # plate normal) is turned to face the camera, so the "from_top"
    # approaches run straight down the optical axis.
    R_facing = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])
    facing = {"R": R_facing.tolist(), "t": [0.0, 0.0, 0.7], "score": 0.9}
    empty = np.zeros((0, 3))
    plan = planner.plan([facing], empty, keep_rejected=True)
    by_name = dict((c.grasp_name, c) for c in plan)
    worst = 0.0
    for grasp in planner.grasps:
        expected = Transform.from_Rt(R_facing, [0.0, 0.0, 0.7]) \
            @ grasp.transform
        worst = max(worst, expected.delta_to(
            by_name[grasp.name].T_camera_grasp)[1])
    check("grasp poses compose as T_camera_object @ T_object_grasp",
          worst < 1e-12, "%.2e m" % worst)

    accepted = set(c.grasp_name for c in plan if c.accepted)
    check("plate facing the camera: only the reachable faces survive",
          accepted == {"suction_plate_top", "parallel_stem_from_top",
                       "parallel_crossbar_from_top"},
          str(sorted(accepted)))
    turned = {"R": (np.diag([1.0, -1.0, -1.0]) @ R_facing).tolist(),
              "t": [0.0, 0.0, 0.7], "score": 0.9}
    flipped = set(c.grasp_name for c in planner.plan([turned], empty))
    check("part turned over: the other faces survive instead",
          flipped == {"suction_plate_bottom", "parallel_stem_from_bottom"},
          str(sorted(flipped)))

    # Hand-eye composition.
    T_base_camera = FramedTransform(
        "base", "camera",
        Transform.from_Rt(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0]]), [0.5, 0.0, 1.2]))
    with_base = planner.plan([facing], empty, T_base_camera=T_base_camera)
    top = with_base[0]
    expected = T_base_camera.transform @ top.T_camera_grasp
    check("T_base_grasp = T_base_camera @ T_camera_grasp",
          top.T_base_grasp is not None
          and expected.delta_to(top.T_base_grasp)[1] < 1e-12)
    try:
        planner.plan([facing], empty,
                     T_base_camera=FramedTransform("base", "flange",
                                                   Transform.identity()))
        check("a mis-named hand-eye is refused",
              bool(planner.last_rejections)
              and "hand-eye" in planner.last_rejections[0],
              str(planner.last_rejections))
    except Exception as exc:                              # must not raise
        check("a mis-named hand-eye is refused", False, repr(exc))

    # A synthetic obstacle: measured points strung along the suction
    # grasp's own approach axis, offset sideways by a known distance.
    reference = by_name["suction_plate_top"]
    contact, approach = reference.T_camera_grasp.t, reference.approach_camera
    sideways = np.array([1.0, 0.0, 0.0])
    axis_points = contact[None, :] - approach[None, :] * \
        np.linspace(0.005, 0.045, 40)[:, None]
    for offset_mm, expect_block in ((6.0, True), (12.0, False)):
        obstacle = axis_points + sideways * (offset_mm / 1000.0)
        result = planner.plan([facing], obstacle, keep_rejected=True)
        candidate = dict((c.grasp_name, c) for c in result)[
            "suction_plate_top"]
        check("an obstacle %.0f mm off the axis %s the 8 mm tool"
              % (offset_mm, "blocks" if expect_block else "clears"),
              (not candidate.accepted) == expect_block,
              "clearance %.1f mm" % (candidate.clearance_m * 1000.0))

    # The part's own surface must never block its own grasp.
    posed = Transform.from_Rt(R_facing, [0.0, 0.0, 0.7])
    surface = posed.apply(planner._model_points)
    result = planner.plan([facing], surface, keep_rejected=True)
    own = dict((c.grasp_name, c) for c in result)["suction_plate_top"]
    check("the target instance does not block itself", own.accepted,
          own.reason)
    # ...but a second part lying on top of it does: the same CAD, moved
    # 10 mm up the approach path towards the camera.
    neighbour = np.vstack([surface, surface - approach * 0.010])
    result = planner.plan([facing], neighbour, keep_rejected=True)
    covered = dict((c.grasp_name, c) for c in result)["suction_plate_top"]
    check("a part lying on top blocks the grasp", not covered.accepted,
          "clearance %.1f mm" % (covered.clearance_m * 1000.0))

    # Ranking: the same pose nearer the camera must outrank the far one.
    near = {"R": R_facing.tolist(), "t": [0.0, 0.0, 0.680], "score": 0.80}
    far = {"R": R_facing.tolist(), "t": [0.0, 0.0, 0.720], "score": 0.80}
    order = planner.plan([far, near], empty)
    check("top of the pile ranks first at equal score",
          order[0].pose_index == 1, "pose %d first" % order[0].pose_index)
    strong_far = {"R": R_facing.tolist(), "t": [0.0, 0.0, 0.720],
                  "score": 0.95}
    weak_near = {"R": R_facing.tolist(), "t": [0.0, 0.0, 0.680],
                 "score": 0.72}
    order = planner.plan([strong_far, weak_near], empty)
    check("a 0.95-vs-0.72 score gap outranks 40 mm of pile depth",
          order[0].pose_index == 0, "pose %d first" % order[0].pose_index)
    close_near = dict(weak_near, score=0.90)
    order = planner.plan([strong_far, close_near], empty)
    check("a 0.95-vs-0.90 gap does not: top of the pile wins",
          order[0].pose_index == 1, "pose %d first" % order[0].pose_index)

    # Bad frames degrade, bad configuration raises.
    for label, scene in (("an empty cloud", np.zeros((0, 3))),
                         ("a wrong-shaped array", np.zeros((5, 7))),
                         ("a NaN-filled cloud", np.full((100, 3), np.nan))):
        try:
            planner.plan([facing], scene)
            check("%s does not raise" % label, True)
        except Exception as exc:
            check("%s does not raise" % label, False, repr(exc))
    try:
        planner.plan([{"R": [[1, 2], [3, 4]], "t": [0, 0, 1], "score": 1.0}],
                     empty)
        check("a malformed pose is reported, not raised",
              bool(planner.last_rejections), str(planner.last_rejections))
    except Exception as exc:
        check("a malformed pose is reported, not raised", False, repr(exc))

    import tempfile
    with open(default_grasps_path()) as handle:
        doc = json.load(handle)
    with tempfile.TemporaryDirectory() as tmp:
        for label, mutate in (
                ("units in millimetres", lambda d: d.update({"units": "mm"})),
                ("frame not the object frame",
                 lambda d: d.update({"frame": "camera"})),
                ("an unknown grasp type",
                 lambda d: d["grasps"][0].update({"type": "magnetic"})),
                ("approach disagreeing with R",
                 lambda d: d["grasps"][0].update({"approach": [1, 0, 0]})),
                ("a suction grasp with no patch radius",
                 lambda d: d["grasps"][0].update({"patch_radius_m": None})),
                ("a grasp 30 mm off the part",
                 lambda d: d["grasps"][0].update({"position": [0.2, 0, 0]}))):
            payload = json.loads(json.dumps(doc))
            mutate(payload)
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as handle:
                json.dump(payload, handle)
            try:
                GraspPlanner(cad, path)
                check("refuses %s" % label, False, "loaded")
            except GraspConfigError as exc:
                print("  refuses %-32s %s..." % (label + ":", str(exc)[:40]))

    print("  %d failure(s)" % failures)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _self_check() else 0)
