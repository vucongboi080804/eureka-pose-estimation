"""Rigid transforms, and the frame names that keep them the right way round.

Every pose in this repository is ``T_camera_object`` -- ``p_camera = R @
p_cad + t``, OpenCV camera convention. A cell adds two more frames (the
robot base and the tool) and that is enough for a silent inversion to put
a gripper 40 mm off the part with no error anywhere in the log. So a
transform here carries the two frames it maps between, composition
refuses to join transforms whose frames do not meet, and the only legal
reading of

    T_base_object = T_base_camera @ T_camera_object

is the one that type-checks at run time.

:class:`Transform` is the algebra (4x4 homogeneous, numpy, nothing else).
:class:`FramedTransform` is the same thing with a ``parent`` and a
``child`` name attached; it is what crosses a module boundary.

Rotations are validated on construction rather than trusted: a
calibration file edited by hand, a quaternion that was never normalised
and a matrix transposed on the way in all present as a non-orthonormal
R, and all three are cheaper to catch here than in the field.

Run ``python -m deploy.pick.frames`` for the self-check.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

#: Largest tolerated deviation of ``R.T @ R`` from the identity, and of
#: ``det(R)`` from +1. A rotation built in float64 and composed a few
#: dozen deep stays inside 1e-9, so 1e-6 never rejects honest arithmetic;
#: and 1e-6 of skew is 0.2 millidegrees, two thousand times finer than
#: the 0.5 deg a hand-eye calibration is allowed to be off by
#: (calibration.MAX_ROTATION_RESIDUAL_DEG), so nothing usable slips past.
ORTHONORMAL_TOL = 1e-6

#: Beyond this a translation is a units mistake, not a pose: the cell's
#: whole working volume -- camera 0.7 m above the tray, robot reach a
#: couple of metres -- fits in 10 m. Catching millimetres-fed-as-metres
#: at construction is the point.
MAX_TRANSLATION_M = 10.0


class FrameError(ValueError):
    """A transform that is not a rigid motion, or frames that do not meet."""


class Transform:
    """A rigid motion as a 4x4 homogeneous matrix.

    Immutable by convention: every operation returns a new instance and
    :attr:`matrix` hands out a copy, so a transform stored in a
    calibration object cannot be mutated by whoever borrowed it.
    """

    __slots__ = ("_m",)

    def __init__(self, matrix: np.ndarray, check: bool = True):
        """Args:
            matrix: (4, 4) homogeneous transform.
            check: Validate the rotation block and the bottom row. Only
                pass False for a matrix this module just built.
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise FrameError("transform must be 4x4, got %r" % (m.shape,))
        if check:
            _check_rotation(m[:3, :3])
            _check_translation(m[:3, 3])
            if not np.allclose(m[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
                raise FrameError("bottom row must be [0 0 0 1], got %s"
                                 % np.array2string(m[3], precision=6))
        self._m = m.copy()
        self._m.flags.writeable = False

    # -- construction ----------------------------------------------------

    @classmethod
    def from_Rt(cls, R: np.ndarray, t: Sequence[float],
                check: bool = True) -> "Transform":
        """Assemble from a 3x3 rotation and a 3-vector translation."""
        rot = np.asarray(R, dtype=np.float64)
        trans = np.asarray(t, dtype=np.float64).reshape(-1)
        if rot.shape != (3, 3):
            raise FrameError("R must be 3x3, got %r" % (rot.shape,))
        if trans.shape != (3,):
            raise FrameError("t must have 3 entries, got %r" % (trans.shape,))
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = trans
        return cls(m, check=check)

    @classmethod
    def identity(cls) -> "Transform":
        return cls(np.eye(4), check=False)

    @classmethod
    def from_quaternion(cls, quat_xyzw: Sequence[float],
                        t: Sequence[float]) -> "Transform":
        """Build from ``(x, y, z, w)`` -- the order ROS and most robot
        controllers speak. The quaternion is normalised first, so a
        controller's four rounded decimals do not trip the orthonormality
        check."""
        q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
        if q.shape != (4,):
            raise FrameError("quaternion must have 4 entries (x, y, z, w)")
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            raise FrameError("quaternion has zero norm")
        x, y, z, w = q / n
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return cls.from_Rt(R, t)

    @classmethod
    def from_list(cls, rows: Sequence[Sequence[float]]) -> "Transform":
        """Rebuild from :meth:`to_list` (JSON round-trip)."""
        return cls(np.asarray(rows, dtype=np.float64))

    # -- accessors -------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """A writable copy of the 4x4."""
        return self._m.copy()

    @property
    def R(self) -> np.ndarray:
        return self._m[:3, :3].copy()

    @property
    def t(self) -> np.ndarray:
        return self._m[:3, 3].copy()

    def to_Rt(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.R, self.t

    def to_list(self) -> List[List[float]]:
        """Nested lists of floats, for JSON. See schema.py: nothing but
        standard-library types crosses a process boundary."""
        return [[float(v) for v in row] for row in self._m]

    def quaternion_xyzw(self) -> List[float]:
        """The rotation as ``(x, y, z, w)``, sign-fixed to ``w >= 0``.

        Shepperd's method: pivot on the largest of the four candidate
        denominators, which is what keeps the conversion conditioned near
        the 180-degree rotations this part's near-symmetry produces.
        """
        m = self._m
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        sign = -1.0 if w < 0.0 else 1.0     # q and -q are the same rotation
        return [sign * x, sign * y, sign * z, sign * w]

    # -- algebra ---------------------------------------------------------

    def inverse(self) -> "Transform":
        """The inverse, built from the transpose rather than a solve: for
        a rigid motion that is exact and cannot degrade the rotation."""
        R = self._m[:3, :3]
        m = np.eye(4)
        m[:3, :3] = R.T
        m[:3, 3] = -R.T @ self._m[:3, 3]
        return Transform(m, check=False)

    def compose(self, other: "Transform") -> "Transform":
        """``self @ other``: apply ``other`` first, then ``self``."""
        if not isinstance(other, Transform):
            raise FrameError("can only compose with a Transform, got %r"
                             % type(other).__name__)
        return Transform(self._m @ other._m, check=False)

    __matmul__ = compose

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map points. Accepts (3,) or (N, 3); returns the same shape.

        Written as ``P @ R.T + t`` so the (N, 3) layout the whole
        pipeline uses never has to be transposed into columns.
        """
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p = p.reshape(1, 3) if single else p
        if p.ndim != 2 or p.shape[1] != 3:
            raise FrameError("points must be (3,) or (N, 3), got %r"
                             % (np.asarray(points).shape,))
        out = p @ self._m[:3, :3].T + self._m[:3, 3]
        return out[0] if single else out

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate directions -- no translation. An approach axis is a
        direction, and translating one is the classic silent bug."""
        v = np.asarray(vectors, dtype=np.float64)
        single = v.ndim == 1
        v = v.reshape(1, 3) if single else v
        out = v @ self._m[:3, :3].T
        return out[0] if single else out

    # -- measurement -----------------------------------------------------

    def rotation_angle_deg(self) -> float:
        """The rotation's angle about its own axis, degrees."""
        return rotation_angle_deg(self._m[:3, :3])

    @property
    def translation_norm_m(self) -> float:
        return float(np.linalg.norm(self._m[:3, 3]))

    def delta_to(self, other: "Transform") -> Tuple[float, float]:
        """How far ``other`` is from ``self``: (degrees, metres).

        The rotation part is the angle of ``self^-1 @ other``, which is
        the only frame-independent way to say "these two poses differ".
        """
        d = self.inverse() @ other
        return d.rotation_angle_deg(), d.translation_norm_m

    def __repr__(self) -> str:
        t = self._m[:3, 3]
        return ("Transform(rot=%.3f deg, t=[%.4f, %.4f, %.4f] m)"
                % (self.rotation_angle_deg(), t[0], t[1], t[2]))


class FramedTransform:
    """A :class:`Transform` that knows which frames it maps between.

    ``FramedTransform("base", "camera", T)`` is ``T_base_camera``: it
    takes a point expressed in the camera frame and gives it in the base
    frame. Composition is only defined when the left operand's child is
    the right operand's parent, which is exactly the rule a reader
    applies by eye when reading ``T_base_camera @ T_camera_object`` --
    made enforceable.
    """

    __slots__ = ("parent", "child", "transform")

    def __init__(self, parent: str, child: str, transform: Transform):
        if not parent or not child:
            raise FrameError("both frame names must be non-empty")
        if parent == child:
            raise FrameError("parent and child frames are both %r" % parent)
        if not isinstance(transform, Transform):
            raise FrameError("transform must be a Transform, got %r"
                             % type(transform).__name__)
        self.parent = parent
        self.child = child
        self.transform = transform

    @property
    def name(self) -> str:
        return "T_%s_%s" % (self.parent, self.child)

    def inverse(self) -> "FramedTransform":
        return FramedTransform(self.child, self.parent,
                               self.transform.inverse())

    def compose(self, other: "FramedTransform") -> "FramedTransform":
        if not isinstance(other, FramedTransform):
            raise FrameError("can only compose with a FramedTransform, got %r"
                             % type(other).__name__)
        if self.child != other.parent:
            raise FrameError(
                "cannot compose %s with %s: %r does not meet %r "
                "(did you mean %s.inverse()?)"
                % (self.name, other.name, self.child, other.parent,
                   other.name))
        return FramedTransform(self.parent, other.child,
                               self.transform @ other.transform)

    __matmul__ = compose

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map points given in the child frame into the parent frame."""
        return self.transform.apply(points)

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        return self.transform.rotate(vectors)

    def __repr__(self) -> str:
        return "%s: %r" % (self.name, self.transform)


def framed(parent: str, child: str, matrix_or_transform) -> FramedTransform:
    """Name a transform. Accepts a :class:`Transform` or a 4x4."""
    t = (matrix_or_transform if isinstance(matrix_or_transform, Transform)
         else Transform(matrix_or_transform))
    return FramedTransform(parent, child, t)


def chain(*links: FramedTransform) -> FramedTransform:
    """Compose a chain left to right, checking every joint.

    ``chain(T_base_camera, T_camera_object)`` -> ``T_base_object``.
    """
    if not links:
        raise FrameError("chain() needs at least one transform")
    out = links[0]
    for link in links[1:]:
        out = out @ link
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix about its own axis, degrees.

    ``arccos`` of the trace is ill-conditioned near 0 and 180 degrees,
    but the alternative (an axis-angle extraction) is worse near the same
    places and longer; the clip is what keeps a matrix that is
    orthonormal only to 1e-9 from producing a NaN.
    """
    rot = np.asarray(R, dtype=np.float64)
    if rot.shape != (3, 3):
        raise FrameError("R must be 3x3, got %r" % (rot.shape,))
    cos = (float(np.trace(rot)) - 1.0) / 2.0
    return float(np.degrees(math.acos(max(-1.0, min(1.0, cos)))))


def angle_between_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle between two direction vectors, degrees."""
    u = np.asarray(a, dtype=np.float64).reshape(3)
    v = np.asarray(b, dtype=np.float64).reshape(3)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        raise FrameError("cannot take the angle of a zero-length direction")
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(math.acos(max(-1.0, min(1.0, cos)))))


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Nearest rotation to ``R`` (polar decomposition via SVD).

    For repairing a rotation a solver returned with 1e-7 of drift -- not
    for rescuing a matrix that is wrong. Callers check the correction
    they got away with; this module never calls it silently.
    """
    u, _, vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = u @ vt
    if np.linalg.det(out) < 0:          # keep a right-handed frame
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def frame_from_axes(z_axis: Sequence[float], x_axis: Sequence[float]
                    ) -> np.ndarray:
    """Rotation whose columns are (x, y, z) with ``y = z x x``.

    Used to build a grasp frame from the two directions that mean
    something physically -- the approach and the closing axis -- instead
    of writing nine numbers by hand. ``x_axis`` is orthogonalised against
    ``z_axis``, so a hand-written pair that is a degree off still yields
    an exact rotation.
    """
    z = np.asarray(z_axis, dtype=np.float64).reshape(3)
    x = np.asarray(x_axis, dtype=np.float64).reshape(3)
    nz = np.linalg.norm(z)
    if nz < 1e-12:
        raise FrameError("z_axis has zero length")
    z = z / nz
    x = x - np.dot(x, z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        raise FrameError("x_axis is parallel to z_axis; the frame is "
                         "underdetermined")
    x = x / nx
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _check_rotation(R: np.ndarray) -> None:
    err = float(np.abs(R.T @ R - np.eye(3)).max())
    det = float(np.linalg.det(R))
    if err > ORTHONORMAL_TOL or abs(det - 1.0) > ORTHONORMAL_TOL:
        raise FrameError(
            "rotation is not orthonormal: max|R'R - I| = %.3e, det = %.9f "
            "(tolerance %.0e). A transposed matrix, an unnormalised "
            "quaternion or a hand-edited calibration file all look like "
            "this." % (err, det, ORTHONORMAL_TOL))


def _check_translation(t: np.ndarray) -> None:
    if not np.all(np.isfinite(t)):
        raise FrameError("translation is not finite: %s" % (t,))
    if float(np.linalg.norm(t)) > MAX_TRANSLATION_M:
        raise FrameError(
            "translation %.1f m exceeds %.1f m -- millimetres fed as metres?"
            % (float(np.linalg.norm(t)), MAX_TRANSLATION_M))


def _self_check() -> int:
    """Round-trip, composition and frame-guard checks. Returns failures."""
    rng = np.random.default_rng(7)
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        print("  %-46s %s%s" % (name, "ok" if ok else "FAIL",
                                "" if ok else "  " + detail))
        if not ok:
            failures += 1

    def random_transform(scale=0.4):
        q = rng.normal(size=4)
        return Transform.from_quaternion(q / np.linalg.norm(q),
                                         rng.normal(size=3) * scale)

    print("frames.py self-check")
    a, b = random_transform(), random_transform()

    ident = a @ a.inverse()
    d_rot, d_t = Transform.identity().delta_to(ident)
    check("inverse round-trip A @ A^-1 == I", d_rot < 1e-9 and d_t < 1e-12,
          "%.3e deg, %.3e m" % (d_rot, d_t))

    pts = rng.normal(size=(500, 3)) * 0.1
    back = a.inverse().apply(a.apply(pts))
    check("apply round-trip on 500 points",
          float(np.abs(back - pts).max()) < 1e-12,
          "%.3e m" % float(np.abs(back - pts).max()))

    lhs = (a @ b).apply(pts)
    rhs = a.apply(b.apply(pts))
    check("(A @ B) p == A (B p)", float(np.abs(lhs - rhs).max()) < 1e-12,
          "%.3e m" % float(np.abs(lhs - rhs).max()))

    q = a.quaternion_xyzw()
    d_rot, d_t = a.delta_to(Transform.from_quaternion(q, a.t))
    check("quaternion round-trip", d_rot < 1e-9, "%.3e deg" % d_rot)
    # 180 deg about each axis: the branch Shepperd's method exists for,
    # and the flips this part's near-symmetry keeps producing.
    worst = 0.0
    for axis in range(3):
        diag = -np.ones(3)
        diag[axis] = 1.0                      # pi about that axis
        pi_flip = Transform.from_Rt(np.diag(diag), [0.0, 0.0, 0.0])
        rt = Transform.from_quaternion(pi_flip.quaternion_xyzw(), pi_flip.t)
        worst = max(worst, pi_flip.delta_to(rt)[0])
        worst = max(worst, abs(pi_flip.rotation_angle_deg() - 180.0))
    check("180 deg flips: quaternion + angle", worst < 1e-6,
          "%.3e deg" % worst)

    to_from = Transform.from_list(a.to_list())
    check("JSON list round-trip", a.delta_to(to_from)[0] < 1e-12)

    R = frame_from_axes([0, -1, 0], [1, 0, 0])
    check("frame_from_axes is a rotation with z = approach",
          abs(np.linalg.det(R) - 1) < 1e-12
          and np.allclose(R[:, 2], [0, -1, 0]))
    R2 = frame_from_axes([0, -1, 0], [1, 0.02, 0])   # x not perpendicular
    check("frame_from_axes orthogonalises a sloppy x",
          float(np.abs(R2.T @ R2 - np.eye(3)).max()) < 1e-12)

    # Named composition: the whole point of the module.
    T_bc = framed("base", "camera", random_transform(1.0))
    T_co = framed("camera", "object", random_transform(0.7))
    T_bo = T_bc @ T_co
    check("T_base_camera @ T_camera_object -> T_base_object",
          T_bo.name == "T_base_object")
    p_obj = rng.normal(size=(50, 3)) * 0.02
    check("named composition agrees with manual product",
          float(np.abs(T_bo.apply(p_obj)
                       - T_bc.apply(T_co.apply(p_obj))).max()) < 1e-12)
    try:
        _ = T_co @ T_bc
        check("mismatched frames raise", False, "no exception")
    except FrameError as exc:
        check("mismatched frames raise", "does not meet" in str(exc))
    check("inverse swaps the frame names",
          T_bc.inverse().name == "T_camera_base")

    # Bad input must fail loudly.
    for name, bad in (("non-orthonormal rotation",
                       lambda: Transform.from_Rt(np.eye(3) * 1.01, [0, 0, 0])),
                      ("transposed-looking (reflection)",
                       lambda: Transform.from_Rt(np.diag([1.0, 1.0, -1.0]),
                                                 [0, 0, 0])),
                      ("millimetres fed as metres",
                       lambda: Transform.from_Rt(np.eye(3), [700.0, 0, 0])),
                      ("wrong shape",
                       lambda: Transform(np.eye(3)))):
        try:
            bad()
            check("rejects %s" % name, False, "no exception")
        except FrameError:
            check("rejects %s" % name, True)

    print("  %d failure(s)" % failures)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _self_check() else 0)
