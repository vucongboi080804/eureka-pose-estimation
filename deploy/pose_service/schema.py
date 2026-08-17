"""The wire contract between the cell and the pose service.

Everything crossing the socket is defined here, in plain standard-library
types: a PLC, a ROS node or a shell script must be able to parse a
response without importing NumPy, so no array ever leaves as anything but
nested lists of floats. Conversion is explicit in both directions rather
than reflected off the dataclasses, because a serialisation this small is
cheaper to read than the machinery that would generate it, and because
the request side has to reject malformed input with a precise reason.

Response fields only ever get added, never repurposed; ``SCHEMA_VERSION``
is what a cell pins against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#: Bumped on any change to the field set below. Minor: fields added.
SCHEMA_VERSION = "1.0"

#: The cell's decision for this frame: grab the top pose, or rescan.
GATE_PICK = "pick"
GATE_RESCAN = "rescan"


class RequestError(ValueError):
    """A request the service cannot even parse (HTTP 400)."""


@dataclass(frozen=True)
class PoseEstimateDTO:
    """One pose, in the submission's conventions: ``T_camera_object``.

    ``score`` is what a cell gates on. It is the product of the two
    factors, kept alongside it because they fail differently: a low
    ``seg_confidence`` with a high ``depth_verification`` is a part the
    segmenter half-recognised but geometry confirmed (a new lighting
    condition), the other way round is a confident mask the depth map
    refused (a pose worth logging before the bin is disturbed).
    """

    R: List[List[float]]                        # 3x3, object -> camera
    t: List[float]                              # 3, metres
    score: float
    seg_confidence: Optional[float] = None
    depth_verification: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"R": [list(row) for row in self.R], "t": list(self.t),
                "score": self.score, "seg_confidence": self.seg_confidence,
                "depth_verification": self.depth_verification}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PoseEstimateDTO":
        rot = _matrix3x3(payload.get("R"), "R")
        trans = _vector3(payload.get("t"), "t")
        return cls(R=rot, t=trans, score=float(payload["score"]),
                   seg_confidence=_optional_float(payload.get("seg_confidence")),
                   depth_verification=_optional_float(
                       payload.get("depth_verification")))


@dataclass(frozen=True)
class FrameResult:
    """Everything the service knows about one frame.

    A failed frame is still a result: ``poses`` empty, ``error`` set and
    ``gate`` = rescan. A cell reacts to a bad frame exactly as it reacts
    to a bin it cannot pick from, which is the behaviour that keeps a
    line running.
    """

    scene_id: str
    poses: List[PoseEstimateDTO]
    #: Wall-clock split of this frame: decode, prepare, segment, register,
    #: total. See PoseService for how each is measured.
    timings_ms: Dict[str, float]
    #: Masks the segmenters proposed before registration -- the number
    #: that collapses first when the scene leaves the training domain.
    n_proposals: int
    gate: str
    config_digest: str
    service_version: str
    schema_version: str = SCHEMA_VERSION
    error: Optional[str] = None

    @property
    def best(self) -> Optional[PoseEstimateDTO]:
        """The pose a cell would grab, highest score first."""
        return self.poses[0] if self.poses else None

    def to_dict(self) -> Dict[str, Any]:
        return {"scene_id": self.scene_id,
                "poses": [p.to_dict() for p in self.poses],
                "timings_ms": dict(self.timings_ms),
                "n_proposals": self.n_proposals,
                "gate": self.gate,
                "config_digest": self.config_digest,
                "service_version": self.service_version,
                "schema_version": self.schema_version,
                "error": self.error}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrameResult":
        return cls(scene_id=str(payload["scene_id"]),
                   poses=[PoseEstimateDTO.from_dict(p)
                          for p in payload.get("poses", [])],
                   timings_ms={str(k): float(v) for k, v
                               in payload.get("timings_ms", {}).items()},
                   n_proposals=int(payload.get("n_proposals", 0)),
                   gate=str(payload.get("gate", GATE_RESCAN)),
                   config_digest=str(payload.get("config_digest", "")),
                   service_version=str(payload.get("service_version", "")),
                   schema_version=str(payload.get("schema_version", "")),
                   error=payload.get("error"))


@dataclass(frozen=True)
class EstimateRequest:
    """One frame to estimate, given either by path or by value.

    ``scene_dir`` is for a service that shares a filesystem with whatever
    writes the frames (the offline runner, a bench rig, this repository's
    own scenes). The inline form is for a camera on the other side of a
    socket: PNG bytes, base64 in the JSON body, which keeps depth exactly
    as integer ticks -- a lossy transport of the depth map would move
    poses by more than the accuracy budget allows.
    """

    scene_dir: Optional[str] = None
    rgb_png_b64: Optional[str] = None
    depth_png_b64: Optional[str] = None
    K: Optional[List[List[float]]] = None
    depth_scale: Optional[float] = None
    scene_id: Optional[str] = None

    @property
    def is_inline(self) -> bool:
        return self.scene_dir is None

    def to_dict(self) -> Dict[str, Any]:
        payload = {}     # type: Dict[str, Any]
        for key in ("scene_dir", "rgb_png_b64", "depth_png_b64",
                    "depth_scale", "scene_id"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.K is not None:
            payload["K"] = [list(row) for row in self.K]
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "EstimateRequest":
        """Parse and reject: every failure here is a client bug (HTTP 400)."""
        if not isinstance(payload, dict):
            raise RequestError("body must be a JSON object")
        scene_dir = payload.get("scene_dir")
        inline_keys = ("rgb_png_b64", "depth_png_b64", "K", "depth_scale")
        given_inline = [k for k in inline_keys if payload.get(k) is not None]
        if scene_dir is not None and given_inline:
            raise RequestError("give either scene_dir or an inline frame, "
                               "not both (%s)" % ", ".join(given_inline))
        if scene_dir is not None:
            if not isinstance(scene_dir, str) or not scene_dir:
                raise RequestError("scene_dir must be a non-empty string")
            return cls(scene_dir=scene_dir,
                       scene_id=_optional_str(payload.get("scene_id")))
        missing = [k for k in inline_keys if payload.get(k) is None]
        if missing:
            raise RequestError("inline frame needs %s" % ", ".join(missing))
        depth_scale = float(payload["depth_scale"])
        if not depth_scale > 0:
            raise RequestError("depth_scale must be positive, got %r"
                               % (depth_scale,))
        for key in ("rgb_png_b64", "depth_png_b64"):
            if not isinstance(payload[key], str):
                raise RequestError("%s must be a base64 string" % key)
        return cls(rgb_png_b64=payload["rgb_png_b64"],
                   depth_png_b64=payload["depth_png_b64"],
                   K=_matrix3x3(payload["K"], "K"), depth_scale=depth_scale,
                   scene_id=_optional_str(payload.get("scene_id")))


def _matrix3x3(raw: Any, name: str) -> List[List[float]]:
    try:
        rows = [[float(v) for v in row] for row in raw]
    except (TypeError, ValueError):
        raise RequestError("%s must be a 3x3 array of numbers" % name)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise RequestError("%s must be 3x3, got %s"
                           % (name, [len(r) for r in rows]))
    return rows


def _vector3(raw: Any, name: str) -> List[float]:
    try:
        vector = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise RequestError("%s must be an array of 3 numbers" % name)
    if len(vector) != 3:
        raise RequestError("%s must have 3 entries, got %d"
                           % (name, len(vector)))
    return vector


def _optional_float(raw: Any) -> Optional[float]:
    return None if raw is None else float(raw)


def _optional_str(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise RequestError("scene_id must be a non-empty string")
    return raw
