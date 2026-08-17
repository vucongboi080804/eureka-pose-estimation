"""One RGB-D frame, and the wire form the rest of the cell speaks.

A camera and an estimator in different processes agree on exactly this:
two PNG images, a 3x3 intrinsic matrix, and the scale that turns depth
ticks into metres. PNG because depth has to cross the socket as the exact
integer ticks the sensor reported -- a lossy depth map would move poses by
more than the accuracy budget allows -- and base64 inside JSON because a
PLC, a ROS node or a shell script must be able to take a frame apart with
no codec and no NumPy.

The four estimator fields (``rgb_png_b64``, ``depth_png_b64``, ``K``,
``depth_scale``) carry the names deploy/pose_service/schema.py already
accepts, so a cell forwards a frame to the estimator without rewriting or
re-encoding it (:func:`estimate_body`). The camera's own fields --
``frame_id``, ``timestamp_ns``, ``source`` -- ride alongside, because
"which frame produced this pose" is the first question asked after a
cell drops a part.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

#: PNG compression level for both images. 1, not the default 3: the link
#: is loopback or a LAN, where the tens of milliseconds a higher level
#: spends squeezing a frame cost more than the extra bytes do. Every
#: level is lossless, which is the property depth actually depends on.
PNG_COMPRESSION = 1

#: The fields a frame body must carry for the pose service to estimate
#: from it, in the names deploy/pose_service/schema.py parses.
ESTIMATE_FIELDS = ("rgb_png_b64", "depth_png_b64", "K", "depth_scale")


@dataclass(frozen=True)
class Frame:
    """One registered RGB-D capture, exactly as the sensor reported it.

    Frozen because a frame is a record of something that happened: the
    realism knobs and any rectification act on the arrays *before* one is
    constructed, so that whatever a caller receives is what was
    transmitted. Validation happens here rather than at the socket, so a
    frame that cannot be estimated from is refused where it was made.
    """

    #: Monotonic within one source, starting at 1. A gap means frames
    #: were dropped or skipped, which is why it is served and logged.
    frame_id: int
    #: Host clock at capture (``time.time_ns``). The host clock, not the
    #: device's, because only the host's epoch is comparable with the
    #: pose service's logs and the robot's.
    timestamp_ns: int
    rgb: np.ndarray = field(repr=False)          # (H, W, 3) BGR uint8
    depth_raw: np.ndarray = field(repr=False)    # (H, W) uint16 ticks
    K: np.ndarray = field(repr=False)            # (3, 3) colour intrinsics
    depth_scale: float                           # metres per tick
    #: Where the pixels came from, e.g. ``scene_folder:test/000001`` or
    #: ``realsense:923322071127``.
    source_name: str = "camera"

    def __post_init__(self) -> None:
        rgb = np.ascontiguousarray(self.rgb)
        depth = np.ascontiguousarray(self.depth_raw)
        intrinsics = np.asarray(self.K, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be (H, W, 3) uint8 BGR, got %r %s"
                             % (rgb.shape, rgb.dtype))
        if depth.ndim != 2 or depth.dtype != np.uint16:
            # uint16 is not fussiness: it is what every RGB-D SDK delivers
            # and the only depth PNG that round-trips a tick exactly.
            raise ValueError("depth_raw must be (H, W) uint16 ticks, got "
                             "%r %s" % (depth.shape, depth.dtype))
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth %r is not registered to rgb %r"
                             % (depth.shape, rgb.shape[:2]))
        if intrinsics.shape != (3, 3):
            raise ValueError("K must be 3x3, got %r" % (intrinsics.shape,))
        if not float(self.depth_scale) > 0:
            raise ValueError("depth_scale must be positive, got %r"
                             % (self.depth_scale,))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_raw", depth)
        object.__setattr__(self, "K", intrinsics)
        object.__setattr__(self, "depth_scale", float(self.depth_scale))

    # -- shape -----------------------------------------------------------

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    def summary(self) -> str:
        """One line for a log or a console."""
        return ("frame %d  %dx%d  %s  depth_scale %g  %.0f%% of depth valid"
                % (self.frame_id, self.width, self.height, self.source_name,
                   self.depth_scale, 100.0 * self.valid_depth_fraction()))

    def valid_depth_fraction(self) -> float:
        """Share of pixels carrying a measurement (0 means no return).

        The number that collapses when a bin is full of shiny parts, so
        it is worth having on the camera side of the socket too.
        """
        return float(np.count_nonzero(self.depth_raw)) / self.depth_raw.size

    # -- wire ------------------------------------------------------------

    def to_wire(self) -> Dict[str, Any]:
        """The JSON body ``GET /v1/frame`` serves."""
        return {"frame_id": self.frame_id,
                "timestamp_ns": self.timestamp_ns,
                "rgb_png_b64": encode_png_b64(self.rgb),
                "depth_png_b64": encode_png_b64(self.depth_raw),
                "K": [[float(v) for v in row] for row in self.K],
                "depth_scale": self.depth_scale,
                "source": self.source_name}

    @classmethod
    def from_wire(cls, payload: Dict[str, Any]) -> "Frame":
        """Rebuild a frame from that body, for a client that wants pixels."""
        if not isinstance(payload, dict):
            raise ValueError("frame body must be a JSON object")
        missing = [k for k in ESTIMATE_FIELDS if payload.get(k) is None]
        if missing:
            raise ValueError("frame body is missing %s" % ", ".join(missing))
        return cls(frame_id=int(payload.get("frame_id", 0)),
                   timestamp_ns=int(payload.get("timestamp_ns", 0)),
                   rgb=decode_png_b64(payload["rgb_png_b64"],
                                      cv2.IMREAD_COLOR, "rgb_png_b64"),
                   depth_raw=decode_png_b64(payload["depth_png_b64"],
                                            cv2.IMREAD_UNCHANGED,
                                            "depth_png_b64"),
                   K=payload["K"], depth_scale=float(payload["depth_scale"]),
                   source_name=str(payload.get("source", "camera")))

    def to_scene(self, scene_id: Optional[str] = None) -> Any:
        """Wrap this frame as the pipeline's :class:`src.scene_io.Scene`.

        Imported inside the function on purpose: this module runs on the
        camera side of the cell, which needs OpenCV and NumPy but not the
        estimator's Open3D/torch stack, and must stay importable there.

        The conversion is the one ``deploy/live_adapter.scene_from_arrays``
        performs -- ticks times scale, in metres, 0 = no measurement --
        expressed here so that a camera process never has to import the
        pipeline to answer "what would the estimator see".
        """
        from src.scene_io import Scene
        return Scene(scene_id=scene_id or self.source_name, rgb=self.rgb,
                     depth=self.depth_raw.astype(np.float64) *
                     self.depth_scale, K=self.K)


def estimate_body(wire: Dict[str, Any],
                  scene_id: Optional[str] = None) -> Dict[str, Any]:
    """Rewrite a ``/v1/frame`` body as a ``POST /v1/estimate`` body.

    A pass-through: the base64 strings are moved, never decoded and
    re-encoded, so nothing between the sensor and the pose can quietly
    resample a depth map. ``scene_id`` labels the frame in the pose
    service's logs and results; it defaults to the camera's own
    identification of the frame.
    """
    missing = [k for k in ESTIMATE_FIELDS if wire.get(k) is None]
    if missing:
        raise ValueError("frame body is missing %s" % ", ".join(missing))
    body = {key: wire[key] for key in ESTIMATE_FIELDS}
    body["scene_id"] = scene_id or "%s#%d" % (wire.get("source", "camera"),
                                              int(wire.get("frame_id", 0)))
    return body


def encode_png_b64(image: np.ndarray) -> str:
    """PNG-encode an 8-bit colour or 16-bit depth array, base64 for JSON."""
    ok, buffer = cv2.imencode(".png", image,
                              [int(cv2.IMWRITE_PNG_COMPRESSION),
                               PNG_COMPRESSION])
    if not ok:
        raise ValueError("cannot PNG-encode a %s array of %s"
                         % (image.shape, image.dtype))
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_png_b64(text: str, flags: int, name: str) -> np.ndarray:
    """Inverse of :func:`encode_png_b64`, with the caller's field named.

    ``flags`` is ``cv2.IMREAD_COLOR`` for the image and
    ``cv2.IMREAD_UNCHANGED`` for depth -- the colour flag would quietly
    turn 16-bit ticks into 8-bit grey.
    """
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError("%s is not base64: %s" % (name, exc))
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("%s is not a readable PNG (%d bytes)"
                         % (name, len(raw)))
    return image
