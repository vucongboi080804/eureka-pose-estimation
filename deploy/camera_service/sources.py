"""Where frames come from: recorded scenes, or a RealSense on the cell.

Both implementations answer the same four calls -- ``open``, ``read``,
``close``, ``describe`` -- so the server above them never learns which
one it is driving. That is the point of the seam: the cell that is
demonstrated, benchmarked and regression-tested against a folder of
recorded scenes is byte-for-byte the cell that runs against a sensor,
because the only thing that changes is which class was constructed.

Two failures are told apart, because a cell reacts to them differently.
:class:`FrameSourceError` means the source cannot produce frames at all
(no SDK, no device, an empty split) and belongs on the launch line.
:class:`FrameReadError` means *this* frame did not arrive -- a half
written capture, a dropped USB packet -- and is something to count and
skip, never something to fall over.
"""

from __future__ import annotations

import abc
import json
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frame import Frame

#: How long :meth:`RealSenseSource.read` waits for a frameset before it
#: calls the frame lost. Five sensor periods at 30 fps: long enough to
#: ride out a USB hiccup, short enough that a cell's watchdog is not the
#: first thing to notice an unplugged camera.
REALSENSE_TIMEOUT_MS = 5000

#: Depth ticks are unsigned 16-bit; simulated noise is clipped into that
#: range rather than allowed to wrap a far surface around to zero.
MAX_DEPTH_TICK = 65535


class FrameSourceError(RuntimeError):
    """The source cannot produce frames at all -- fail on the launch line."""


class FrameReadError(RuntimeError):
    """One frame did not arrive. Count it, skip it, keep serving."""


class SourceExhausted(RuntimeError):
    """A finite stream reached its end (``loop`` off). Not an error."""


class FrameSource(abc.ABC):
    """A stream of registered RGB-D frames with an explicit lifecycle.

    Usable as a context manager::

        with SceneFolderSource(root, "test") as source:
            frame = source.read()

    Implementations are not thread-safe: one device, one reader. The
    server serialises reads with a lock rather than each source growing
    its own.
    """

    #: Set by :meth:`open`, cleared by :meth:`close`.
    is_open = False

    @property
    @abc.abstractmethod
    def is_live(self) -> bool:
        """True when frames age: a sensor's frame is worthless a second
        later, a replay's is not. The server uses this to decide whether
        a frame it read at startup may still be handed to the first
        caller."""

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the device or the folder. Raises FrameSourceError."""

    @abc.abstractmethod
    def read(self) -> Frame:
        """The next frame. Raises FrameReadError or SourceExhausted."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release everything. Safe to call twice, and on a failed open."""

    @abc.abstractmethod
    def describe(self) -> Dict[str, Any]:
        """What this source is and how far through it is, for /healthz."""

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class SceneFolderSource(FrameSource):
    """Replay a release folder as a camera stream.

    ``root/split/<scene>/{rgb.png,depth.png,camera.json}`` is the layout
    every scene in this repository already has, so the frames a reviewer
    can check against ground truth are the frames the cell sees. This is
    the source that runs here and on the board when no sensor is
    attached, and the one every end-to-end test uses.

    The realism knobs are a simulation aid, not a sensor model. They make
    a recording *harder* than it was -- fewer depth returns, noisier
    ticks, a less punctual link -- so that a bench run exercises the
    paths a cell takes when a bin is full of shiny parts. Nothing here is
    calibrated against a real sensor, and none of it is on by default.
    """

    def __init__(self, root: str, split: str = "test",
                 scenes: Optional[Sequence[str]] = None, loop: bool = True,
                 fps: Optional[float] = None, depth_dropout: float = 0.0,
                 depth_noise_mm: float = 0.0, jitter_ms: float = 0.0,
                 seed: int = 0):
        """Args:
            root: Release folder holding the split directories.
            split: ``test``, ``train``, or whatever the release ships.
            scenes: Ids to replay, in this order; None replays the whole
                split, sorted, which is the order a reviewer expects.
            loop: Wrap at the end instead of raising SourceExhausted.
            fps: Pace reads to this rate; None or 0 delivers as fast as
                the caller asks, which is what a cell taking one frame
                per pick wants.
            depth_dropout: Fraction of depth pixels zeroed.
            depth_noise_mm: Standard deviation of the noise added to
                measured depth, in millimetres, before requantisation.
            jitter_ms: Uniform 0..N ms of extra latency per read.
            seed: Makes the two random knobs reproducible.
        """
        self.root = os.path.abspath(root)
        self.split = split
        self.loop = bool(loop)
        self.fps = float(fps or 0.0)
        self.depth_dropout = float(depth_dropout)
        self.depth_noise_mm = float(depth_noise_mm)
        self.jitter_ms = float(jitter_ms)
        self.seed = int(seed)
        self._requested = tuple(scenes) if scenes else ()
        self._scenes = ()           # type: Tuple[str, ...]
        self._cursor = 0
        self._laps = 0
        self._frame_id = 0
        self._next_due = 0.0
        # RandomState rather than the newer Generator: it is the API that
        # exists in every NumPy the board might ship, and reproducibility
        # is the only thing asked of it.
        self._rng = np.random.RandomState(self.seed)

    name = "scene_folder"

    @property
    def is_live(self) -> bool:
        return False

    def open(self) -> None:
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FrameSourceError("no such split directory: %s" % split_dir)
        if self._requested:
            missing = [s for s in self._requested
                       if not os.path.isdir(os.path.join(split_dir, s))]
            if missing:
                raise FrameSourceError("scenes not in %s: %s"
                                       % (split_dir, ", ".join(missing)))
            self._scenes = self._requested
        else:
            self._scenes = tuple(sorted(
                name for name in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, name))))
        if not self._scenes:
            raise FrameSourceError("no scene folders in %s" % split_dir)
        self._cursor = 0
        self._next_due = time.monotonic()
        self.is_open = True

    def read(self) -> Frame:
        if not self.is_open:
            raise FrameSourceError("read() before open()")
        self._pace()
        scene_id = self._next_scene()
        # The id is spent before the load, so a scene that fails to read
        # leaves a gap in the sequence rather than disappearing silently.
        self._frame_id += 1
        rgb, depth, camera = self._load(scene_id)
        depth = self._degrade(depth)
        try:
            return Frame(frame_id=self._frame_id, timestamp_ns=time.time_ns(),
                         rgb=rgb, depth_raw=depth, K=camera["K"],
                         depth_scale=float(camera["depth_scale"]),
                         source_name="%s:%s/%s" % (self.name, self.split,
                                                   scene_id))
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameReadError("%s/%s: %s" % (self.split, scene_id, exc))

    def close(self) -> None:
        self.is_open = False

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.name, "live": False, "root": self.root,
                "split": self.split, "scenes": len(self._scenes),
                "cursor": self._cursor, "laps": self._laps,
                "loop": self.loop, "fps": self.fps,
                "next_scene": self._peek(),
                "realism": {"depth_dropout": self.depth_dropout,
                            "depth_noise_mm": self.depth_noise_mm,
                            "jitter_ms": self.jitter_ms, "seed": self.seed}}

    # -- internals -------------------------------------------------------

    def _peek(self) -> Optional[str]:
        """The scene the next read would serve, or None when there is none."""
        if not self._scenes:
            return None
        if self._cursor >= len(self._scenes):
            return self._scenes[0] if self.loop else None
        return self._scenes[self._cursor]

    def _next_scene(self) -> str:
        if self._cursor >= len(self._scenes):
            if not self.loop:
                raise SourceExhausted("replayed all %d scenes of %s"
                                      % (len(self._scenes),
                                         os.path.join(self.root, self.split)))
            self._cursor = 0
            self._laps += 1
        scene_id = self._scenes[self._cursor]
        self._cursor += 1
        return scene_id

    def _pace(self) -> None:
        """Hold the configured frame rate, then add the jitter knob."""
        if self.fps > 0:
            now = time.monotonic()
            if now < self._next_due:
                time.sleep(self._next_due - now)
            # From the due time, not from now: a slow read is absorbed by
            # the next period instead of shifting the whole stream.
            self._next_due = max(self._next_due + 1.0 / self.fps,
                                 time.monotonic())
        if self.jitter_ms > 0:
            time.sleep(self._rng.uniform(0.0, self.jitter_ms) / 1000.0)

    def _load(self, scene_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Read one scene folder, or say precisely what was wrong with it."""
        scene_dir = os.path.join(self.root, self.split, scene_id)
        rgb = cv2.imread(os.path.join(scene_dir, "rgb.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FrameReadError("cannot read %s/rgb.png" % scene_dir)
        # IMREAD_UNCHANGED, or OpenCV would hand back 8-bit grey and the
        # whole depth map would be wrong by three orders of magnitude.
        depth = cv2.imread(os.path.join(scene_dir, "depth.png"),
                           cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FrameReadError("cannot read %s/depth.png" % scene_dir)
        if depth.dtype != np.uint16:
            raise FrameReadError("%s/depth.png is %s, expected 16-bit"
                                 % (scene_dir, depth.dtype))
        try:
            with open(os.path.join(scene_dir, "camera.json")) as handle:
                camera = json.load(handle)
        except (OSError, ValueError) as exc:
            raise FrameReadError("cannot read %s/camera.json: %s"
                                 % (scene_dir, exc))
        return rgb, depth, camera

    def _degrade(self, depth: np.ndarray) -> np.ndarray:
        """Apply the depth knobs, in the order a sensor would impose them."""
        if not (self.depth_noise_mm > 0 or self.depth_dropout > 0):
            return depth
        measured = depth > 0
        noisy = depth.astype(np.float64)
        if self.depth_noise_mm > 0:
            # Only where something was measured: adding noise to a hole
            # would invent a surface the sensor never saw.
            noise = self._rng.normal(0.0, self.depth_noise_mm, depth.shape)
            noisy = np.where(measured, noisy + noise, 0.0)
        if self.depth_dropout > 0:
            keep = self._rng.random_sample(depth.shape) >= self.depth_dropout
            noisy = np.where(keep, noisy, 0.0)
        # Rounded back to integer ticks: the transport and the sensor
        # both quantise, so a bench frame must be quantised too.
        return np.clip(np.rint(noisy), 0, MAX_DEPTH_TICK).astype(np.uint16)


class RealSenseSource(FrameSource):
    """Frames from an Intel RealSense through pyrealsense2.

    UNTESTED: no RealSense was attached to any machine this repository was
    written on, so this path is written to be short and obviously correct
    rather than tuned. What it does is the documented minimum for a
    depth camera feeding this pipeline, and each step matters:

    * depth is aligned to the colour stream (``rs.align``), because the
      pipeline requires depth registered pixel-for-pixel to the image;
    * ``K`` comes from the *colour* profile after alignment, since that
      is the frame the aligned depth now lives in;
    * ``depth_scale`` is read off the device rather than assumed, because
      it differs between models (D400 ships 0.001, others do not).

    The SDK is imported inside :meth:`open`, so this module -- and the
    server that chooses between sources -- stays importable on a machine
    that has never seen librealsense.
    """

    def __init__(self, serial: Optional[str] = None, width: int = 1280,
                 height: int = 720, fps: int = 30):
        """Args:
            serial: Device serial to open; None takes the first found.
            width, height: Colour resolution; depth is aligned onto it.
            fps: Stream rate requested from both sensors.
        """
        self.serial = serial or None
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._pipeline = None       # type: Any
        self._align = None          # type: Any
        self._K = None              # type: Optional[np.ndarray]
        self._depth_scale = 0.0
        self._frame_id = 0

    name = "realsense"

    @property
    def is_live(self) -> bool:
        return True

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise FrameSourceError(
                "the realsense source needs the pyrealsense2 package "
                "(pip install pyrealsense2, or the librealsense build for "
                "this JetPack): %s" % exc)
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        # bgr8 is OpenCV's channel order, which is the pipeline's too, so
        # no frame is ever converted between the sensor and the estimator.
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            raise FrameSourceError("cannot start the RealSense pipeline "
                                   "(serial %s, %dx%d@%d): %s"
                                   % (self.serial or "first", self.width,
                                      self.height, self.fps, exc))
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        colour = profile.get_stream(rs.stream.color)
        intrinsics = colour.as_video_stream_profile().get_intrinsics()
        self._K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                            [0.0, intrinsics.fy, intrinsics.ppy],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self._depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale())
        if not self._depth_scale > 0:
            raise FrameSourceError("device reported depth_scale %r"
                                   % (self._depth_scale,))
        self.is_open = True

    def read(self) -> Frame:
        if not self.is_open:
            raise FrameSourceError("read() before open()")
        try:
            frames = self._pipeline.wait_for_frames(REALSENSE_TIMEOUT_MS)
        except RuntimeError as exc:
            raise FrameReadError("no frameset within %d ms: %s"
                                 % (REALSENSE_TIMEOUT_MS, exc))
        aligned = self._align.process(frames)
        colour_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not colour_frame or not depth_frame:
            raise FrameReadError("frameset arrived without both streams")
        self._frame_id += 1
        # np.asanyarray on the SDK buffer is a view into memory the SDK
        # will reuse; copy before the frame outlives this call.
        rgb = np.array(np.asanyarray(colour_frame.get_data()), copy=True)
        depth = np.array(np.asanyarray(depth_frame.get_data()), copy=True)
        try:
            # The host clock, not the device's: only the host's epoch is
            # comparable with the pose service's and the robot's logs.
            return Frame(frame_id=self._frame_id, timestamp_ns=time.time_ns(),
                         rgb=rgb, depth_raw=depth, K=self._K,
                         depth_scale=self._depth_scale,
                         source_name="%s:%s" % (self.name,
                                                self.serial or "first"))
        except ValueError as exc:
            raise FrameReadError("unusable frameset: %s" % exc)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                # Already stopped, or the device went away: closing must
                # not be the thing that stops a shutdown from finishing.
                pass
            self._pipeline = None
        self.is_open = False

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.name, "live": True,
                "serial": self.serial or "first",
                "width": self.width, "height": self.height, "fps": self.fps,
                "depth_scale": self._depth_scale}


def source_from_config(config: Any) -> FrameSource:
    """Build the source a :class:`~.config.CameraConfig` asks for.

    The one place the kind string turns into a class, so the server never
    branches on it.
    """
    if config.source == "scene_folder":
        return SceneFolderSource(root=config.root, split=config.split,
                                 scenes=config.scene_ids() or None,
                                 loop=config.loop, fps=config.fps,
                                 depth_dropout=config.depth_dropout,
                                 depth_noise_mm=config.depth_noise_mm,
                                 jitter_ms=config.jitter_ms, seed=config.seed)
    if config.source == "realsense":
        return RealSenseSource(serial=config.rs_serial, width=config.rs_width,
                               height=config.rs_height, fps=config.rs_fps)
    raise FrameSourceError("unknown source kind: %r" % (config.source,))
