"""A recorded session: the format on disk, the reader, and the replay source.

A cell customer records what the camera saw and replays it later -- to
reproduce a fault with the frames that caused it, to re-run a model
change against real captures, to show the cell working with no hardware
in the room. That only means anything if the replay is the recording:
the estimator segments the colour image and registers against the depth
map, so a stream that quietly requantises either one returns poses that
were never measured.

The bar a replay has to clear is not "the poses agree", because the
pipeline does not agree with itself. Estimating this repository's
``test`` split twice from the same folder moved one instance of scene
000048 by 7.0 mm and 177 deg -- a symmetry rival winning the second time
-- and changed the instance count on 000027 (1 to 2) and on 000046 (1 to
3), with nothing but the seeds between the two runs. Poses measured
against that prove nothing about a replay. Bytes do: the PNG the camera
puts on the wire from a replayed frame is identical to the PNG it puts
on the wire from the release folder, on all 40 scenes, so the estimator
cannot tell the two apart at all.

A lossy stream fails that test by inspection. Passing the colour frames
through OpenCV's MPEG-4 default moved every pixel (mean 3.6 to 4.1 of
255 per channel, worst 107) and took scene 000048 from six instances to
four. So both streams a replay reads are lossless, and a lossy copy
exists only for a human:

    session.json   per-frame intrinsics, depth scale, scene ids,
                   timestamps and checksums, plus a digest of itself
    color.mkv      FFV1, the colour frames exactly as recorded
    depth.mkv      FFV1, the 16-bit depth ticks exactly as recorded
    preview.mp4    the colour frames, lossy and small, to watch

A video stream is one frame size and this release is not: 38 of the 40
``test`` scenes are 960x640 and two are 920x728. Rather than split a
session per geometry, every frame is placed at the origin of a canvas
big enough for all of them, its true size is written into the sidecar,
and the reader crops back to it before anything sees the pixels. The
padding is zeros, which is "no measurement" in a depth map and black in
a picture; the checksums are taken on the true frame, so the crop has to
land exactly or the frame is refused.

FFV1 rather than a 16-bit PNG tar because it is lossless, it is in every
FFMPEG the deployment has (the board's own container, OpenCV 4.10 on
aarch64, decodes what OpenCV 5.0 writes here bit for bit, on both frame
sizes and after a backward seek), and a single container seeks. Depth is
carried as two 8-bit planes rather than ``gray16le`` because
``cv2.VideoWriter`` is an 8-bit picture interface: handed a 16-bit array
it either refuses the frame and leaves a file nothing can decode (what
OpenCV 5.0 does here) or keeps the low byte, which on this data would
bring 748 ticks back as 236 and lose half a metre without raising
anything. Splitting the ticks ourselves is the version of that which
cannot fail silently, and the per-frame checksums are what prove it did
not.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import CAMERA_VERSION
from .frame import Frame
from .sources import (FrameReadError, FrameSource, FrameSourceError,
                      SourceExhausted)

#: Bumped when the layout below changes in a way an older reader would
#: misread. A reader refuses a session whose major version it does not
#: know rather than guessing at the difference.
SESSION_VERSION = "1.0"

SIDECAR_NAME = "session.json"
COLOR_NAME = "color.mkv"
DEPTH_NAME = "depth.mkv"
PREVIEW_NAME = "preview.mp4"

#: Everything a session directory is made of, sidecar first. A writer
#: opening over an existing session removes these before it writes: a
#: stream this recording does not produce -- the preview, when it is
#: switched off -- would otherwise survive and be listed in the new
#: sidecar with its own byte count and checksum, so it would verify, and
#: a customer would watch the previous recording without anything
#: looking wrong.
SESSION_FILES = (SIDECAR_NAME, COLOR_NAME, DEPTH_NAME, PREVIEW_NAME)

#: Lossless intra-frame codec for both real streams. Kept as the four
#: characters rather than a packed fourcc so importing this module costs
#: nothing on a build whose ``VideoWriter`` cannot encode -- a board
#: replays sessions, it does not record them.
LOSSLESS_FOURCC = "FFV1"

#: The watchable copy. MPEG-4 Part 2, because that is the encoder every
#: OpenCV ships; H.264 needs a libx264 that a GPL-free build does not
#: have, and a preview is not worth a codec the recorder might lack.
PREVIEW_FOURCC = "mp4v"

#: How depth is carried in an 8-bit picture: little-endian uint16 split
#: across the blue (low byte) and green (high byte) planes.
DEPTH_ENCODING = "u16_le_split_bg"

#: Playback rate written into both containers when the caller says
#: nothing. Two frames a second is roughly a cell's pick rate, so a
#: recording plays back at about the speed the cell ran at.
DEFAULT_FPS = 2.0

#: Frames worth stepping over one at a time before asking the decoder to
#: seek. Decoding is far cheaper than a seek for a neighbouring frame,
#: and every seek is a chance for an older build to land somewhere else.
SEQUENTIAL_SCAN = 8


class SessionError(RuntimeError):
    """A session that cannot be read as recorded."""


class SessionCorrupt(SessionError):
    """Bytes on disk disagree with the checksums written beside them."""


# -- pixels ---------------------------------------------------------------

def pack_depth(depth: np.ndarray) -> np.ndarray:
    """One depth map as the 8-bit BGR picture the codec stores."""
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("depth must be (H, W) uint16 ticks, got %r %s"
                         % (depth.shape, depth.dtype))
    packed = np.zeros(depth.shape + (3,), dtype=np.uint8)
    packed[:, :, 0] = (depth & 0xFF).astype(np.uint8)
    packed[:, :, 1] = (depth >> 8).astype(np.uint8)
    return packed


def unpack_depth(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_depth`, on what the decoder handed back."""
    if packed.ndim != 3 or packed.shape[2] < 2:
        raise ValueError("packed depth must be (H, W, >=2) uint8, got %r"
                         % (packed.shape,))
    return np.ascontiguousarray(
        packed[:, :, 0].astype(np.uint16) |
        (packed[:, :, 1].astype(np.uint16) << 8))


def digest_rgb(rgb: np.ndarray) -> str:
    """Checksum of a colour frame, over the bytes themselves."""
    return hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def digest_depth(depth: np.ndarray) -> str:
    """Checksum of a depth frame, in an explicit byte order.

    ``<u2`` rather than the native dtype so a session recorded on one
    architecture verifies on another; both machines here are
    little-endian, and a checksum that only holds while that is true is
    not a checksum.
    """
    return hashlib.sha1(
        np.ascontiguousarray(depth).astype("<u2", copy=False)
        .tobytes()).hexdigest()


def digest_file(path: str) -> str:
    """Checksum of a whole stream file, read in chunks."""
    sha = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _sidecar_digest(body: Dict[str, Any]) -> str:
    """Checksum of everything in the sidecar that changes a replay.

    Catches the edit nobody would notice: a ``depth_scale`` retyped in a
    text editor moves every pose in the session by a factor, and the
    pixels would still check out.
    """
    return hashlib.sha1(json.dumps(body, sort_keys=True,
                                   separators=(",", ":"))
                        .encode("utf-8")).hexdigest()


class SessionFrame(NamedTuple):
    """One recorded capture.

    The first four fields are exactly what a camera reports, in the order
    :class:`~.frame.Frame` takes them; the last three say which recorded
    capture they came from.
    """

    rgb: np.ndarray
    depth_raw: np.ndarray
    K: np.ndarray
    depth_scale: float
    scene_id: str
    timestamp_ns: int
    index: int


# -- writing --------------------------------------------------------------

class SessionWriter:
    """Build a session directory, one frame at a time.

    Lifecycle is explicit because three encoders and a sidecar have to be
    finished together: nothing on disk is a readable session until
    :meth:`close` has written ``session.json``, which is deliberate --- a
    recording interrupted halfway leaves a directory that a reader
    refuses rather than one it half-believes.
    """

    def __init__(self, path: str, fps: float = DEFAULT_FPS,
                 preview: bool = True,
                 canvas: Optional[Tuple[int, int]] = None):
        """Args:
            path: Directory to create and fill.
            fps: Playback rate stamped into both containers.
            preview: Also write the lossy ``preview.mp4``. Off when the
                recording is headed for a regression suite that will
                never look at it.
            canvas: ``(width, height)`` of the streams, which must hold
                the largest frame that will be appended. None takes the
                first frame's size, which is what a camera of one
                geometry wants; a caller recording a release that mixes
                sizes has to know the largest before the first frame is
                encoded, because a stream's size cannot change later.
        """
        self.path = os.path.abspath(path)
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        self.preview = bool(preview)
        self.canvas = ((int(canvas[0]), int(canvas[1])) if canvas else None)
        self.width = 0
        self.height = 0
        self._color = None              # type: Any
        self._depth = None              # type: Any
        self._preview = None            # type: Any
        self._records = []              # type: List[Dict[str, Any]]
        self._ids = set()               # type: set
        self._open = False

    def open(self) -> "SessionWriter":
        """Claim the directory, clearing any session already in it.

        The sidecar goes first, which is what makes an interrupted
        overwrite safe: from that moment until :meth:`close` there is no
        session here for a reader to half-believe.
        """
        os.makedirs(self.path, exist_ok=True)
        for name in SESSION_FILES:
            stale = os.path.join(self.path, name)
            if os.path.isfile(stale):
                os.remove(stale)
        self._open = True
        return self

    def append(self, rgb: np.ndarray, depth_raw: np.ndarray, K: Any,
               depth_scale: float, scene_id: str,
               timestamp_ns: Optional[int] = None) -> int:
        """Add one capture. Returns its index in the session."""
        if not self._open:
            raise SessionError("append() before open()")
        rgb = np.ascontiguousarray(rgb)
        depth_raw = np.ascontiguousarray(depth_raw)
        # The same validation Frame applies, applied before anything is
        # encoded: a session must not be able to hold a frame the cell
        # would refuse to serve.
        Frame(frame_id=len(self._records) + 1, timestamp_ns=0, rgb=rgb,
              depth_raw=depth_raw, K=K, depth_scale=depth_scale)
        if str(scene_id) in self._ids:
            # A scene id is how a replay asks for a frame, so a repeated
            # one makes every later copy unreachable: the reader would
            # hand out the first match and nothing would report the
            # frames nobody can get to.
            raise SessionError("scene id %r is already in this session; a "
                               "session addresses frames by id, so they "
                               "have to be unique" % (str(scene_id),))
        if not self._records:
            self._start(*(self.canvas or (rgb.shape[1], rgb.shape[0])))
        if rgb.shape[1] > self.width or rgb.shape[0] > self.height:
            # A stream's frame size is fixed once the encoder is open, and
            # an encoder handed an oversized frame would rescale it into a
            # different camera without saying so.
            raise SessionError("%s is %dx%d, larger than the session canvas "
                               "%dx%d" % (scene_id, rgb.shape[1],
                                          rgb.shape[0], self.width,
                                          self.height))
        colour = self._fit(rgb)
        self._color.write(colour)
        self._depth.write(self._fit(pack_depth(depth_raw)))
        if self._preview is not None:
            self._preview.write(colour)
        self._records.append({
            "index": len(self._records), "scene_id": str(scene_id),
            "timestamp_ns": int(timestamp_ns if timestamp_ns is not None
                                else time.time_ns()),
            "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
            "depth_scale": float(depth_scale),
            "K": [[float(v) for v in row]
                  for row in np.asarray(K, dtype=np.float64).reshape(3, 3)],
            "rgb_sha1": digest_rgb(rgb),
            "depth_sha1": digest_depth(depth_raw)})
        self._ids.add(str(scene_id))
        return len(self._records) - 1

    def close(self) -> Dict[str, Any]:
        """Finish the encoders, write the sidecar, return the manifest."""
        if not self._open:
            raise SessionError("close() before open()")
        for writer in (self._color, self._depth, self._preview):
            if writer is not None:
                writer.release()
        self._color = self._depth = self._preview = None
        self._open = False
        if not self._records:
            raise SessionError("no frames were recorded into %s" % self.path)
        streams = {"color": self._stream_entry(COLOR_NAME, LOSSLESS_FOURCC,
                                               True),
                   "depth": self._stream_entry(DEPTH_NAME, LOSSLESS_FOURCC,
                                               True)}
        streams["depth"]["encoding"] = DEPTH_ENCODING
        preview = self._stream_entry(PREVIEW_NAME, PREVIEW_FOURCC, False)
        if preview is not None:
            streams["preview"] = preview
        # width/height are the streams' canvas; each frame record carries
        # the size the camera actually reported for it.
        body = {"session_version": SESSION_VERSION, "width": self.width,
                "height": self.height, "fps": self.fps, "streams": streams,
                "frames": self._records}
        manifest = dict(body)
        manifest["digest"] = _sidecar_digest(body)
        manifest["recorder"] = "camera_service/%s" % CAMERA_VERSION
        manifest["created_ns"] = time.time_ns()
        with open(os.path.join(self.path, SIDECAR_NAME), "w") as handle:
            json.dump(manifest, handle, indent=1)
            handle.write("\n")
        return manifest

    def __enter__(self) -> "SessionWriter":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Only seal a session that was recorded to the end: an exception
        # on the way through should leave no sidecar, so the half-written
        # streams cannot be mistaken for a replayable session.
        if exc_type is None and self._open:
            self.close()

    # -- internals -------------------------------------------------------

    def _start(self, width: int, height: int) -> None:
        """Open the encoders, which need the frame size to exist."""
        self.width, self.height = int(width), int(height)
        self._color = self._writer(COLOR_NAME, LOSSLESS_FOURCC)
        self._depth = self._writer(DEPTH_NAME, LOSSLESS_FOURCC)
        if self.preview:
            # A missing MPEG-4 encoder loses the convenience copy, not
            # the recording: the two streams that matter are already open.
            self._preview = self._writer(PREVIEW_NAME, PREVIEW_FOURCC,
                                         required=False)

    def _fit(self, picture: np.ndarray) -> np.ndarray:
        """Place a frame at the origin of the canvas, zero-padded."""
        if picture.shape[1] == self.width and picture.shape[0] == self.height:
            return picture
        canvas = np.zeros((self.height, self.width, picture.shape[2]),
                          dtype=picture.dtype)
        canvas[:picture.shape[0], :picture.shape[1]] = picture
        return canvas

    def _writer(self, name: str, fourcc: str, required: bool = True) -> Any:
        path = os.path.join(self.path, name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                 self.fps, (self.width, self.height), True)
        if writer.isOpened():
            return writer
        writer.release()
        if required:
            raise SessionError("this OpenCV cannot encode %s into %s; a "
                               "session needs a lossless codec" % (fourcc,
                                                                   path))
        return None

    def _stream_entry(self, name: str, fourcc: str,
                      lossless: bool) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.path, name)
        if not os.path.isfile(path):
            return None
        return {"file": name, "codec": fourcc, "lossless": lossless,
                "bytes": os.path.getsize(path), "sha1": digest_file(path)}


# -- reading --------------------------------------------------------------

class SessionReader:
    """Read a recorded session back, checking it against its own record.

    Every frame handed out has had its colour and depth checksummed
    against the value written at record time, because the whole point of
    a session is that the pixels are the ones the cell saw. A decoder
    that lands on the wrong frame after a seek, a file truncated by a
    half-finished copy and a depth stream someone re-encoded all look
    identical to a caller until something checks.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.manifest = {}              # type: Dict[str, Any]
        self.records = ()               # type: Tuple[Dict[str, Any], ...]
        self.is_open = False
        self._color = None              # type: Any
        self._depth = None              # type: Any
        self._next = {}                 # type: Dict[str, int]

    def open(self) -> "SessionReader":
        """Load and check the sidecar, then open both streams."""
        self.manifest = self._load_manifest()
        self.records = tuple(self.manifest["frames"])
        self._color = self._capture("color")
        self._depth = self._capture("depth")
        self._next = {"color": 0, "depth": 0}
        self.is_open = True
        return self

    def close(self) -> None:
        """Release both decoders. Safe to call twice, and on a failed open."""
        for cap in (self._color, self._depth):
            if cap is not None:
                cap.release()
        self._color = self._depth = None
        self.is_open = False

    def __enter__(self) -> "SessionReader":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def frame_count(self) -> int:
        return len(self.records)

    def scene_ids(self) -> Tuple[str, ...]:
        """The recorded scene ids, in recording order."""
        return tuple(str(r["scene_id"]) for r in self.records)

    def index_of(self, scene_id: str) -> int:
        """Where a scene id sits in the session, or -1."""
        ids = self.scene_ids()
        return ids.index(scene_id) if scene_id in ids else -1

    def read(self, index: int) -> SessionFrame:
        """The frame at ``index``, verified. Random access is the point.

        Raises:
            SessionCorrupt: the pixels do not match their checksum, twice
                -- once where the decoder put us, and again after a scan
                from the start of the stream.
            SessionError: the reader is closed, or the index is not one.
        """
        if not self.is_open:
            raise SessionError("read() before open()")
        if not 0 <= index < len(self.records):
            raise SessionError("frame %d is not in a session of %d"
                               % (index, len(self.records)))
        record = self.records[index]
        try:
            return self._frame(record, index, rescan=False)
        except SessionCorrupt:
            # A seek that landed on a neighbouring frame and a genuinely
            # damaged file are the same symptom; only the retry tells
            # them apart, and it is cheap next to being wrong.
            self._rewind()
            return self._frame(record, index, rescan=True)

    def verify(self) -> Dict[str, Any]:
        """Re-hash the stream files against the sidecar.

        The check for a session that arrived over a network: it reads
        every byte, so it is not what happens on each frame, but it says
        in one call whether the containers are the ones recorded.
        """
        streams = {}                    # type: Dict[str, Any]
        for name, entry in sorted(self.manifest.get("streams", {}).items()):
            path = os.path.join(self.path, entry["file"])
            actual = digest_file(path)
            streams[name] = {"file": entry["file"],
                             "bytes": os.path.getsize(path), "sha1": actual,
                             "ok": actual == entry["sha1"]}
        return {"ok": all(v["ok"] for v in streams.values()),
                "streams": streams}

    def describe(self) -> Dict[str, Any]:
        """What this session is, for a log line or a health endpoint."""
        streams = self.manifest.get("streams", {})
        return {"path": self.path,
                "session_version": self.manifest.get("session_version"),
                "recorder": self.manifest.get("recorder"),
                "created_ns": self.manifest.get("created_ns"),
                "frames": len(self.records),
                "width": self.manifest.get("width"),
                "height": self.manifest.get("height"),
                "recorded_fps": self.manifest.get("fps"),
                "bytes": {k: v.get("bytes") for k, v in streams.items()}}

    # -- internals -------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        sidecar = os.path.join(self.path, SIDECAR_NAME)
        try:
            with open(sidecar) as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as exc:
            raise SessionError("cannot read %s: %s" % (sidecar, exc))
        if not isinstance(manifest, dict):
            raise SessionError("%s must be a JSON object" % sidecar)
        version = str(manifest.get("session_version", ""))
        if version.split(".")[0] != SESSION_VERSION.split(".")[0]:
            raise SessionError("%s is session format %r; this reader speaks "
                               "%s" % (sidecar, version, SESSION_VERSION))
        frames = manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise SessionError("%s records no frames" % sidecar)
        body = {"session_version": manifest.get("session_version"),
                "width": manifest.get("width"),
                "height": manifest.get("height"), "fps": manifest.get("fps"),
                "streams": manifest.get("streams"), "frames": frames}
        if _sidecar_digest(body) != manifest.get("digest"):
            raise SessionCorrupt("%s has been changed since it was recorded: "
                                 "its digest does not match its contents"
                                 % sidecar)
        for entry in ("color", "depth"):
            stream = manifest.get("streams", {}).get(entry)
            if not isinstance(stream, dict):
                raise SessionError("%s has no %s stream" % (sidecar, entry))
            path = os.path.join(self.path, stream["file"])
            if not os.path.isfile(path):
                raise SessionError("%s is missing" % path)
            # Length before contents: a copy cut short is the common
            # failure, and it is answered without hashing 30 MB.
            if os.path.getsize(path) != stream["bytes"]:
                raise SessionCorrupt("%s is %d bytes, %d were recorded"
                                     % (path, os.path.getsize(path),
                                        stream["bytes"]))
        return manifest

    def _capture(self, entry: str) -> Any:
        path = os.path.join(self.path,
                            self.manifest["streams"][entry]["file"])
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise SessionError("this OpenCV cannot decode %s; the session "
                               "needs an FFMPEG build with FFV1" % path)
        return cap

    def _rewind(self) -> None:
        for name, cap in (("color", self._color), ("depth", self._depth)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._next[name] = 0

    def _frame(self, record: Dict[str, Any], index: int,
               rescan: bool) -> SessionFrame:
        width, height = int(record["width"]), int(record["height"])
        rgb = np.ascontiguousarray(
            self._decode("color", index, rescan)[:height, :width])
        depth = unpack_depth(
            self._decode("depth", index, rescan)[:height, :width])
        actual_rgb, actual_depth = digest_rgb(rgb), digest_depth(depth)
        if actual_rgb != record["rgb_sha1"]:
            raise SessionCorrupt("frame %d (%s): colour is %s, %s was "
                                 "recorded" % (index, record["scene_id"],
                                               actual_rgb, record["rgb_sha1"]))
        if actual_depth != record["depth_sha1"]:
            raise SessionCorrupt("frame %d (%s): depth is %s, %s was "
                                 "recorded" % (index, record["scene_id"],
                                               actual_depth,
                                               record["depth_sha1"]))
        return SessionFrame(rgb=rgb, depth_raw=depth,
                            K=np.asarray(record["K"], dtype=np.float64),
                            depth_scale=float(record["depth_scale"]),
                            scene_id=str(record["scene_id"]),
                            timestamp_ns=int(record["timestamp_ns"]),
                            index=index)

    def _decode(self, entry: str, index: int, rescan: bool) -> np.ndarray:
        """Get the decoder to ``index`` and take the picture there."""
        cap = self._color if entry == "color" else self._depth
        cursor = self._next[entry]
        step = index - cursor
        if step < 0 or (step > SEQUENTIAL_SCAN and not rescan):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise SessionError("%s stream cannot seek to frame %d"
                                   % (entry, index))
            cursor = index
        while cursor < index:
            if not cap.grab():
                raise SessionCorrupt("%s stream ended at frame %d, %d were "
                                     "recorded" % (entry, cursor,
                                                   len(self.records)))
            cursor += 1
        ok, picture = cap.read()
        self._next[entry] = cursor + 1
        if not ok or picture is None:
            raise SessionCorrupt("%s stream has no frame %d" % (entry, index))
        return picture


# -- replaying ------------------------------------------------------------

class VideoSource(FrameSource):
    """Serve a recorded session as if it were the camera.

    The same four calls :class:`~.sources.SceneFolderSource` answers, so
    the camera service, the cell loop and every test above them cannot
    tell a session from a folder of scenes -- which is what makes a
    recording usable as a regression fixture and as a demo.

    Deliberately without the folder source's realism knobs. Those exist
    to make a bench run *harder* than the recording; a session exists to
    reproduce exactly what happened, and a replay that adds noise to a
    captured fault is no longer evidence of it.
    """

    name = "session"

    def __init__(self, path: str, scenes: Optional[Sequence[str]] = None,
                 loop: bool = True, fps: Optional[float] = None):
        """Args:
            path: Session directory written by :class:`SessionWriter`.
            scenes: Recorded scene ids to replay, in this order; None
                replays the whole session in recording order.
            loop: Wrap at the end instead of raising SourceExhausted.
            fps: Pace reads to this rate; None or 0 delivers a frame as
                fast as it is asked for. Independent of the rate stamped
                into the containers, which is how fast a human watches.
        """
        self.path = os.path.abspath(path)
        self.loop = bool(loop)
        self.fps = float(fps or 0.0)
        self._requested = tuple(scenes) if scenes else ()
        self._reader = None             # type: Optional[SessionReader]
        self._order = ()                # type: Tuple[int, ...]
        self._cursor = 0
        self._laps = 0
        self._frame_id = 0
        self._next_due = 0.0

    @classmethod
    def from_config(cls, config: Any) -> "VideoSource":
        """Build the source a :class:`~.config.CameraConfig` asks for.

        ``root`` is the session directory. Kept here rather than in
        sources.py so that resolving a source costs no session import on
        a cell replaying a folder or driving a sensor.
        """
        return cls(path=config.root, scenes=config.scene_ids() or None,
                   loop=config.loop, fps=config.fps)

    @property
    def is_live(self) -> bool:
        return False

    def open(self) -> None:
        reader = SessionReader(self.path)
        try:
            reader.open()
        except SessionError as exc:
            # A session that cannot be trusted must stop the service on
            # its launch line: serving it would hand the cell frames
            # nobody can attribute.
            raise FrameSourceError("%s: %s" % (self.path, exc))
        self._reader = reader
        if self._requested:
            missing = [s for s in self._requested if reader.index_of(s) < 0]
            if missing:
                reader.close()
                self._reader = None
                raise FrameSourceError("scenes not in %s: %s"
                                       % (self.path, ", ".join(missing)))
            self._order = tuple(reader.index_of(s) for s in self._requested)
        else:
            self._order = tuple(range(reader.frame_count))
        if not self._order:
            raise FrameSourceError("no frames to replay in %s" % self.path)
        self._cursor = 0
        self._next_due = time.monotonic()
        self.is_open = True

    def read(self) -> Frame:
        if not self.is_open or self._reader is None:
            raise FrameSourceError("read() before open()")
        self._pace()
        index = self._next_index()
        # The id is spent before the decode, so a frame that fails its
        # checksum leaves a gap in the sequence rather than vanishing.
        self._frame_id += 1
        try:
            captured = self._reader.read(index)
        except SessionError as exc:
            raise FrameReadError("%s frame %d: %s" % (self.path, index, exc))
        try:
            # The host clock now, not the recorded one: this frame is
            # arriving now, and the pose service's log is stamped now.
            # When a replay has to be lined up against the robot log of
            # the shift it came from, the capture time is in the sidecar.
            return Frame(frame_id=self._frame_id, timestamp_ns=time.time_ns(),
                         rgb=captured.rgb, depth_raw=captured.depth_raw,
                         K=captured.K, depth_scale=captured.depth_scale,
                         source_name="%s:%s/%s" % (self.name,
                                                   os.path.basename(self.path),
                                                   captured.scene_id))
        except ValueError as exc:
            raise FrameReadError("%s frame %d: %s" % (self.path, index, exc))

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self.is_open = False

    def describe(self) -> Dict[str, Any]:
        described = {"kind": self.name, "live": False, "path": self.path,
                     "scenes": len(self._order), "cursor": self._cursor,
                     "laps": self._laps, "loop": self.loop, "fps": self.fps,
                     "next_scene": self._peek()}
        if self._reader is not None:
            described["session"] = self._reader.describe()
        return described

    # -- internals -------------------------------------------------------

    def _peek(self) -> Optional[str]:
        """The scene the next read would serve, or None when there is none."""
        if not self._order or self._reader is None:
            return None
        if self._cursor >= len(self._order):
            if not self.loop:
                return None
            return self._reader.records[self._order[0]]["scene_id"]
        return self._reader.records[self._order[self._cursor]]["scene_id"]

    def _next_index(self) -> int:
        if self._cursor >= len(self._order):
            if not self.loop:
                raise SourceExhausted("replayed all %d frames of %s"
                                      % (len(self._order), self.path))
            self._cursor = 0
            self._laps += 1
        index = self._order[self._cursor]
        self._cursor += 1
        return index

    def _pace(self) -> None:
        """Hold the configured frame rate."""
        if self.fps > 0:
            now = time.monotonic()
            if now < self._next_due:
                time.sleep(self._next_due - now)
            # From the due time, not from now: a slow read is absorbed by
            # the next period instead of shifting the whole stream.
            self._next_due = max(self._next_due + 1.0 / self.fps,
                                 time.monotonic())
