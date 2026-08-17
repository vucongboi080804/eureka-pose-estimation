"""Resolved configuration for the camera service.

A cell that cannot see cannot pick, so everything the camera needs is
settled and checked at startup: the source exists, the scenes it will
replay are on disk, the knobs are inside the ranges they mean anything
in. A camera that comes up misconfigured must fail on its launch line,
not three hours into a shift with an empty gripper.

Standard library only, and no import of NumPy or OpenCV: the entry point
reads and validates a configuration before paying for the imaging stack,
and a badly typed port number should be reported in milliseconds.

Mirrors deploy/pose_service/config.py deliberately -- same JSON file
shape, same ``PREFIX + FIELD`` environment overlay -- so an operator who
has configured one service already knows how to configure the other.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

#: Log levels the service understands, least to most severe.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

#: The sources a frame can come from: recorded scenes replayed as a
#: stream, or an Intel RealSense on the cell.
SOURCE_KINDS = ("scene_folder", "realsense")

#: Highest frame rate the pacing accepts. Above this the replay is
#: bounded by disk and PNG decode anyway, and the number is more likely
#: to be a typo than a request.
MAX_FPS = 240.0

#: Largest simulated depth noise, millimetres. A structured-light sensor
#: that was this wrong at a bin's working distance would be scrapped, so
#: a larger value is a misplaced decimal point.
MAX_DEPTH_NOISE_MM = 100.0

_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


class ConfigError(ValueError):
    """A configuration the service refuses to start with."""


@dataclass(frozen=True)
class CameraConfig:
    """Everything the camera service needs, resolved once.

    Defaults replay this repository's ``test`` split from the repository
    root: the configuration that demonstrates a whole cell on a board
    with no camera attached to it.
    """

    #: Where frames come from; one of :data:`SOURCE_KINDS`.
    source: str = "scene_folder"
    #: Release folder holding the split directories (``root/split/scene``).
    root: str = "."
    split: str = "test"
    #: Comma-separated scene ids to replay, in this order; empty replays
    #: every scene in the split, sorted. A JSON list is accepted too.
    scenes: str = ""
    #: Wrap around at the end of the list instead of refusing further
    #: frames. On for a demo that has to keep running unattended.
    loop: bool = True
    #: Frames per second the source paces itself to; 0 delivers a frame as
    #: fast as it is asked for, which is what a cell taking one frame per
    #: pick actually wants.
    fps: float = 0.0

    #: -- simulation aids, all off by default -------------------------
    #: These make a replay *harder* than the recording, to exercise the
    #: cell's failure paths on a bench. None of them is a sensor model:
    #: they are three independent knobs, not a calibrated RealSense.
    #: Fraction of depth pixels zeroed, standing in for the returns a
    #: shiny or dark surface swallows.
    depth_dropout: float = 0.0
    #: Standard deviation of the noise added to non-zero depth, in
    #: millimetres, rounded back to integer ticks as a sensor would.
    depth_noise_mm: float = 0.0
    #: Uniform 0..N ms of extra latency per read, standing in for a link
    #: that is not as steady as a local disk.
    jitter_ms: float = 0.0
    #: Seed for the two random knobs above, so a bench run that found a
    #: failure can be replayed exactly.
    seed: int = 0

    #: -- RealSense --------------------------------------------------
    #: Serial of the device to open; empty takes the first one found.
    rs_serial: Optional[str] = None
    rs_width: int = 1280
    rs_height: int = 720
    rs_fps: int = 30

    log_level: str = "INFO"
    #: Loopback by default: a camera service is not something to expose
    #: to a plant network by accident. 8081 leaves 8080 to the pose
    #: service, so both can run on one board unconfigured.
    host: str = "127.0.0.1"
    port: int = 8081

    def __post_init__(self) -> None:
        # Absolute root: the startup log line and every error message
        # should be unambiguous when systemd launched the unit from /.
        object.__setattr__(self, "root", os.path.abspath(self.root))
        if not self.rs_serial:
            object.__setattr__(self, "rs_serial", None)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "CameraConfig":
        """Load a JSON object of field name -> value."""
        try:
            with open(path) as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read config %s: %s" % (path, exc))
        if not isinstance(raw, dict):
            raise ConfigError("config %s must be a JSON object" % path)
        return cls._build(raw, source="file %s" % path)

    @classmethod
    def from_env(cls, prefix: str = "CAM_",
                 env: Optional[Mapping[str, str]] = None,
                 base: Optional["CameraConfig"] = None) -> "CameraConfig":
        """Overlay ``PREFIX + FIELD_NAME`` environment variables on ``base``.

        ``base`` is the configuration being overridden (the defaults when
        it is None), which is how the entry point layers a file under the
        environment: the image ships the file, the operator moves the
        camera to another split without editing it.
        """
        env = os.environ if env is None else env
        values = dataclasses.asdict(base) if base is not None else {}
        for field in dataclasses.fields(cls):
            key = prefix + field.name.upper()
            if key in env:
                values[field.name] = env[key]
        return cls._build(values, source="environment %s*" % prefix)

    @classmethod
    def _build(cls, values: Mapping[str, Any], source: str) -> "CameraConfig":
        """Coerce a mapping of loosely typed values into a config."""
        known = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ConfigError("unknown setting(s) in %s: %s"
                              % (source, ", ".join(unknown)))
        kwargs = {}     # type: Dict[str, Any]
        for name, raw in values.items():
            if name == "scenes" and isinstance(raw, (list, tuple)):
                # A JSON config naturally writes a list; the environment
                # can only write a string. Both end up as one string.
                raw = ",".join(str(item) for item in raw)
            try:
                kwargs[name] = _coerce(raw, known[name].default)
            except (TypeError, ValueError) as exc:
                raise ConfigError("%s: bad value for %s: %s"
                                  % (source, name, exc))
        return cls(**kwargs)

    # -- derived ---------------------------------------------------------

    def scene_ids(self) -> Tuple[str, ...]:
        """The requested scene ids, in order; empty means the whole split."""
        return tuple(s.strip() for s in self.scenes.split(",") if s.strip())

    def split_dir(self) -> str:
        return os.path.join(self.root, self.split)

    # -- validation ------------------------------------------------------

    def validate(self) -> "CameraConfig":
        """Fail loudly on anything the service cannot honour.

        Called once at startup, before the source is opened. Returns self
        so it can be chained onto a constructor.
        """
        if self.source not in SOURCE_KINDS:
            raise ConfigError("source must be one of %s, got %r"
                              % (", ".join(SOURCE_KINDS), self.source))
        if self.source == "scene_folder":
            self._validate_scene_folder()
        else:
            self._validate_realsense()
        if not 0.0 <= self.fps <= MAX_FPS:
            raise ConfigError("fps must be in [0, %g], got %r"
                              % (MAX_FPS, self.fps))
        if not 0.0 <= self.depth_dropout < 1.0:
            # 1.0 would zero every pixel: a camera that reports nothing
            # is a broken camera, not a hard scene.
            raise ConfigError("depth_dropout must be in [0, 1), got %r"
                              % (self.depth_dropout,))
        if not 0.0 <= self.depth_noise_mm <= MAX_DEPTH_NOISE_MM:
            raise ConfigError("depth_noise_mm must be in [0, %g], got %r"
                              % (MAX_DEPTH_NOISE_MM, self.depth_noise_mm))
        if not 0.0 <= self.jitter_ms <= 10000.0:
            raise ConfigError("jitter_ms must be in [0, 10000], got %r"
                              % (self.jitter_ms,))
        if self.log_level not in LOG_LEVELS:
            raise ConfigError("log_level must be one of %s, got %r"
                              % (", ".join(LOG_LEVELS), self.log_level))
        if not self.host:
            raise ConfigError("host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ConfigError("port must be in [1, 65535], got %r"
                              % (self.port,))
        return self

    def _validate_scene_folder(self) -> None:
        """Prove the replay can produce frames before anything is served."""
        if not os.path.isdir(self.root):
            raise ConfigError("root: no such directory: %s" % self.root)
        split_dir = self.split_dir()
        if not os.path.isdir(split_dir):
            raise ConfigError("split: no such directory: %s" % split_dir)
        requested = self.scene_ids()
        if requested:
            missing = [s for s in requested
                       if not os.path.isdir(os.path.join(split_dir, s))]
            if missing:
                raise ConfigError("scenes: not in %s: %s"
                                  % (split_dir, ", ".join(missing)))
        elif not any(os.path.isdir(os.path.join(split_dir, name))
                     for name in os.listdir(split_dir)):
            raise ConfigError("split %s holds no scene folders" % split_dir)

    def _validate_realsense(self) -> None:
        if self.rs_width < 64 or self.rs_height < 64:
            raise ConfigError("rs_width/rs_height must be at least 64, got "
                              "%rx%r" % (self.rs_width, self.rs_height))
        if not 1 <= self.rs_fps <= MAX_FPS:
            raise ConfigError("rs_fps must be in [1, %g], got %r"
                              % (MAX_FPS, self.rs_fps))
        if self.scene_ids() or self.depth_dropout or self.depth_noise_mm:
            # Silently ignoring them would let a bench configuration
            # reach a real cell and be believed.
            raise ConfigError("scenes and the depth realism knobs apply to "
                              "the scene_folder source only")

    # -- provenance ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        """One line for the startup log."""
        if self.source == "scene_folder":
            where = "%s scenes=%s" % (self.split_dir(),
                                      ",".join(self.scene_ids()) or "all")
        else:
            where = "realsense serial=%s %dx%d@%d" % (
                self.rs_serial or "first", self.rs_width, self.rs_height,
                self.rs_fps)
        return ("source=%s %s loop=%s fps=%g dropout=%.3f noise=%.1fmm "
                "jitter=%.0fms seed=%d bind=%s:%d"
                % (self.source, where, "on" if self.loop else "off", self.fps,
                   self.depth_dropout, self.depth_noise_mm, self.jitter_ms,
                   self.seed, self.host, self.port))


def _coerce(raw: Any, default: Any) -> Any:
    """Convert a JSON or environment value to the type of ``default``.

    The default value carries the type, which stays honest under
    ``from __future__ import annotations`` (where field annotations are
    strings) and needs no parallel type table to drift out of date.
    """
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError("expected a boolean, got %r" % (raw,))
    if isinstance(default, int):
        return int(str(raw).strip())
    if isinstance(default, float):
        return float(str(raw).strip())
    # Strings, and the optional serial whose default is None.
    if raw is None:
        return None
    return str(raw)
