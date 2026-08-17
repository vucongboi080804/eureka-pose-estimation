"""Resolved configuration for the pose service.

A cell that moves a robot on a pose has to be able to prove afterwards
which weights, thresholds and gates produced it, so the configuration is
frozen at startup, hashed into a digest, and carried in every response.
Absurd or unsatisfiable settings are rejected before a single weight is
read -- a bin-picking cell that comes up misconfigured must fail on the
launch, not on the first pick.

The module is deliberately dependency-free (standard library only). The
entry point loads and validates the configuration *before* importing
NumPy, Open3D or torch, because that is the last moment at which
``OMP_NUM_THREADS`` can still be set: libgomp reads it once, when Open3D
first loads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

#: Log levels the service understands, least to most severe.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


class ConfigError(ValueError):
    """A configuration the service refuses to start with."""


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the service needs, resolved once and provable afterwards.

    Defaults are the shipped configuration: the two segmenters and the
    thresholds behind the reported accuracy, run from the repository root.
    Paths are stored absolute (see :meth:`__post_init__`) so that the
    digest identifies the actual files, not a working directory.
    """

    #: CAD model in the object frame, metres.
    cad_path: str = os.path.join("model", "3d_model.ply")
    #: Primary instance segmenter (ultralytics weights).
    seg_weights: str = os.path.join("weights", "part-seg.pt")
    #: Second segmenter sharing the proposal pool; None to run one alone.
    extra_seg_weights: Optional[str] = os.path.join("weights",
                                                    "part-seg-synthetic.pt")
    #: Inference resolution for both segmenters. 960 is what the weights
    #: were trained and measured at; a slower board can trade recall for
    #: latency here, which is the one knob that moves segmenter time.
    seg_imgsz: int = 960
    #: Confidence floor for a mask to enter the proposal pool. Low on
    #: purpose: wrong proposals cost registration time, not accuracy,
    #: because depth verification disposes of them.
    seg_conf: float = 0.25
    #: Pick mode: stop the sweep at the first pose the pipeline considers
    #: a committed pick, instead of ranking the whole bin. One pick per
    #: cycle is what a cell actually consumes.
    pick: bool = True
    #: The pipeline's own stop threshold (``src.detect_seg.PICK_SCORE``),
    #: restated here so it is digested and returned with every pose.
    #: Startup rejects a value that has drifted away from the pipeline.
    pick_score: float = 0.8
    #: The cell's gate: a top pose below this is reported as ``rescan``.
    #: 0.7 keeps precision 1.00 at 5 mm on cross-validated train scenes
    #: (analysis/score_calibration.md).
    accept_score: float = 0.7
    #: Frames estimated at once. The registration stack is CPU-bound and
    #: already uses every core, and ultralytics models are not re-entrant,
    #: so a second concurrent frame buys nothing and costs memory.
    max_concurrency: int = 1
    #: OMP_NUM_THREADS for Open3D's OpenMP pools; 0 leaves the environment
    #: as it is (which lets libgomp use every core).
    omp_threads: int = 0
    #: Run one dummy inference at startup so the first real frame is not
    #: the slow one. Off shifts several seconds of lazy initialisation
    #: onto the first pick.
    warmup: bool = True
    #: How long a request may wait for a free estimator slot before the
    #: service rejects it. The frame is never touched, so the caller
    #: retries; it is not a failed pick.
    request_timeout_s: float = 30.0
    log_level: str = "INFO"
    #: Loopback by default: the cell talks to a service on its own board.
    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        # Absolute paths make the digest identify files rather than a
        # working directory, and make the startup log line unambiguous
        # when the service is launched by systemd from /.
        for field_name in ("cad_path", "seg_weights", "extra_seg_weights"):
            value = getattr(self, field_name)
            if value:
                object.__setattr__(self, field_name, os.path.abspath(value))
            elif field_name == "extra_seg_weights":
                object.__setattr__(self, field_name, None)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "ServiceConfig":
        """Load a JSON object of field name -> value.

        JSON rather than YAML because the runtime must not grow a
        dependency for its own configuration file, and every field is a
        scalar anyway.
        """
        try:
            with open(path) as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read config %s: %s" % (path, exc))
        if not isinstance(raw, dict):
            raise ConfigError("config %s must be a JSON object" % path)
        return cls._build(raw, source="file %s" % path)

    @classmethod
    def from_env(cls, prefix: str = "POSE_",
                 env: Optional[Mapping[str, str]] = None,
                 base: Optional["ServiceConfig"] = None) -> "ServiceConfig":
        """Overlay ``PREFIX + FIELD_NAME`` environment variables on ``base``.

        ``base`` is the configuration to override (the defaults when it is
        None), which is how the entry point layers a file under the
        environment: a container image ships the file, the orchestrator
        tunes a value without rewriting it.
        """
        env = os.environ if env is None else env
        values = dataclasses.asdict(base) if base is not None else {}
        for field in dataclasses.fields(cls):
            key = prefix + field.name.upper()
            if key in env:
                values[field.name] = env[key]
        return cls._build(values, source="environment %s*" % prefix)

    @classmethod
    def _build(cls, values: Mapping[str, Any], source: str) -> "ServiceConfig":
        """Coerce a mapping of loosely typed values into a config."""
        known = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ConfigError("unknown setting(s) in %s: %s"
                              % (source, ", ".join(unknown)))
        kwargs = {}
        for name, raw in values.items():
            try:
                kwargs[name] = _coerce(raw, known[name].default)
            except (TypeError, ValueError) as exc:
                raise ConfigError("%s: bad value for %s: %s"
                                  % (source, name, exc))
        return cls(**kwargs)

    # -- validation ------------------------------------------------------

    def validate(self) -> "ServiceConfig":
        """Fail loudly on anything the service cannot honour.

        Called once at startup, before any weight is read. Returns self so
        it can be chained onto a constructor.
        """
        for name, path in (("cad_path", self.cad_path),
                           ("seg_weights", self.seg_weights),
                           ("extra_seg_weights", self.extra_seg_weights)):
            if path and not os.path.isfile(path):
                raise ConfigError("%s: no such file: %s" % (name, path))
        if self.seg_imgsz < 64 or self.seg_imgsz > 4096 or self.seg_imgsz % 32:
            raise ConfigError("seg_imgsz must be a multiple of 32 in "
                              "[64, 4096], got %r" % (self.seg_imgsz,))
        for name in ("seg_conf", "pick_score", "accept_score"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ConfigError("%s must be in (0, 1], got %r"
                                  % (name, value))
        if self.pick and self.accept_score > self.pick_score:
            # The sweep would stop at pick_score and the cell would then
            # refuse every pose it was handed: a cell that never picks.
            raise ConfigError(
                "accept_score %.2f above pick_score %.2f in pick mode: the "
                "sweep stops before any pose can clear the gate"
                % (self.accept_score, self.pick_score))
        if not 1 <= self.max_concurrency <= 16:
            raise ConfigError("max_concurrency must be in [1, 16], got %r"
                              % (self.max_concurrency,))
        if not 0 <= self.omp_threads <= 256:
            raise ConfigError("omp_threads must be in [0, 256], got %r"
                              % (self.omp_threads,))
        if self.request_timeout_s <= 0:
            raise ConfigError("request_timeout_s must be positive, got %r"
                              % (self.request_timeout_s,))
        if self.log_level not in LOG_LEVELS:
            raise ConfigError("log_level must be one of %s, got %r"
                              % (", ".join(LOG_LEVELS), self.log_level))
        if not self.host:
            raise ConfigError("host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ConfigError("port must be in [1, 65535], got %r"
                              % (self.port,))
        return self

    # -- provenance ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        """Short stable hash of every setting, returned with every pose."""
        canonical = json.dumps(self.to_dict(), sort_keys=True,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def summary(self) -> str:
        """One line for the startup log."""
        return ("digest=%s cad=%s seg=%s extra=%s imgsz=%d conf=%.2f "
                "pick=%s pick_score=%.2f accept=%.2f concurrency=%d omp=%d "
                "warmup=%s timeout=%.0fs bind=%s:%d"
                % (self.digest(), self.cad_path, self.seg_weights,
                   self.extra_seg_weights or "-", self.seg_imgsz,
                   self.seg_conf, "on" if self.pick else "off",
                   self.pick_score, self.accept_score, self.max_concurrency,
                   self.omp_threads, "on" if self.warmup else "off",
                   self.request_timeout_s, self.host, self.port))


def _coerce(raw: Any, default: Any) -> Any:
    """Convert a JSON or environment value to the type of ``default``.

    The default value carries the type, which keeps this honest under
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
    # Strings, and the optional path whose default is None.
    if raw is None:
        return None
    text = str(raw)
    # Only the optional path treats "" as "not set"; an empty required
    # string is left alone so validate() can name it in the error.
    return (text or None) if default is None else text
