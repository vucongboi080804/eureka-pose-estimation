"""The pose estimator as a long-lived service, with no transport in it.

This is the whole product minus the socket: hand it a frame, get back the
poses, the timing split and the cell's decision. It is deliberately
importable on its own -- a cell that would rather link the estimator into
its own process than talk HTTP to it gets the same behaviour, the same
gates and the same provenance by constructing a :class:`PoseService`.

Nothing here re-implements estimation. The frame goes through exactly the
calls ``scripts/run_pipeline.py`` makes per scene -- ``PoseEstimator``
with the class-level colour mask, then ``detect_scene_hybrid`` -- so the
service and the offline runner cannot drift apart in accuracy. What the
service adds is what a running cell needs and a batch script does not:
one frame's failure never leaves this module as an exception, the timing
of each stage is measured, and the counters a monitoring system scrapes
are kept.

The one thing a request timeout here cannot do is cancel work already
running: the pipeline's geometric safety net takes tens of seconds when
the segmenters find nothing, which is the case a cell must cover with its
own client-side watchdog (deploy/board/README.md).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.masks import part_pixel_mask
from src.detect_seg import PICK_SCORE, detect_scene_hybrid
from src.register import PoseEstimator
from src.scene_io import Scene, load_scene

from . import SERVICE_VERSION
from .adapter import scene_from_arrays
from .config import ServiceConfig
from .models import ModelBundle
from .schema import GATE_PICK, GATE_RESCAN, FrameResult, PoseEstimateDTO

#: How many recent latencies the percentiles are taken over. A cell picks
#: a part every few seconds, so this is roughly the last quarter of an
#: hour -- long enough to be stable, short enough that a board slowing
#: down shows up while the shift is still running.
LATENCY_WINDOW = 256


class ServiceBusy(RuntimeError):
    """Every estimator slot was occupied for the whole request timeout.

    The frame was never looked at, so this is not a failed pick: the
    caller should retry or drop the frame, and the counters keep it apart
    from frames that were estimated and came back empty.
    """


class PoseService:
    """Estimate poses for one frame at a time, and remember how it went."""

    def __init__(self, config: ServiceConfig,
                 bundle: Optional[ModelBundle] = None):
        """Args:
            config: Resolved settings; validated here unless a bundle is
                injected, in which case the caller has already vouched
                for it (that is the seam tests use).
            bundle: An already-loaded bundle to borrow instead of
                building one. The service then does not own it and will
                not close it.
        """
        self.config = config
        self._owns_bundle = bundle is None
        if self._owns_bundle:
            config.validate()
            if config.pick and abs(config.pick_score - PICK_SCORE) > 1e-9:
                # The stop threshold lives in src.detect_seg; a config
                # that claims a different one would put a number in every
                # response that no pose was ever judged against.
                raise ValueError(
                    "pick_score %.3f disagrees with the pipeline's "
                    "PICK_SCORE %.3f" % (config.pick_score, PICK_SCORE))
            bundle = ModelBundle(config)
        self._bundle = bundle
        self._digest = config.digest()
        self._started_at = time.time()
        self._slots = threading.BoundedSemaphore(config.max_concurrency)
        self._counter_lock = threading.Lock()
        self._latencies_ms = deque(maxlen=LATENCY_WINDOW)   # type: deque
        self._frames = 0
        self._poses = 0
        self._failures = 0
        self._rejected = 0
        self._picks = 0
        self._latency_sum_ms = 0.0
        self._last_score = 0.0
        self._labelled = 0

    # -- lifecycle -------------------------------------------------------

    def start(self, frame_shape: Tuple[int, int] = (640, 960)) -> None:
        """Load the weights and, unless configured off, warm them.

        Args:
            frame_shape: ``(height, width)`` of the camera's colour frame,
                used only for the dummy warmup inference. The dataset's
                960x640 is the default because that is what the shipped
                weights were trained at.
        """
        self._bundle.load()
        if self.config.warmup:
            self._bundle.warmup(frame_shape)

    def close(self) -> None:
        if self._owns_bundle:
            self._bundle.close()

    @property
    def is_loaded(self) -> bool:
        return self._bundle.is_loaded

    @property
    def is_ready(self) -> bool:
        """Loaded, and warm if warming was asked for."""
        return self._bundle.is_loaded and (self._bundle.is_warm
                                           or not self.config.warmup)

    # -- estimation ------------------------------------------------------

    def estimate(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, K: Any,
                 depth_scale: float,
                 scene_id: Optional[str] = None) -> FrameResult:
        """Poses for one registered RGB-D frame, best first.

        Args:
            rgb_bgr: (H, W, 3) uint8 colour frame, OpenCV channel order.
            depth_raw: (H, W) raw depth ticks, 0 = no measurement.
            K: (3, 3) intrinsics of the colour frame.
            depth_scale: Metres per depth tick.
            scene_id: Label for logs and the response.

        Returns:
            A :class:`FrameResult`, always. A frame the pipeline cannot
            handle comes back with no poses, ``gate`` = rescan and
            ``error`` set -- a cell treats it exactly like a bin it
            cannot pick from.

        Raises:
            ServiceBusy: no estimator slot became free in time, so
                nothing was estimated. Retry; it is not a failed pick.
        """
        label = scene_id or self._next_label()
        return self._run(label, lambda: scene_from_arrays(
            rgb_bgr, depth_raw, K, depth_scale, scene_id=label))

    def estimate_png_frame(self, rgb_png: bytes, depth_png: bytes, K: Any,
                           depth_scale: float,
                           scene_id: Optional[str] = None) -> FrameResult:
        """Like :meth:`estimate`, from the PNG bytes a camera sends.

        PNG keeps depth as exact integer ticks; a lossy transport of the
        depth map would move poses by more than the accuracy budget.
        """
        label = scene_id or self._next_label()
        return self._run(label, lambda: scene_from_arrays(
            _imdecode(rgb_png, cv2.IMREAD_COLOR, "rgb"),
            _imdecode(depth_png, cv2.IMREAD_UNCHANGED, "depth"),
            K, depth_scale, scene_id=label))

    def estimate_scene_dir(self, path: str) -> FrameResult:
        """Poses for a scene folder (rgb.png, depth.png, camera.json).

        The folder is read by ``src.scene_io.load_scene``, the loader the
        offline runner uses, so a service that shares a filesystem with
        the capture rig reads frames exactly as the batch run does.
        """
        path = os.path.normpath(path)
        split_dir, scene_id = os.path.split(path)
        root, split = os.path.split(split_dir)
        return self._run(scene_id or path,
                         lambda: load_scene(root, split, scene_id))

    def _run(self, label: str,
             build_scene: Callable[[], Scene]) -> FrameResult:
        """Estimate one frame under a slot, timed, and never raising.

        Every entry point funnels through here so that the timing split,
        the concurrency cap and the failure contract are defined once: a
        frame the pipeline cannot handle comes back with no poses,
        ``gate`` = rescan and ``error`` set, which a cell treats exactly
        like a bin it cannot pick from.
        """
        timings = {}    # type: Dict[str, float]
        with self._slot():
            started = time.perf_counter()
            try:
                mark = time.perf_counter()
                scene = build_scene()
                timings["decode_ms"] = _ms_since(mark)
                return self._estimate_scene(scene, timings, started)
            except Exception as exc:      # the failure contract, above
                return self._failed(label, timings, started, exc)

    def _estimate_scene(self, scene: Scene, timings: Dict[str, float],
                        started: float) -> FrameResult:
        """The pipeline, instrumented. Runs under an estimator slot."""
        config = self.config

        mark = time.perf_counter()
        estimator = PoseEstimator(self._bundle.cad, scene.depth, scene.K,
                                  part_mask=part_pixel_mask(scene.rgb))
        timings["prepare_ms"] = _ms_since(mark)

        primary = _SegmenterProbe(self._bundle.primary)
        extra = (_SegmenterProbe(self._bundle.extra)
                 if self._bundle.extra is not None else None)
        mark = time.perf_counter()
        found = detect_scene_hybrid(scene, estimator, primary,
                                    extra_model=extra, conf=config.seg_conf,
                                    pick=config.pick,
                                    imgsz=config.seg_imgsz)
        detect_ms = _ms_since(mark)

        segment_ms = primary.elapsed_ms + (extra.elapsed_ms if extra else 0.0)
        timings["segment_ms"] = round(segment_ms, 1)
        # Registration is everything the detector did that was not
        # segmenter inference: RANSAC, ICP, the flip rivals, verification
        # and the polish. Measured by difference because splitting it
        # finer would mean instrumenting src/, and the ratio -- not the
        # decomposition -- is what sizes a board.
        timings["register_ms"] = round(detect_ms - segment_ms, 1)

        poses = [_pose_dto(est) for est in found]
        gate = (GATE_PICK if poses and poses[0].score >= config.accept_score
                else GATE_RESCAN)
        timings["total_ms"] = _ms_since(started)
        self._record(timings["total_ms"], poses, gate, failed=False)
        return FrameResult(scene_id=scene.scene_id, poses=poses,
                           timings_ms=timings,
                           n_proposals=primary.n_proposals
                           + (extra.n_proposals if extra else 0),
                           gate=gate, config_digest=self._digest,
                           service_version=SERVICE_VERSION)

    def _failed(self, scene_id: str, timings: Dict[str, float],
                started: float, exc: BaseException) -> FrameResult:
        """A frame that could not be estimated, counted as a failure."""
        timings["total_ms"] = _ms_since(started)
        self._record(timings["total_ms"], [], GATE_RESCAN, failed=True)
        return FrameResult(scene_id=scene_id, poses=[], timings_ms=timings,
                           n_proposals=0, gate=GATE_RESCAN,
                           config_digest=self._digest,
                           service_version=SERVICE_VERSION,
                           error="%s: %s" % (type(exc).__name__, exc))

    @contextmanager
    def _slot(self) -> Iterator[None]:
        """Hold one estimator slot for the duration of a frame.

        Concurrency is capped rather than queued without limit:
        registration saturates every core it is given, so a second frame
        in flight only makes both slower and doubles peak memory. A
        caller that cannot get a slot in time is told so immediately,
        which is something a cell can act on -- unlike a request that
        silently queues past its cycle time.
        """
        if not self._slots.acquire(timeout=self.config.request_timeout_s):
            with self._counter_lock:
                self._rejected += 1
            raise ServiceBusy("no estimator slot free after %.1f s"
                              % self.config.request_timeout_s)
        try:
            yield
        finally:
            self._slots.release()

    # -- observability ---------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Liveness and readiness in one object."""
        return {"status": "ready" if self.is_ready
                else ("loading" if not self.is_loaded else "warming"),
                "loaded": self.is_loaded, "ready": self.is_ready,
                "service_version": SERVICE_VERSION,
                "config_digest": self._digest,
                "uptime_s": round(time.time() - self._started_at, 1),
                "memory": self._bundle.memory_footprint()}

    def stats(self) -> Dict[str, Any]:
        """Counters a cell logs and a monitoring system scrapes."""
        with self._counter_lock:
            window = sorted(self._latencies_ms)
            frames, poses, failures = self._frames, self._poses, self._failures
            rejected, picks = self._rejected, self._picks
            latency_sum, last_score = self._latency_sum_ms, self._last_score
        return {"frames": frames, "poses": poses, "failures": failures,
                "rejected": rejected, "picks": picks,
                "rescans": frames - picks,
                "latency_ms": {"p50": _percentile(window, 0.50),
                               "p95": _percentile(window, 0.95),
                               "max": round(window[-1], 1) if window else 0.0,
                               "window": len(window),
                               "sum": round(latency_sum, 1)},
                "last_score": round(last_score, 4)}

    def _record(self, latency_ms: float, poses: Sequence[PoseEstimateDTO],
                gate: str, failed: bool) -> None:
        with self._counter_lock:
            self._frames += 1
            self._poses += len(poses)
            self._latencies_ms.append(latency_ms)
            self._latency_sum_ms += latency_ms
            if failed:
                self._failures += 1
            if gate == GATE_PICK:
                self._picks += 1
            self._last_score = poses[0].score if poses else 0.0

    def _next_label(self) -> str:
        """A name for a frame the caller did not name."""
        with self._counter_lock:
            self._labelled += 1
            return "frame-%d" % self._labelled


class _SegmenterProbe:
    """A segmenter that records how long it took and what it proposed.

    It stands in for the model where the detector calls it, which is the
    only place the two numbers a board is sized by can be read without
    instrumenting ``src/``: inference time, and the masks that entered the
    proposal pool before any of them was registered. The split between
    "the network" and "the geometry" decides whether a slow cycle wants a
    faster CPU or a smaller image; ``n_proposals`` is what collapses first
    when a scene leaves the segmenters' training domain.

    It wraps one frame's calls and is thrown away with the frame, so it
    holds no state shared between requests, and it passes every argument
    through untouched.
    """

    def __init__(self, model: Any):
        self._model = model
        self.elapsed_ms = 0.0
        self.n_proposals = 0

    def __call__(self, image: np.ndarray, **kwargs: Any) -> Any:
        started = time.perf_counter()
        results = self._model(image, **kwargs)
        self.elapsed_ms += _ms_since(started)
        for result in results:
            if result.masks is not None:
                self.n_proposals += len(result.masks.data)
        return results


def _pose_dto(estimate: Any) -> PoseEstimateDTO:
    """Convert one PoseEstimate into the wire form (no NumPy scalars)."""
    return PoseEstimateDTO(
        R=[[float(v) for v in row] for row in np.asarray(estimate.R)],
        t=[float(v) for v in np.asarray(estimate.t).ravel()],
        score=round(float(estimate.submission_score), 4),
        seg_confidence=round(float(estimate.seg_conf), 4),
        depth_verification=round(float(estimate.confidence), 4))


def _imdecode(payload: bytes, flags: int, name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("%s image is not a decodable PNG (%d bytes)"
                         % (name, len(payload)))
    return image


def _ms_since(mark: float) -> float:
    return round((time.perf_counter() - mark) * 1000.0, 1)


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted sequence."""
    if not ordered:
        return 0.0
    rank = max(1, int(round(q * len(ordered))))
    return round(ordered[rank - 1], 1)
