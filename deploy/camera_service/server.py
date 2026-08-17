"""HTTP front end for the camera service.

    .venv/bin/python -m deploy.camera_service.server --config camera.json

The transport is the standard library's threading HTTP server, for the
same reasons the pose service gives: this runs beside the estimator on a
4 GB board that has to be installed through an air gap, so the camera is
not allowed to cost a dependency tree. The endpoints are the conventional
ones (``/healthz``, ``/readyz``, ``/metrics``) plus the two the cell
actually consumes:

    GET /v1/frame        the next frame, in the body POST /v1/estimate
                         accepts (deploy/pose_service/schema.py)
    GET /v1/intrinsics   K, depth_scale and the image size, without
                         taking a frame off the stream
    GET /preview.mjpg    the colour frames as MJPEG, so a human can watch
                         the cell in a browser
    GET /                a page holding that stream in an <img>

Frames are pulled, never pushed: one read per request, serialised by a
lock because a camera is one device. The preview never pulls a frame of
its own -- it re-serves the last frame the cell was given, so watching
the cell cannot change what the cell sees next, and a browser left open
in a control room cannot starve a pick.

Status codes mean what they say, so a cell can branch on them without
parsing prose: 503 the camera is not ready or is not delivering, 410 a
finite replay has ended, 404 no such route.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, Optional, Tuple

import cv2

from . import CAMERA_VERSION, WIRE_VERSION
from .config import CameraConfig, ConfigError
from .frame import Frame
from .sources import (FrameReadError, FrameSource, FrameSourceError,
                      SourceExhausted, source_from_config)

#: How many recent reads the latency percentiles and the frame-rate
#: estimate are taken over. At the few frames a second a cell consumes
#: this is the last minute or two: long enough to be stable, short
#: enough that a camera slowing down shows up while the shift is running.
LATENCY_WINDOW = 128

#: Unreadable frames skipped inside one request before the caller is
#: told the camera is not delivering. Skipping a bad frame is normal (a
#: half-written capture, a dropped packet); skipping forever would turn
#: a dead camera into a hung request.
MAX_READ_ATTEMPTS = 3

#: Upper bound on preview parts per second. The preview only re-serves
#: frames the cell already took, so this caps the JPEG encoding a watched
#: service does, not the stream itself.
PREVIEW_MAX_FPS = 10.0

#: How long a preview connection waits for a new frame before re-sending
#: the current one. A browser needs traffic to keep the connection and
#: the picture alive; a second is also the longest a graceful shutdown
#: waits for a watcher to notice.
PREVIEW_KEEPALIVE_S = 1.0

#: JPEG quality of the preview. It is for a human deciding whether the
#: bin looks right; the estimator never sees these bytes -- it gets the
#: lossless PNGs from /v1/frame.
PREVIEW_QUALITY = 80

_LOG = logging.getLogger("camera_service")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Every setting can also be given as CAM_<FIELD> in the "
               "environment, which overrides the file.")
    parser.add_argument("--config", default=None,
                        help="JSON file of CameraConfig fields; defaults "
                             "and CAM_* environment variables fill the rest")
    parser.add_argument("--once", action="store_true",
                        help="Open the source, answer exactly one request "
                             "and exit -- the smoke test after an install")
    args = parser.parse_args(argv)

    try:
        base = (CameraConfig.from_file(args.config) if args.config
                else CameraConfig())
        config = CameraConfig.from_env(base=base).validate()
    except ConfigError as exc:
        print("configuration error: %s" % exc, file=sys.stderr)
        return 2

    _configure_logging(config.log_level)
    _LOG.info("starting", extra={"fields": {"version": CAMERA_VERSION,
                                            "wire": WIRE_VERSION,
                                            "config": config.summary()}})
    stream = CameraStream(source_from_config(config))
    httpd = _CameraHTTPServer((config.host, config.port), _Handler)
    httpd.stream = stream
    httpd.config = config
    try:
        return _serve(httpd, stream, config, once=args.once)
    finally:
        httpd.server_close()
        stream.close()


def _serve(httpd: "_CameraHTTPServer", stream: "CameraStream",
           config: CameraConfig, once: bool) -> int:
    """Run until a signal arrives, or until one request in ``once`` mode."""
    if once:
        # Open before answering: the single request this mode exists for
        # is meant to be a real frame, not a 503.
        stream.start()
        _LOG.info("ready", extra={"fields": stream.health()})
        httpd.timeout = 60.0
        httpd.handle_request()
        return 0

    stop = threading.Event()
    failure = []            # type: list

    def _open() -> None:
        try:
            stream.start()
        except (FrameSourceError, FrameReadError, SourceExhausted) as exc:
            # A camera that can never deliver should die on its launch
            # line, not sit at 503 waiting for someone to notice.
            _LOG.error("source failed", extra={"fields": {
                "error": "%s: %s" % (type(exc).__name__, exc)}})
            failure.append(exc)
            stop.set()
            return
        _LOG.info("ready", extra={"fields": stream.health()})

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda s, f: stop.set())
    threading.Thread(target=_open, name="open", daemon=True).start()
    threading.Thread(target=httpd.serve_forever, name="http",
                     daemon=True).start()
    _LOG.info("listening", extra={"fields": {"host": config.host,
                                             "port": config.port}})
    stop.wait()
    _LOG.info("draining", extra={"fields": stream.stats()})
    # Wake the preview connections first: they hold handler threads that
    # server_close() would otherwise wait on for as long as a browser is
    # open. shutdown() then stops the accept loop, and the caller's
    # server_close() joins whatever frame is still being read.
    stream.stop()
    httpd.shutdown()
    return 1 if failure else 0


class CameraStream:
    """An opened source, its counters, and the frame the preview shows.

    The transport-free half of the service: it owns the source's
    lifecycle, serialises reads, keeps the numbers a monitoring system
    scrapes, and publishes each frame to whoever is watching. A cell that
    would rather link the camera into its own process than talk HTTP to
    it constructs one of these.
    """

    def __init__(self, source: FrameSource):
        self.source = source
        self.started_at = time.time()
        self.stopping = threading.Event()
        # One device, one reader. Held across a whole read (including the
        # source's own pacing), so a second caller waits rather than two
        # requests interleaving on one sensor.
        self._read_lock = threading.Lock()
        # Guards everything a watcher reads: the published frame, the
        # encoded preview and the counters.
        self._cond = threading.Condition()
        self._latest = None             # type: Optional[Frame]
        self._pending = None            # type: Optional[Frame]
        self._preview_jpeg = None       # type: Optional[Tuple[int, bytes]]
        self._frames = 0
        self._read_errors = 0
        self._preview_clients = 0
        self._latencies = deque(maxlen=LATENCY_WINDOW)    # type: Deque[float]
        self._arrivals = deque(maxlen=LATENCY_WINDOW)     # type: Deque[float]

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Open the source and prove it delivers, by reading one frame.

        Readiness means "this camera has produced a frame", not "the
        object was constructed": an operator watching /readyz should
        learn about an unplugged sensor before the first pick, not
        during it.

        The proving frame is kept for the first caller when the source is
        a replay -- dropping it would mean the first recorded scene were
        never served -- and dropped when the source is live, where a
        frame captured at startup is stale by definition.
        """
        self.source.open()
        frame = self._read_once()
        if not self.source.is_live:
            with self._cond:
                self._pending = frame

    def stop(self) -> None:
        """Tell the preview connections to finish. Safe to call twice."""
        self.stopping.set()
        with self._cond:
            self._cond.notify_all()

    def close(self) -> None:
        """Release the source. Safe to call twice, and on a failed start."""
        self.stop()
        self.source.close()

    @property
    def is_ready(self) -> bool:
        return self.source.is_open and self._frames > 0

    # -- frames ----------------------------------------------------------

    def next_frame(self) -> Frame:
        """The next frame, skipping up to :data:`MAX_READ_ATTEMPTS` bad ones.

        Raises:
            FrameReadError: nothing readable came out of the source.
            SourceExhausted: a finite replay has ended.
        """
        with self._read_lock:
            with self._cond:
                pending, self._pending = self._pending, None
            if pending is not None:
                return pending
            last = None         # type: Optional[FrameReadError]
            for _ in range(MAX_READ_ATTEMPTS):
                try:
                    return self._read_once()
                except FrameReadError as exc:
                    # A frame the source could not deliver is counted and
                    # skipped: one bad capture must not end a shift.
                    last = exc
                    with self._cond:
                        self._read_errors += 1
                    _LOG.warning("frame skipped", extra={"fields": {
                        "error": str(exc), "read_errors": self._read_errors}})
            raise last if last is not None else FrameReadError("no frame")

    def _read_once(self) -> Frame:
        """One timed read, published to the watchers and counted."""
        started = time.perf_counter()
        frame = self.source.read()
        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._cond:
            self._latest = frame
            self._frames += 1
            self._latencies.append(latency_ms)
            self._arrivals.append(time.monotonic())
            self._cond.notify_all()
        _LOG.debug("frame", extra={"fields": {
            "frame_id": frame.frame_id, "source": frame.source_name,
            "latency_ms": round(latency_ms, 1),
            "valid_depth": round(frame.valid_depth_fraction(), 3)}})
        return frame

    def intrinsics(self) -> Optional[Dict[str, Any]]:
        """Calibration and image size, from the last frame seen.

        Answered from the frame already in hand rather than by taking a
        new one: a cell asking what the camera's geometry is must not
        consume the frame it was about to pick from.
        """
        with self._cond:
            frame = self._latest
        if frame is None:
            return None
        return {"K": [[float(v) for v in row] for row in frame.K],
                "depth_scale": frame.depth_scale,
                "width": frame.width, "height": frame.height,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source": frame.source_name,
                "camera_version": CAMERA_VERSION,
                "wire_version": WIRE_VERSION}

    def preview_jpeg(self, after_id: int,
                     timeout: float) -> Optional[Tuple[int, bytes]]:
        """The current colour frame as JPEG, waiting briefly for a newer one.

        Returns ``(frame_id, bytes)``, or None while no frame has been
        read yet. The encoding is cached by frame id, so ten people
        watching cost one encode per frame, not ten.
        """
        with self._cond:
            if self._latest is not None and self._latest.frame_id == after_id:
                self._cond.wait(timeout)
            frame = self._latest
            cached = self._preview_jpeg
        if frame is None:
            return None
        if cached is not None and cached[0] == frame.frame_id:
            return cached
        ok, buffer = cv2.imencode(".jpg", frame.rgb,
                                  [int(cv2.IMWRITE_JPEG_QUALITY),
                                   PREVIEW_QUALITY])
        if not ok:
            return cached
        entry = (frame.frame_id, buffer.tobytes())
        with self._cond:
            self._preview_jpeg = entry
        return entry

    def watcher(self, delta: int) -> None:
        """Count a preview connection arriving (+1) or leaving (-1)."""
        with self._cond:
            self._preview_clients += delta

    # -- numbers ---------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        with self._cond:
            frame, frames = self._latest, self._frames
            errors = self._read_errors
        ready = self.is_ready
        return {"status": "ok" if ready else "starting",
                "open": self.source.is_open, "ready": ready,
                "camera_version": CAMERA_VERSION, "wire_version": WIRE_VERSION,
                "uptime_s": round(time.time() - self.started_at, 1),
                "frames": frames, "read_errors": errors,
                "last_frame": frame.summary() if frame is not None else None,
                "source": self.source.describe()}

    def stats(self) -> Dict[str, Any]:
        # Everything a scrape reports is taken in one go, so the numbers
        # in one exposition describe one moment rather than three.
        with self._cond:
            latencies = sorted(self._latencies)
            arrivals = list(self._arrivals)
            frame, frames = self._latest, self._frames
            errors, watchers = self._read_errors, self._preview_clients
        return {"frames": frames, "read_errors": errors,
                "preview_clients": watchers,
                "fps": _rate(arrivals),
                "last_frame_id": frame.frame_id if frame else 0,
                "last_timestamp_ns": frame.timestamp_ns if frame else 0,
                "latency_ms": {"p50": _percentile(latencies, 0.50),
                               "p95": _percentile(latencies, 0.95),
                               "sum": round(sum(latencies), 1),
                               "window": len(latencies)}}


class _CameraHTTPServer(ThreadingHTTPServer):
    """Threading server that waits for in-flight reads on close."""

    daemon_threads = False      # so server_close() joins the handlers
    allow_reuse_address = True

    stream = None               # type: Any
    config = None               # type: Any

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Keep the log one JSON object per line whatever a peer does.

        The default prints a bare traceback to stderr, which the
        collector reading this stream cannot parse -- and a browser
        closing a preview tab or a cell abandoning a slow request is not
        an incident worth a stack trace. Anything else still gets logged
        at ERROR, with the type and message that identify it.
        """
        exc = sys.exc_info()[1]
        peer = "%s:%s" % client_address if client_address else "?"
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            _LOG.debug("peer hung up", extra={"fields": {
                "peer": peer, "error": str(exc)}})
            return
        _LOG.error("request failed", extra={"fields": {
            "peer": peer, "error": "%s: %s" % (type(exc).__name__, exc)}})


class _Handler(BaseHTTPRequestHandler):
    """One request. Routing and status codes only -- no capture here."""

    server_version = "camera-service/" + CAMERA_VERSION
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        stream = self.server.stream
        if route == "/healthz":
            health = stream.health()
            self._json(200 if health["open"] else 503, health)
        elif route == "/readyz":
            health = stream.health()
            self._json(200 if health["ready"] else 503, health)
        elif route == "/metrics":
            self._text(200, _render_metrics(stream),
                       "text/plain; version=0.0.4; charset=utf-8")
        elif route == "/stats":
            self._json(200, stream.stats())
        elif route == "/v1/frame":
            self._frame(stream)
        elif route == "/v1/intrinsics":
            self._intrinsics(stream)
        elif route == "/preview.mjpg":
            self._preview(stream)
        elif route == "/":
            self._text(200, _PREVIEW_PAGE, "text/html; charset=utf-8")
        else:
            self._error(404, "no such route: %s" % route)

    # -- routes ----------------------------------------------------------

    def _frame(self, stream: CameraStream) -> None:
        if not stream.is_ready:
            self._json(503, {"error": "camera is not ready",
                             "health": stream.health()})
            return
        started = time.perf_counter()
        try:
            frame = stream.next_frame()
        except SourceExhausted as exc:
            # 410 rather than 503: the stream is not coming back, and a
            # replay ending is how an unattended demo stops cleanly.
            self._error(410, str(exc))
            return
        except FrameReadError as exc:
            self._error(503, "camera is not delivering frames: %s" % exc)
            return
        except Exception as exc:                    # noqa: BLE001
            _LOG.error("handler failed", extra={"fields": {
                "error": "%s: %s" % (type(exc).__name__, exc)}})
            self._error(500, "internal error")
            return
        body = frame.to_wire()
        _LOG.info("frame", extra={"fields": {
            "frame_id": frame.frame_id, "source": frame.source_name,
            "valid_depth": round(frame.valid_depth_fraction(), 3),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 1)}})
        self._json(200, body)

    def _intrinsics(self, stream: CameraStream) -> None:
        intrinsics = stream.intrinsics()
        if intrinsics is None:
            self._json(503, {"error": "no frame has been read yet",
                             "health": stream.health()})
            return
        self._json(200, intrinsics)

    def _preview(self, stream: CameraStream) -> None:
        """Stream the colour frames as multipart JPEG until the peer goes."""
        boundary = "framebound"
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=%s" % boundary)
        # No Content-Length is possible on a stream, so the connection
        # itself delimits the response and must be closed at the end.
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True
        stream.watcher(+1)
        last_id = -1
        try:
            while not stream.stopping.is_set():
                entry = stream.preview_jpeg(last_id, PREVIEW_KEEPALIVE_S)
                if entry is None:               # nothing captured yet
                    continue
                last_id, jpeg = entry
                self.wfile.write(
                    ("--%s\r\nContent-Type: image/jpeg\r\n"
                     "Content-Length: %d\r\nX-Frame-Id: %d\r\n\r\n"
                     % (boundary, len(jpeg), last_id)).encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(1.0 / PREVIEW_MAX_FPS)
        except (BrokenPipeError, ConnectionResetError):
            # The browser tab was closed. Not an event worth an ERROR.
            _LOG.debug("preview closed", extra={"fields": {
                "peer": self.address_string()}})
        finally:
            stream.watcher(-1)

    # -- plumbing --------------------------------------------------------

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        self._text(status, json.dumps(payload) + "\n", "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message, "wire_version": WIRE_VERSION})

    def _text(self, status: int, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route the server's own chatter into the structured log."""
        _LOG.debug("http", extra={"fields": {"peer": self.address_string(),
                                             "message": fmt % args}})


#: The preview stream in a page, because Chrome renders
#: multipart/x-mixed-replace inside an <img> but no longer as a document.
_PREVIEW_PAGE = """<!doctype html>
<title>camera service</title>
<style>body{background:#111;color:#ccc;font:14px sans-serif;margin:0;
padding:1rem}img{max-width:100%;image-rendering:pixelated}</style>
<p>colour frames as the cell receives them
(<a href="/healthz" style="color:#8bf">health</a>,
<a href="/metrics" style="color:#8bf">metrics</a>)</p>
<img src="/preview.mjpg" alt="camera preview">
"""


def _render_metrics(stream: CameraStream) -> str:
    """Prometheus text exposition, format version 0.0.4."""
    stats = stream.stats()
    latency = stats["latency_ms"]
    lines = [
        "# HELP camera_build_info Camera service version and wire version.",
        "# TYPE camera_build_info gauge",
        'camera_build_info{version="%s",wire="%s",source="%s"} 1'
        % (CAMERA_VERSION, WIRE_VERSION, stream.source.name),
        "# HELP camera_ready Source open and at least one frame read.",
        "# TYPE camera_ready gauge",
        "camera_ready %d" % (1 if stream.is_ready else 0),
        "# HELP camera_frames_total Frames delivered by the source.",
        "# TYPE camera_frames_total counter",
        "camera_frames_total %d" % stats["frames"],
        "# HELP camera_read_errors_total Frames the source could not "
        "deliver.",
        "# TYPE camera_read_errors_total counter",
        "camera_read_errors_total %d" % stats["read_errors"],
        "# HELP camera_fps Frames a second over the last %d reads."
        % latency["window"],
        "# TYPE camera_fps gauge",
        "camera_fps %.2f" % stats["fps"],
        "# HELP camera_preview_clients Open MJPEG preview connections.",
        "# TYPE camera_preview_clients gauge",
        "camera_preview_clients %d" % stats["preview_clients"],
        "# HELP camera_last_frame_id Id of the most recent frame.",
        "# TYPE camera_last_frame_id gauge",
        "camera_last_frame_id %d" % stats["last_frame_id"],
        "# HELP camera_read_latency_ms Read latency; quantiles over the "
        "last %d." % latency["window"],
        "# TYPE camera_read_latency_ms summary",
        'camera_read_latency_ms{quantile="0.5"} %.1f' % latency["p50"],
        'camera_read_latency_ms{quantile="0.95"} %.1f' % latency["p95"],
        "camera_read_latency_ms_sum %.1f" % latency["sum"],
        "camera_read_latency_ms_count %d" % stats["frames"],
    ]
    return "\n".join(lines) + "\n"


def _percentile(ordered: list, q: float) -> float:
    if not ordered:
        return 0.0
    rank = max(1, int(round(q * len(ordered))))
    return round(ordered[rank - 1], 1)


def _rate(arrivals: list) -> float:
    """Frames a second across the window, 0 until two frames have landed."""
    if len(arrivals) < 2:
        return 0.0
    span = arrivals[-1] - arrivals[0]
    return round((len(arrivals) - 1) / span, 2) if span > 0 else 0.0


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, on stderr, for a log collector to read.

    The same shape the pose service emits, written out again rather than
    imported from it: the camera has to be deployable on a machine that
    carries no estimator, and ten lines of formatter are a cheaper price
    for that than a package dependency in the wrong direction.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3),
                   "level": record.levelname,
                   "event": record.getMessage(),
                   "thread": record.threadName}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    _LOG.handlers = [handler]
    _LOG.setLevel(getattr(logging, level))
    _LOG.propagate = False


if __name__ == "__main__":
    sys.exit(main())
