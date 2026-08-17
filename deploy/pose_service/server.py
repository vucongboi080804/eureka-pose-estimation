"""HTTP front end for the pose service.

    .venv/bin/python -m deploy.pose_service.server --config cell.json

The transport is the standard library's threading HTTP server. A cell
talks to this over loopback on its own board: there is no fan-out, no
authentication boundary and no third party, and one estimator saturates
the CPU anyway -- so the traffic a real framework would buy capacity for
does not exist, while its dependency footprint would have to be pinned,
carried through an air gap and kept building on aarch64/glibc 2.27. The
endpoints are the conventional ones (``/healthz``, ``/readyz``,
``/metrics``) so a Kubernetes probe or a Prometheus scrape needs no
adapter.

Anything before the estimator is transport: base64, HTTP status codes,
the request body. Anything after it is :mod:`service`. The status codes
mean what they say -- 400 the request is malformed, 422 the frame could
not be read, 503 the service is not ready or has no free slot -- because
a cell has to branch on them without parsing prose.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import dataclasses
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import SCHEMA_VERSION, SERVICE_VERSION
from .config import ConfigError, ServiceConfig
from .schema import EstimateRequest, RequestError

#: Largest request body accepted, bytes. Two PNGs of a 960x640 frame
#: base64-encoded come to a couple of megabytes; this leaves room for a
#: much larger sensor and still refuses a body that would exhaust a 4 GB
#: board before it could be parsed.
MAX_BODY_BYTES = 64 * 1024 * 1024

#: glibc's mallopt parameter for the arena cap (malloc.h, M_ARENA_MAX).
_M_ARENA_MAX = -8

_LOG = logging.getLogger("pose_service")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Every setting can also be given as POSE_<FIELD> in the "
               "environment, which overrides the file.")
    parser.add_argument("--config", default=None,
                        help="JSON file of ServiceConfig fields; defaults "
                             "and POSE_* environment variables fill the rest")
    parser.add_argument("--once", action="store_true",
                        help="Load the models, answer exactly one request "
                             "and exit -- the smoke test after an install")
    # The two settings that belong to an invocation rather than to a
    # deployment: a second instance on a spare port while the cell's own
    # keeps running, or a temporary bind off loopback while commissioning.
    # Everything else is the config file and POSE_*.
    parser.add_argument("--host", default=None,
                        help="Override the configured bind address")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the configured port")
    args = parser.parse_args(argv)

    try:
        base = (ServiceConfig.from_file(args.config) if args.config
                else ServiceConfig())
        config = ServiceConfig.from_env(base=base)
        overrides = {name: value
                     for name, value in (("host", args.host),
                                         ("port", args.port))
                     if value is not None}
        if overrides:
            config = dataclasses.replace(config, **overrides)
        config = config.validate()
    except ConfigError as exc:
        print("configuration error: %s" % exc, file=sys.stderr)
        return 2

    _configure_logging(config.log_level)
    if config.omp_threads:
        # Before the first heavy import: libgomp reads OMP_NUM_THREADS
        # once, when Open3D loads it.
        os.environ["OMP_NUM_THREADS"] = str(config.omp_threads)
    arenas = _cap_malloc_arenas()
    _LOG.info("starting", extra={"fields": {"version": SERVICE_VERSION,
                                            "schema": SCHEMA_VERSION,
                                            "malloc_arena_max": arenas,
                                            "config": config.summary()}})

    # Bind before importing the estimator: NumPy, Open3D and torch cost
    # seconds on a desktop and half a minute on an A57, and a caller that
    # starts the service and immediately probes it should be told "not
    # ready", not "connection refused". The two mean different things to a
    # cell, to an install script and to a container probe.
    try:
        httpd = _PoseHTTPServer((config.host, config.port), _Handler)
    except OSError as exc:
        # A port already taken is the ordinary way a second instance is
        # started by mistake; systemd would restart into it five times
        # before giving up, so say what happened in one line.
        print("cannot bind %s:%d: %s" % (config.host, config.port, exc),
              file=sys.stderr)
        return 2
    httpd.config = config
    _LOG.info("listening", extra={"fields": {"host": config.host,
                                             "port": config.port}})
    try:
        return _serve(httpd, config, once=args.once)
    finally:
        httpd.server_close()
        if httpd.service is not None:
            httpd.service.close()


def _build_service(config: ServiceConfig) -> Any:
    """Import the estimator and load the weights. Slow, and never on the
    accept path -- that is the whole reason the socket is bound first."""
    from .service import PoseService     # noqa: E402 -- after the env is set
    service = PoseService(config)
    service.start()
    return service


def _serve(httpd: "_PoseHTTPServer", config: ServiceConfig,
           once: bool) -> int:
    """Run until a signal arrives, or until one request in ``once`` mode."""
    if once:
        # Load first: the single request this mode answers is meant to be a
        # real pick, not the 503 a probe would get. The socket is already
        # bound, so an install script may fire its request immediately and
        # have it wait in the backlog instead of losing a race.
        httpd.service = _build_service(config)
        _LOG.info("ready", extra={"fields": httpd.service.health()})
        httpd.timeout = max(60.0, config.request_timeout_s * 2)
        httpd.handle_request()
        return 0

    stop = threading.Event()
    failure = []            # type: list

    def _load() -> None:
        try:
            httpd.service = _build_service(config)
        except Exception as exc:                    # noqa: BLE001
            # A service that can never answer should die on its launch
            # line, not sit at 503 waiting for someone to notice.
            _LOG.error("model load failed", extra={"fields": {
                "error": "%s: %s" % (type(exc).__name__, exc)}})
            failure.append(exc)
            stop.set()
            return
        _LOG.info("ready", extra={"fields": httpd.service.health()})

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda s, f: stop.set())
    threading.Thread(target=_load, name="load", daemon=True).start()
    threading.Thread(target=httpd.serve_forever, name="http",
                     daemon=True).start()
    stop.wait()
    if httpd.service is not None:
        _LOG.info("draining", extra={"fields": httpd.service.stats()})
    # shutdown() stops the accept loop; server_close() (in the caller)
    # joins the request threads, so a pick already being estimated is
    # still delivered.
    httpd.shutdown()
    return 1 if failure else 0


def _cap_malloc_arenas() -> Optional[int]:
    """Stop glibc from giving every request thread its own heap arena.

    ``http.server`` runs each request on a new thread, and glibc hands a
    new thread its own malloc arena, whose freed memory it then never
    returns to the OS. Measured on this pipeline (60 frames of
    test/000001, pick mode): the process grew 513 MB with arenas
    uncapped and 53 MB with one arena -- the difference between a
    service that survives a shift on a 4 GB board and one that does not.
    Registration is the only heavy allocator and it runs one frame at a
    time, so a single arena costs no throughput here.

    An explicit ``MALLOC_ARENA_MAX`` in the environment wins; glibc reads
    that at process start, and an operator who set it meant it.

    Returns:
        The cap applied, or None where nothing was capped (a non-glibc
        libc, or the operator already chose).
    """
    if "MALLOC_ARENA_MAX" in os.environ:
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        if libc.mallopt(_M_ARENA_MAX, 1) != 1:
            return None
    except (OSError, AttributeError):
        return None
    return 1


class _PoseHTTPServer(ThreadingHTTPServer):
    """Threading server that waits for in-flight requests on close."""

    daemon_threads = False      # so server_close() joins the handlers
    allow_reuse_address = True

    #: The estimator -- None until the weights are loaded. The socket is
    #: bound before the estimator is imported, so every route has to answer
    #: during that window instead of assuming one exists.
    service = None              # type: Any
    config = None               # type: Any


class _Handler(BaseHTTPRequestHandler):
    """One request. Routing and status codes only -- no estimation here."""

    server_version = "pose-service/" + SERVICE_VERSION
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        service = self.server.service
        if route in ("/healthz", "/readyz"):
            health = (_starting(self.server.config) if service is None
                      else service.health())
            wanted = "loaded" if route == "/healthz" else "ready"
            self._json(200 if health[wanted] else 503, health)
        elif route == "/metrics":
            self._text(200, _render_metrics(service, self.server.config),
                       "text/plain; version=0.0.4; charset=utf-8")
        elif route == "/v1/estimate":
            self._error(405, "use POST")
        else:
            self._error(404, "no such route: %s" % route)

    def do_POST(self) -> None:
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route != "/v1/estimate":
            self._error(404, "no such route: %s" % route)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            # Refused without reading the body, so this connection still
            # holds a frame's worth of bytes that are not a request; a
            # keep-alive client would parse them as the next one.
            self.close_connection = True
            self._error(413, "body of %d bytes exceeds the %d byte limit"
                        % (length, MAX_BODY_BYTES))
            return
        service = self.server.service
        if service is None or not service.is_ready:
            self.close_connection = True        # body unread, as above
            self._json(503, {"error": "service is not ready",
                             "health": (_starting(self.server.config)
                                        if service is None
                                        else service.health())})
            return
        try:
            request = EstimateRequest.from_dict(self._read_json())
        except RequestError as exc:
            self._error(400, str(exc))
            return
        except ValueError as exc:
            self._error(400, "malformed JSON body: %s" % exc)
            return

        from .service import ServiceBusy      # noqa: E402 -- lazy by design
        started = time.perf_counter()
        try:
            result = _estimate(service, request)
        except ServiceBusy as exc:
            self._json(503, {"error": str(exc)})
            return
        except RequestError as exc:
            # Raised while unpacking the inline frame (base64), which can
            # only fail on something the caller sent.
            self._error(400, str(exc))
            return
        except Exception as exc:                    # noqa: BLE001
            # The service swallows frame failures itself, so anything
            # arriving here is a service bug, not a bad frame.
            _LOG.error("handler failed", extra={"fields": {
                "error": "%s: %s" % (type(exc).__name__, exc)}})
            self._error(500, "internal error")
            return

        payload = result.to_dict()
        _LOG.info("estimate", extra={"fields": {
            "scene_id": result.scene_id, "gate": result.gate,
            "poses": len(result.poses), "n_proposals": result.n_proposals,
            "score": result.best.score if result.best else 0.0,
            "timings_ms": result.timings_ms, "error": result.error,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 1)}})
        # 422 rather than 500: the request was well formed, the frame was
        # not usable. A cell retries the frame; it does not restart us.
        self._json(200 if result.error is None else 422, payload)

    # -- plumbing --------------------------------------------------------

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise RequestError("empty body; Content-Length is required")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        self._text(status, json.dumps(payload) + "\n", "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message,
                            "schema_version": SCHEMA_VERSION})

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


def _estimate(service: Any, request: EstimateRequest) -> Any:
    """Hand one parsed request to the estimator."""
    if not request.is_inline:
        return service.estimate_scene_dir(request.scene_dir)
    try:
        rgb_png = base64.b64decode(request.rgb_png_b64, validate=True)
        depth_png = base64.b64decode(request.depth_png_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError("rgb_png_b64/depth_png_b64 is not base64: %s" % exc)
    return service.estimate_png_frame(rgb_png, depth_png, request.K,
                                      request.depth_scale,
                                      scene_id=request.scene_id)


def _starting(config: ServiceConfig) -> Dict[str, Any]:
    """Health for the window between binding the socket and having weights.

    A probe during the estimator's import gets this rather than a hung
    connection: importing torch and Open3D is half a minute on an A57, and
    a cell, an install script and a container probe all need an answer.
    """
    return {"status": "starting", "loaded": False, "ready": False,
            "service_version": SERVICE_VERSION,
            "config_digest": config.digest(), "uptime_s": 0.0, "memory": {}}


def _render_metrics(service: Any, config: ServiceConfig) -> str:
    """Prometheus text exposition, format version 0.0.4."""
    if service is None:
        # Scraped during startup: the two series a monitoring system needs
        # to see a board that is up but not yet picking.
        return ('# HELP pose_build_info Service version and configuration '
                'digest.\n# TYPE pose_build_info gauge\n'
                'pose_build_info{version="%s",schema="%s",config_digest='
                '"%s"} 1\n# HELP pose_ready Models loaded and warmed.\n'
                '# TYPE pose_ready gauge\npose_ready 0\n'
                % (SERVICE_VERSION, SCHEMA_VERSION, config.digest()))
    stats = service.stats()
    health = service.health()
    latency = stats["latency_ms"]
    lines = [
        "# HELP pose_build_info Service version and configuration digest.",
        "# TYPE pose_build_info gauge",
        'pose_build_info{version="%s",schema="%s",config_digest="%s"} 1'
        % (SERVICE_VERSION, SCHEMA_VERSION, health["config_digest"]),
        "# HELP pose_ready Models loaded and warmed.",
        "# TYPE pose_ready gauge",
        "pose_ready %d" % (1 if health["ready"] else 0),
        "# HELP pose_frames_total Frames the estimator completed.",
        "# TYPE pose_frames_total counter",
        "pose_frames_total %d" % stats["frames"],
        "# HELP pose_failures_total Frames that came back with an error.",
        "# TYPE pose_failures_total counter",
        "pose_failures_total %d" % stats["failures"],
        "# HELP pose_rejected_total Requests refused with no free slot.",
        "# TYPE pose_rejected_total counter",
        "pose_rejected_total %d" % stats["rejected"],
        "# HELP pose_poses_total Poses returned across all frames.",
        "# TYPE pose_poses_total counter",
        "pose_poses_total %d" % stats["poses"],
        "# HELP pose_picks_total Frames whose top pose cleared the gate.",
        "# TYPE pose_picks_total counter",
        "pose_picks_total %d" % stats["picks"],
        "# HELP pose_last_score Score of the last frame's top pose.",
        "# TYPE pose_last_score gauge",
        "pose_last_score %.4f" % stats["last_score"],
        "# HELP pose_rss_mb Resident memory of the service process.",
        "# TYPE pose_rss_mb gauge",
        "pose_rss_mb %.1f" % health["memory"]["rss_mb"],
        "# HELP pose_latency_ms Frame latency; quantiles over a bounded "
        "window of recent frames.",
        "# TYPE pose_latency_ms summary",
        'pose_latency_ms{quantile="0.5"} %.1f' % latency["p50"],
        'pose_latency_ms{quantile="0.95"} %.1f' % latency["p95"],
        "pose_latency_ms_sum %.1f" % latency["sum"],
        "pose_latency_ms_count %d" % stats["frames"],
    ]
    return "\n".join(lines) + "\n"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, on stderr, for a log collector to read."""

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
