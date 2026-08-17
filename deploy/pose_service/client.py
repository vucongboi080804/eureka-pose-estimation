"""Command-line client for the pose service.

The commissioning tool: it is what an engineer runs on the board to check
that the cell would pick, and what a shell script or a CI job calls to
assert that it still does. Standard library only, so it runs in the
board's bare Python without the estimator's environment.

    python -m deploy.pose_service.client estimate --scene test/000001
    python -m deploy.pose_service.client estimate --scene test/000001 --inline
    python -m deploy.pose_service.client health
    python -m deploy.pose_service.client metrics
    python -m deploy.pose_service.client bench-frame --scene test/000001 -n 10

``--scene`` names a folder the *service* can read; ``--inline`` sends the
frame by value instead, which is the path a camera on another host takes.
``--rgb/--depth/--camera`` do the same from loose files.

Exit codes: 0 the top pose cleared the accept gate, 2 the service says
rescan, 1 the request failed. A cell script can branch on those without
parsing anything.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .schema import GATE_PICK, EstimateRequest, FrameResult

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_RESCAN = 2


#: Where the service listens when nothing says otherwise.
DEFAULT_URL = "http://127.0.0.1:8080"


def main(argv: Optional[list] = None) -> int:
    # The global options are defined once and attached to the top level
    # and to every subcommand, so both orderings a person will actually
    # type work: "client --json estimate ..." and "client estimate
    # ... --json". SUPPRESS keeps whichever position was used from being
    # overwritten by the other position's default.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help="Base URL of the service (default: %s)"
                             % DEFAULT_URL)
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="Client-side watchdog, seconds (default: 120)")
    common.add_argument("--json", action="store_true",
                        default=argparse.SUPPRESS,
                        help="Print the raw response for a machine")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", parents=[common],
                              help="Estimate poses for one frame")
    _add_frame_arguments(estimate)
    bench = sub.add_parser("bench-frame", parents=[common],
                           help="Repeat one frame and report latencies")
    _add_frame_arguments(bench)
    bench.add_argument("-n", "--repeat", type=int, default=10)
    sub.add_parser("health", parents=[common],
                   help="Liveness, readiness and memory")
    sub.add_parser("metrics", parents=[common],
                   help="Prometheus exposition, as served")

    args = parser.parse_args(argv)
    # The defaults are applied here rather than through set_defaults(),
    # which would write them onto the Action objects the subcommands
    # share with the top level and so undo whichever position was used.
    for name, fallback in (("url", DEFAULT_URL), ("timeout", 120.0),
                           ("json", False)):
        if not hasattr(args, name):
            setattr(args, name, fallback)
    try:
        if args.command == "estimate":
            return _estimate(args)
        if args.command == "bench-frame":
            return _bench(args)
        if args.command == "health":
            return _health(args)
        return _metrics(args)
    except _ClientError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_FAILED


def _add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-dir", "--scene", dest="scene_dir",
                        default=None,
                        help="Scene folder with rgb.png, depth.png and "
                             "camera.json, read by the service")
    parser.add_argument("--inline", action="store_true",
                        help="Send the frame by value (base64 PNGs) instead "
                             "of by path")
    parser.add_argument("--rgb", default=None, help="Colour PNG (implies "
                                                    "--inline)")
    parser.add_argument("--depth", default=None, help="Depth PNG")
    parser.add_argument("--camera", default=None,
                        help="camera.json holding K and depth_scale")
    parser.add_argument("--scene-id", default=None,
                        help="Label echoed back in the response")


# -- commands ------------------------------------------------------------

def _estimate(args: argparse.Namespace) -> int:
    result = FrameResult.from_dict(
        _post(args, "/v1/estimate", _frame_request(args).to_dict()))
    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        _print_frame(result)
    if result.error is not None:
        return EXIT_FAILED
    return EXIT_OK if result.gate == GATE_PICK else EXIT_RESCAN


def _bench(args: argparse.Namespace) -> int:
    payload = _frame_request(args).to_dict()
    latencies = []      # type: List[float]
    stages = {}         # type: Dict[str, float]
    poses = 0
    for _ in range(max(1, args.repeat)):
        started = time.perf_counter()
        result = FrameResult.from_dict(_post(args, "/v1/estimate", payload))
        latencies.append((time.perf_counter() - started) * 1000.0)
        poses += len(result.poses)
        for stage, value in result.timings_ms.items():
            stages[stage] = stages.get(stage, 0.0) + value
    ordered = sorted(latencies)
    report = {"repeat": len(latencies), "poses": poses,
              "round_trip_ms": {"p50": _percentile(ordered, 0.50),
                                "p90": _percentile(ordered, 0.90),
                                "p95": _percentile(ordered, 0.95),
                                "max": round(ordered[-1], 1),
                                "mean": round(sum(ordered) / len(ordered), 1)},
              "mean_stage_ms": {k: round(v / len(latencies), 1)
                                for k, v in sorted(stages.items())}}
    if args.json:
        print(json.dumps(report))
    else:
        trip = report["round_trip_ms"]
        print("%d frames, %d poses" % (report["repeat"], report["poses"]))
        print("round trip ms  p50 %.1f  p90 %.1f  p95 %.1f  max %.1f  "
              "mean %.1f" % (trip["p50"], trip["p90"], trip["p95"],
                             trip["max"], trip["mean"]))
        print("mean stage ms  " + "  ".join(
            "%s %.1f" % (k.replace("_ms", ""), v)
            for k, v in report["mean_stage_ms"].items()))
    return EXIT_OK


def _health(args: argparse.Namespace) -> int:
    health = _get_json(args, "/healthz")
    if args.json:
        print(json.dumps(health))
    else:
        memory = health.get("memory", {})
        print("%s  version %s  config %s  up %.0f s  rss %.0f MB "
              "(models +%.0f MB)"
              % (health["status"], health["service_version"],
                 health["config_digest"], health["uptime_s"],
                 memory.get("rss_mb", 0.0), memory.get("delta_mb", 0.0)))
    return EXIT_OK if health.get("ready") else EXIT_FAILED


def _metrics(args: argparse.Namespace) -> int:
    print(_get(args, "/metrics").decode("utf-8"), end="")
    return EXIT_OK


# -- request building ----------------------------------------------------

def _frame_request(args: argparse.Namespace) -> EstimateRequest:
    """Turn the CLI's frame arguments into one request."""
    inline = args.inline or args.rgb is not None
    if args.scene_dir is None and not inline:
        raise _ClientError("give --scene, or --rgb/--depth/--camera")
    if not inline:
        return EstimateRequest(scene_dir=os.path.abspath(args.scene_dir),
                               scene_id=args.scene_id)
    rgb_path = args.rgb or _in_scene(args, "rgb.png")
    depth_path = args.depth or _in_scene(args, "depth.png")
    camera_path = args.camera or _in_scene(args, "camera.json")
    camera = json.loads(_read(camera_path).decode("utf-8"))
    scene_id = args.scene_id
    if scene_id is None and args.scene_dir:
        scene_id = os.path.basename(os.path.normpath(args.scene_dir))
    return EstimateRequest(
        rgb_png_b64=base64.b64encode(_read(rgb_path)).decode("ascii"),
        depth_png_b64=base64.b64encode(_read(depth_path)).decode("ascii"),
        K=camera["K"], depth_scale=float(camera["depth_scale"]),
        scene_id=scene_id)


def _in_scene(args: argparse.Namespace, name: str) -> str:
    if not args.scene_dir:
        raise _ClientError("--inline needs --scene, or all of "
                           "--rgb/--depth/--camera")
    return os.path.join(args.scene_dir, name)


def _read(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise _ClientError(str(exc))


# -- transport -----------------------------------------------------------

class _ClientError(RuntimeError):
    """Anything that stops the client from getting an answer."""


def _post(args: argparse.Namespace, route: str,
          payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url.rstrip("/") + route, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    return json.loads(_send(request, args.timeout).decode("utf-8"))


def _get(args: argparse.Namespace, route: str) -> bytes:
    return _send(urllib.request.Request(args.url.rstrip("/") + route),
                 args.timeout)


def _get_json(args: argparse.Namespace, route: str) -> Dict[str, Any]:
    return json.loads(_get(args, route).decode("utf-8"))


def _send(request: urllib.request.Request, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except socket.timeout:
        # urlopen raises this bare, not wrapped in URLError, when the read
        # runs out of time -- and a frame on a cold or emulated board can
        # legitimately outlast the default. Say which number to raise
        # instead of unwinding urllib's stack at the operator.
        raise _ClientError("no answer from %s within %g s -- the frame may "
                           "still be estimating; raise --timeout"
                           % (request.full_url, timeout))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        # 503 and 422 are answers, not accidents: show what the service
        # said rather than a stack trace of urllib.
        raise _ClientError("HTTP %d from %s: %s"
                           % (exc.code, request.full_url, detail))
    except urllib.error.URLError as exc:
        raise _ClientError("cannot reach %s: %s" % (request.full_url,
                                                    exc.reason))


# -- printing ------------------------------------------------------------

def _print_frame(result: FrameResult) -> None:
    timings = result.timings_ms
    print("%s  gate=%s  %d pose(s) from %d proposals  %.0f ms"
          % (result.scene_id, result.gate, len(result.poses),
             result.n_proposals, timings.get("total_ms", 0.0)))
    print("  " + "  ".join("%s %.0f" % (k.replace("_ms", ""), v)
                           for k, v in sorted(timings.items())))
    if result.error is not None:
        print("  frame failed: %s" % result.error)
        return
    best = result.best
    if best is None:
        print("  nothing verified: rescan or shake the bin")
        return
    print("grasp pose (T_camera_object), score %.3f "
          "(segmenter %.3f x depth %.3f):"
          % (best.score, best.seg_confidence or 0.0,
             best.depth_verification or 0.0))
    for row in best.R:
        print("  [%8.5f %8.5f %8.5f]" % tuple(row))
    print("  t = [%.5f %.5f %.5f] m" % tuple(best.t))
    if result.gate != GATE_PICK:
        print("  below the accept gate: rescan rather than grab")
    print("config %s, service %s, schema %s"
          % (result.config_digest, result.service_version,
             result.schema_version))


def _percentile(ordered: List[float], q: float) -> float:
    rank = max(1, int(round(q * len(ordered))))
    return round(ordered[rank - 1], 1)


if __name__ == "__main__":
    sys.exit(main())
