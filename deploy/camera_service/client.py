"""Command-line client for the camera service.

The commissioning tool: what an engineer runs on the board to see that
the camera is delivering before wondering why the cell will not pick, and
what a shell script or a CI job calls to assert that it still is.

    python -m deploy.camera_service.client info
    python -m deploy.camera_service.client grab --out /tmp/capture
    python -m deploy.camera_service.client stream --n 20

``grab`` writes the frame as ``rgb.png``, ``depth.png`` and
``camera.json`` -- the release layout -- so the folder it produces is one
the pose service can be pointed straight at:

    python -m deploy.pose_service.client estimate --scene /tmp/capture

Standard library only at import time, so this runs in the board's bare
Python: the PNG bytes are written exactly as they arrived, and NumPy and
OpenCV are imported only by :meth:`CameraClient.frame`, which is the one
call that hands back pixels.

Exit codes: 0 the request succeeded, 1 it did not. A cell script can
branch on those without parsing anything.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

EXIT_OK = 0
EXIT_FAILED = 1

#: Where the camera service listens when nothing says otherwise; 8080 is
#: the pose service's.
DEFAULT_URL = "http://127.0.0.1:8081"

#: Client-side watchdog, seconds. A frame is a disk read or a sensor
#: exposure away, so anything slower than this means the camera is stuck,
#: and a cell would rather be told than wait.
DEFAULT_TIMEOUT_S = 30.0


class CameraClientError(RuntimeError):
    """Anything that stops the client from getting an answer."""


class CameraClient:
    """The camera service's endpoints, as methods.

    Holds no connection and no state: each call is one request, so a cell
    can keep an instance around for a shift without owning a socket that
    might have gone stale between picks.
    """

    def __init__(self, url: str = DEFAULT_URL,
                 timeout: float = DEFAULT_TIMEOUT_S):
        self.url = url.rstrip("/")
        self.timeout = float(timeout)

    def frame_wire(self) -> Dict[str, Any]:
        """The next frame as served: base64 PNGs, ``K``, ``depth_scale``.

        The form to forward to the pose service -- see
        :func:`deploy.camera_service.frame.estimate_body` -- because
        moving the strings across is the only way to be certain nothing
        resampled the depth map on the way.
        """
        return self._get_json("/v1/frame")

    def frame(self) -> Any:
        """The next frame decoded into a :class:`~.frame.Frame`.

        Imports OpenCV and NumPy on first use: a client that only moves
        bytes around should not need the imaging stack installed.
        """
        from .frame import Frame
        return Frame.from_wire(self.frame_wire())

    def intrinsics(self) -> Dict[str, Any]:
        """``K``, ``depth_scale`` and the image size, taking no frame."""
        return self._get_json("/v1/intrinsics")

    def health(self) -> Dict[str, Any]:
        """Liveness, readiness, counters and what the source is doing."""
        return self._get_json("/healthz")

    def metrics(self) -> str:
        """The Prometheus exposition, as served."""
        return self._get("/metrics").decode("utf-8")

    # -- transport -------------------------------------------------------

    def _get(self, route: str) -> bytes:
        request = urllib.request.Request(self.url + route)
        try:
            with urllib.request.urlopen(request,
                                        timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            # 503 and 410 are answers, not accidents: show what the
            # service said rather than a stack trace of urllib.
            raise CameraClientError("HTTP %d from %s: %s"
                                    % (exc.code, request.full_url, detail))
        except urllib.error.URLError as exc:
            raise CameraClientError("cannot reach %s: %s"
                                    % (request.full_url, exc.reason))

    def _get_json(self, route: str) -> Dict[str, Any]:
        try:
            return json.loads(self._get(route).decode("utf-8"))
        except ValueError as exc:
            raise CameraClientError("%s did not return JSON: %s"
                                    % (route, exc))


def main(argv: Optional[list] = None) -> int:
    # The global options are defined once and attached to the top level
    # and to every subcommand, so both orderings a person will actually
    # type work: "client --json info" and "client info --json".
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help="Base URL of the service (default: %s)"
                             % DEFAULT_URL)
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="Client-side watchdog, seconds (default: %g)"
                             % DEFAULT_TIMEOUT_S)
    common.add_argument("--json", action="store_true",
                        default=argparse.SUPPRESS,
                        help="Print the raw response for a machine")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    grab = sub.add_parser("grab", parents=[common],
                          help="Save the next frame as a scene folder")
    grab.add_argument("--out", default="capture",
                      help="Directory to write rgb.png, depth.png and "
                           "camera.json into (default: %(default)s)")
    sub.add_parser("info", parents=[common],
                   help="Intrinsics, image size and what the source is doing")
    stream = sub.add_parser("stream", parents=[common],
                            help="Pull N frames and report the latencies")
    stream.add_argument("-n", "--n", dest="count", type=int, default=10,
                        help="Frames to pull (default: %(default)s)")

    args = parser.parse_args(argv)
    # Applied here rather than through set_defaults(), which would write
    # them onto the Action objects the subcommands share with the top
    # level and so undo whichever position was actually used.
    for name, fallback in (("url", DEFAULT_URL),
                           ("timeout", DEFAULT_TIMEOUT_S), ("json", False)):
        if not hasattr(args, name):
            setattr(args, name, fallback)

    client = CameraClient(args.url, args.timeout)
    try:
        if args.command == "grab":
            return _grab(client, args)
        if args.command == "info":
            return _info(client, args)
        return _stream(client, args)
    except CameraClientError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_FAILED


# -- commands ------------------------------------------------------------

def _grab(client: CameraClient, args: argparse.Namespace) -> int:
    """Write one frame to disk in the layout the release folders use."""
    wire = client.frame_wire()
    rgb_png = base64.b64decode(wire["rgb_png_b64"])
    depth_png = base64.b64decode(wire["depth_png_b64"])
    width, height = _png_size(rgb_png)
    camera = {"K": wire["K"], "width": width, "height": height,
              "depth_scale": wire["depth_scale"],
              # Provenance: which camera, which frame, when. A capture
              # that later turns out to be wrong has to be traceable to
              # the moment it was taken.
              "source": wire.get("source"),
              "frame_id": wire.get("frame_id"),
              "timestamp_ns": wire.get("timestamp_ns"),
              "note": "Z_metres = depth.png value * depth_scale."}
    try:
        os.makedirs(args.out, exist_ok=True)
        # The bytes are written exactly as received: re-encoding a depth
        # PNG here would put a second codec between the sensor and the
        # pose, for no gain.
        _write(os.path.join(args.out, "rgb.png"), rgb_png)
        _write(os.path.join(args.out, "depth.png"), depth_png)
        _write(os.path.join(args.out, "camera.json"),
               (json.dumps(camera, indent=1) + "\n").encode("utf-8"))
    except OSError as exc:
        raise CameraClientError("cannot write into %s: %s" % (args.out, exc))
    if args.json:
        print(json.dumps({"out": os.path.abspath(args.out),
                          "frame_id": wire.get("frame_id"),
                          "source": wire.get("source"),
                          "width": width, "height": height}))
    else:
        print("frame %s from %s  %dx%d  ->  %s"
              % (wire.get("frame_id"), wire.get("source"), width, height,
                 os.path.abspath(args.out)))
        print("  rgb.png %d B  depth.png %d B  camera.json"
              % (len(rgb_png), len(depth_png)))
    return EXIT_OK


def _info(client: CameraClient, args: argparse.Namespace) -> int:
    """What the camera is and whether it is delivering."""
    health = client.health()
    intrinsics = client.intrinsics() if health.get("ready") else None
    if args.json:
        print(json.dumps({"health": health, "intrinsics": intrinsics}))
        return EXIT_OK if health.get("ready") else EXIT_FAILED
    source = health.get("source", {})
    print("%s  version %s  up %.0f s  %d frame(s), %d read error(s)"
          % (health.get("status"), health.get("camera_version"),
             health.get("uptime_s", 0.0), health.get("frames", 0),
             health.get("read_errors", 0)))
    print("source %s  %s" % (source.get("kind"), _describe_source(source)))
    if intrinsics is None:
        print("no frame read yet: the camera is not ready")
        return EXIT_FAILED
    print("%dx%d  depth_scale %g  (from frame %s, %s)"
          % (intrinsics["width"], intrinsics["height"],
             intrinsics["depth_scale"], intrinsics["frame_id"],
             intrinsics["source"]))
    for row in intrinsics["K"]:
        print("  [%10.4f %10.4f %10.4f]" % tuple(row))
    return EXIT_OK


def _stream(client: CameraClient, args: argparse.Namespace) -> int:
    """Pull frames back to back and report what the link cost."""
    latencies = []      # type: List[float]
    frames = []         # type: List[Dict[str, Any]]
    for _ in range(max(1, args.count)):
        started = time.perf_counter()
        wire = client.frame_wire()
        latencies.append((time.perf_counter() - started) * 1000.0)
        frames.append({"frame_id": wire.get("frame_id"),
                       "source": wire.get("source"),
                       "bytes": len(wire.get("rgb_png_b64", ""))
                       + len(wire.get("depth_png_b64", ""))})
    ordered = sorted(latencies)
    report = {"frames": len(frames),
              "frame_ids": [f["frame_id"] for f in frames],
              "sources": [f["source"] for f in frames],
              "round_trip_ms": {"p50": _percentile(ordered, 0.50),
                                "p90": _percentile(ordered, 0.90),
                                "p95": _percentile(ordered, 0.95),
                                "max": round(ordered[-1], 1),
                                "mean": round(sum(ordered) / len(ordered), 1)},
              "fps": round(1000.0 * len(latencies) / sum(latencies), 2)}
    if args.json:
        print(json.dumps(report))
        return EXIT_OK
    for frame, latency in zip(frames, latencies):
        print("frame %-6s %-32s %7.1f kB %8.1f ms"
              % (frame["frame_id"], frame["source"],
                 frame["bytes"] / 1024.0, latency))
    trip = report["round_trip_ms"]
    print("%d frames  round trip ms  p50 %.1f  p90 %.1f  p95 %.1f  max %.1f "
          " mean %.1f  (%.2f fps)"
          % (report["frames"], trip["p50"], trip["p90"], trip["p95"],
             trip["max"], trip["mean"], report["fps"]))
    return EXIT_OK


# -- helpers -------------------------------------------------------------

def _write(path: str, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)


def _png_size(png: bytes) -> Tuple[int, int]:
    """Width and height out of a PNG's IHDR, without a codec.

    Eight bytes of signature, then the IHDR chunk's length and type, then
    the two dimensions as big-endian 32-bit integers.
    """
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise CameraClientError("frame did not carry a PNG")
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


def _describe_source(source: Dict[str, Any]) -> str:
    """One line for whichever kind of source is running."""
    if source.get("kind") == "scene_folder":
        return ("%s/%s  %d scene(s), at %s, %d lap(s), loop %s"
                % (source.get("root"), source.get("split"),
                   source.get("scenes", 0), source.get("next_scene"),
                   source.get("laps", 0),
                   "on" if source.get("loop") else "off"))
    if source.get("kind") == "session":
        session = source.get("session") or {}
        return ("%s  %d frame(s) of %d, at %s, %d lap(s), loop %s"
                % (source.get("path"), source.get("scenes", 0),
                   session.get("frames", 0), source.get("next_scene"),
                   source.get("laps", 0),
                   "on" if source.get("loop") else "off"))
    return ("serial %s  %sx%s@%s" % (source.get("serial"),
                                     source.get("width"),
                                     source.get("height"), source.get("fps")))


def _percentile(ordered: List[float], q: float) -> float:
    if not ordered:
        return 0.0
    rank = max(1, int(round(q * len(ordered))))
    return round(ordered[rank - 1], 1)


if __name__ == "__main__":
    sys.exit(main())
