"""The pick cycle as a program: camera, pose, grasp, decision, log.

This is the loop deploy/ARCHITECTURE.md draws, written out. One cycle is
five steps and nothing else:

    1. GET /v1/frame from the camera service
    2. POST /v1/estimate to the pose service -- the same base64 strings,
       moved, never decoded and re-encoded, so nothing between the sensor
       and the pose can resample the depth map
    3. plan grasps on the returned poses (deploy/cell/grasp.py)
    4. ask the policy what to do (deploy/cell/policy.py)
    5. emit one JSON line and one human line

    python -m deploy.cell.runner --camera http://127.0.0.1:8081 \
        --pose http://127.0.0.1:8080 --cycles 5 --out cycles.jsonl
    python -m deploy.cell.runner --once
    python -m deploy.cell.runner --hand-eye cell/hand_eye.json --cycles 0

An integrator reading this file should be able to write the same loop in
their own controller, in their own language, from it -- so it stays five
steps, and everything that is not one of the five steps is either error
handling or a field in the log line.

**What this program does not do.** It commands nothing. There is no
robot behind it, so a PICK is logged, not executed, and no pick outcome
is invented: ``--pick-outcome`` defaults to ``unknown``, the value
deploy/cell/drift.py reserves for a cell with no success sensor. Set it
to ``success``/``slip``/``miss`` on a bench to drive the policy's retry
and escalation branches deliberately. Drift is likewise reported as
``ok``: judging it needs the robot poses a pick returns, which this
program never has.

**Two clocks.** ``stage_ms`` is what the *loop* measured -- the camera
round trip, the local PNG decode, the estimate round trip, planning,
the policy. ``service_ms`` is the pose service's own split of the frame
it estimated. They are kept apart because ``estimate_ms`` minus
``service_ms.total`` is the transport, and confusing the two hides a
slow link inside a slow model.

Exit codes: 0 finished the requested cycles, 1 could not start, 2 stopped
at a policy FAULT under ``--stop-on-terminal``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from . import drift
from .calibration import CalibrationError, HandEye, MOUNT_FIXED
from .grasp import GraspCandidate, GraspConfigError, GraspPlanner, \
    default_grasps_path
from .policy import ACCEPT_SCORE, FAULT, PICK, RETRY, Action, CellFrame, \
    PickHistory, PickPolicy
from ..camera_service.client import CameraClient, CameraClientError
from ..camera_service.frame import Frame, estimate_body
from ..pose_service.schema import FrameResult

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_FAULT = 2

#: The loop's own health, kept apart from the policy's bin states. A
#: camera that will not answer is an infrastructure fault, not a bin that
#: cannot be picked from, and a cell that conflates the two shakes a tray
#: because a cable came loose.
LOOP_OK = "ok"
LOOP_CAMERA_DOWN = "camera_down"
LOOP_POSE_DOWN = "pose_down"
LOOP_BAD_FRAME = "bad_frame"

#: First wait after a service stops answering, seconds, doubling to
#: :data:`MAX_BACKOFF_S`. Half a second is short enough that a service
#: restarted by systemd is picked up almost immediately.
FIRST_BACKOFF_S = 0.5

#: Longest wait between retries. The pose service needs ~24 s to load its
#: weights on the board (results/bench/board_nano640.json, model_load_s),
#: so a cap of 8 s costs at most a third of that in extra idle time while
#: keeping a dead service down to eight requests a minute.
MAX_BACKOFF_S = 8.0

#: Client-side watchdog for one estimate, seconds. A frame takes 2.6-2.7 s
#: on the board (results/bench/board_nano640.json) and tens of seconds
#: when the segmenters find nothing and the geometric safety net runs, so
#: this is generous on purpose: an estimate that outlasts it means the
#: service is stuck, not slow.
DEFAULT_ESTIMATE_TIMEOUT_S = 120.0

#: Client-side watchdog for one frame, seconds. A frame is a disk read or
#: an exposure away; anything slower means the camera is stuck.
DEFAULT_FRAME_TIMEOUT_S = 30.0


class PoseUnreachable(RuntimeError):
    """The pose service did not answer, or did not answer with JSON."""


@dataclass
class CycleRecord:
    """Everything one cycle produced -- the JSON line, plus the pixels.

    The pixels (``frame``) and the planner's candidates are held for a
    caller rendering the cycle (deploy/demo/hud.py); they are not
    serialised. What :meth:`to_json` writes is what a cell keeps.
    """

    cycle: int
    #: Which bin this cycle belongs to. A policy terminal state ends a
    #: bin, so the counters in ``history`` are per bin, not per run.
    bin_index: int
    loop_state: str
    stage_ms: Dict[str, float] = field(default_factory=dict)
    frame: Optional[Frame] = None
    result: Optional[FrameResult] = None
    grasps: List[GraspCandidate] = field(default_factory=list)
    action: Optional[Action] = None
    #: What the cell's success sensor reported after a PICK/RETRY, or
    #: None when nothing was commanded.
    outcome: Optional[str] = None
    history: Dict[str, int] = field(default_factory=dict)
    #: Why the loop could not complete this cycle, and how long it waited
    #: before the next one.
    error: Optional[str] = None
    backoff_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def scene_id(self) -> str:
        if self.result is not None:
            return self.result.scene_id
        return "" if self.frame is None else self.frame.source_name

    @property
    def chosen(self) -> Optional[GraspCandidate]:
        return None if self.action is None else self.action.grasp

    def to_json(self) -> Dict[str, Any]:
        """One line of the run log.

        Carries the poses and the intrinsics as well as the decision, so
        the cycle can be re-drawn later from the log and the frame it
        names, without running the pipeline again.
        """
        result, action = self.result, self.action
        best = None if result is None else result.best
        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                time.gmtime(self.timestamp)),
            "cycle": self.cycle, "bin": self.bin_index,
            "loop": self.loop_state,
            "frame": None if self.frame is None else _frame_json(self.frame),
            "scene": self.scene_id,
            "n_poses": 0 if result is None else len(result.poses),
            "n_proposals": 0 if result is None else result.n_proposals,
            "gate": None if result is None else result.gate,
            "top_score": None if best is None else round(best.score, 4),
            "state": None if action is None else action.state,
            "reason": None if action is None else action.reason,
            "drift": None if action is None else action.drift_verdict,
            "outcome": self.outcome,
            "grasp": None if self.chosen is None else self.chosen.to_dict(),
            "poses": [] if result is None else [p.to_dict()
                                                for p in result.poses],
            "stage_ms": {k: round(v, 1) for k, v in self.stage_ms.items()},
            "service_ms": {} if result is None else dict(result.timings_ms),
            "config_digest": None if result is None else result.config_digest,
            "history": dict(self.history),
            "error": self.error,
            "backoff_s": round(self.backoff_s, 2) or None,
        }

    def summary(self) -> str:
        """The one line a person watching the cell reads."""
        head = "cycle %4d  bin %d" % (self.cycle, self.bin_index)
        if self.loop_state != LOOP_OK:
            return "%s  %-11s %s%s" % (
                head, self.loop_state.upper().replace("_", " "), self.error,
                "" if not self.backoff_s else
                "  retry in %.1f s" % self.backoff_s)
        result, action = self.result, self.action
        best = result.best if result is not None else None
        chosen = self.chosen
        return "%s  frame %-5d %-10s %d pose/%-2d prop  top %-5s gate %-6s " \
               "%-5s %-34s %5.0f ms" % (
                   head, self.frame.frame_id, self.scene_id[:10],
                   len(result.poses), result.n_proposals,
                   "-" if best is None else "%.3f" % best.score, result.gate,
                   action.state.upper(),
                   "-" if chosen is None else
                   "%s pose %d rank %.2f" % (chosen.grasp_name,
                                             chosen.pose_index, chosen.rank),
                   self.stage_ms.get("cycle_ms", 0.0))


class CellLoop:
    """One camera, one pose service, one planner, one policy, one bin.

    Holds the counters between cycles and nothing else: every cycle is a
    fresh pair of HTTP requests, so a service restarted underneath the
    loop is picked up on the next cycle rather than needing this object
    rebuilt.
    """

    def __init__(self, camera_url: str, pose_url: str,
                 planner: GraspPlanner,
                 policy: Optional[PickPolicy] = None,
                 hand_eye: Optional[Any] = None,
                 pick_outcome: str = drift.UNKNOWN,
                 frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
                 estimate_timeout_s: float = DEFAULT_ESTIMATE_TIMEOUT_S):
        """Args:
            camera_url: Base URL of the camera service.
            pose_url: Base URL of the pose service.
            planner: Grasp planner for the part being picked.
            policy: The pick policy; the shipped defaults if None.
            hand_eye: ``T_base_camera`` as a
                :class:`~deploy.cell.frames.FramedTransform`, or None to
                stay in the camera frame.
            pick_outcome: What to report to the policy after a commanded
                pick. ``unknown`` (the default) is a cell with no success
                sensor -- which is what this program is.
        """
        if pick_outcome not in drift.OUTCOMES:
            raise ValueError("pick_outcome must be one of %s, got %r"
                             % (drift.OUTCOMES, pick_outcome))
        self.camera = CameraClient(camera_url, frame_timeout_s)
        self.pose_url = pose_url.rstrip("/")
        self.estimate_timeout_s = float(estimate_timeout_s)
        self.planner = planner
        self.policy = policy or PickPolicy()
        self.hand_eye = hand_eye
        self.pick_outcome = pick_outcome
        self.history = PickHistory()
        self.bin_index = 1
        self.cycle = 0
        self._backoff_s = 0.0

    # -- the cycle -------------------------------------------------------

    def run_cycle(self) -> CycleRecord:
        """One frame, one decision. Never raises on a service or a frame.

        Anything the loop cannot get past -- an unreachable service, a
        frame that will not decode -- comes back as a record with
        ``loop_state`` set and a wait already served, so the caller's loop
        body is unconditional.
        """
        self.cycle += 1
        stage = {}                      # type: Dict[str, float]
        started = time.perf_counter()

        mark = time.perf_counter()
        try:
            wire = self.camera.frame_wire()
        except (CameraClientError, OSError) as exc:
            # The client wraps urllib's errors and names the endpoint in
            # them, but a service killed *during* a request resets the
            # socket and that arrives as a bare OSError. This loop does
            # not stop, so it catches whatever the client lets through
            # and names the endpoint itself.
            return self._down(LOOP_CAMERA_DOWN,
                              str(exc) if isinstance(exc, CameraClientError)
                              else "%s from %s/v1/frame: %s"
                              % (type(exc).__name__, self.camera.url, exc),
                              stage, started)
        stage["camera_ms"] = _ms_since(mark)

        mark = time.perf_counter()
        try:
            frame = Frame.from_wire(wire)
        except ValueError as exc:
            # One unreadable frame is a frame, not an outage: no backoff,
            # the next one is very likely fine.
            return self._down(LOOP_BAD_FRAME, str(exc), stage, started,
                              wait=False)
        stage["decode_ms"] = _ms_since(mark)

        mark = time.perf_counter()
        try:
            payload = _post_json(self.pose_url, "/v1/estimate",
                                 estimate_body(wire, _scene_label(wire)),
                                 self.estimate_timeout_s)
            result = FrameResult.from_dict(payload)
        except (PoseUnreachable, KeyError, TypeError, ValueError) as exc:
            return self._down(LOOP_POSE_DOWN, str(exc), stage, started,
                              frame=frame)
        stage["estimate_ms"] = _ms_since(mark)

        mark = time.perf_counter()
        grasps = self.planner.plan(
            result.poses, frame.depth_raw.astype(np.float64)
            * frame.depth_scale, T_base_camera=self.hand_eye, K=frame.K)
        stage["plan_ms"] = _ms_since(mark)

        mark = time.perf_counter()
        cell_frame = CellFrame.from_service(result, grasps,
                                            self.planner.last_rejections)
        action = self.policy.next_action(cell_frame, self.history)
        self.history.record(action)
        outcome = None
        if action.state in (PICK, RETRY):
            # No robot ran the grasp, so nothing is claimed about it: the
            # default outcome is drift.UNKNOWN, which the policy folds in
            # as neither a success nor a failure.
            outcome = self.pick_outcome
            self.history.report_outcome(outcome)
        stage["policy_ms"] = _ms_since(mark)
        stage["cycle_ms"] = _ms_since(started)

        self._backoff_s = 0.0
        return CycleRecord(cycle=self.cycle, bin_index=self.bin_index,
                           loop_state=LOOP_OK, stage_ms=stage, frame=frame,
                           result=result, grasps=list(grasps), action=action,
                           outcome=outcome, history=self._history_json(),
                           error=result.error)

    def next_bin(self) -> None:
        """Start a fresh bin: new counters, same calibration and policy.

        What an operator does after an EMPTY (swap the tray) or a FAULT
        (clear it). The terminal state stays in the log; only the counters
        that led to it are reset.
        """
        self.bin_index += 1
        self.history = PickHistory()

    def run(self, cycles: int = 0,
            stop_on_terminal: bool = False) -> Iterator[CycleRecord]:
        """Yield cycles until ``cycles`` are done (0 = until stopped)."""
        while cycles <= 0 or self.cycle < cycles:
            record = self.run_cycle()
            yield record
            if record.action is not None and record.action.is_terminal:
                if stop_on_terminal:
                    return
                self.next_bin()

    # -- failure ---------------------------------------------------------

    def _down(self, loop_state: str, error: str, stage: Dict[str, float],
              started: float, wait: bool = True,
              frame: Optional[Frame] = None) -> CycleRecord:
        """Record a cycle the loop could not finish, and serve the wait.

        The wait happens here rather than in the caller so that every
        driver of this class -- the CLI, the renderer, a controller --
        backs off the same way without being asked to remember to.
        """
        backoff = 0.0
        if wait:
            backoff = min(MAX_BACKOFF_S,
                          self._backoff_s * 2.0 or FIRST_BACKOFF_S)
            self._backoff_s = backoff
        stage["cycle_ms"] = _ms_since(started)
        record = CycleRecord(cycle=self.cycle, bin_index=self.bin_index,
                             loop_state=loop_state, stage_ms=stage,
                             frame=frame, history=self._history_json(),
                             error=error, backoff_s=backoff)
        if backoff:
            time.sleep(backoff)
        return record

    def _history_json(self) -> Dict[str, int]:
        history = self.history
        return {"picks": history.picks, "successes": history.successes,
                "consecutive_failures": history.consecutive_failures,
                "rescans_since_progress": history.rescans_since_progress,
                "shakes_since_progress": history.shakes_since_progress,
                "consecutive_barren": history.consecutive_barren,
                "consecutive_no_proposals": history.consecutive_no_proposals}


# -- transport -----------------------------------------------------------

def _post_json(url: str, route: str, payload: Dict[str, Any],
               timeout: float) -> Dict[str, Any]:
    """POST a JSON body and parse the answer, naming what went wrong.

    urllib's own exceptions name a socket; a cell log has to name the
    service, because "connection refused" three levels into a traceback
    is the same line for a camera and for an estimator.
    """
    request = urllib.request.Request(
        url + route, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 400, 422 and 503 are answers: show what the service said.
        detail = exc.read().decode("utf-8", "replace").strip()
        raise PoseUnreachable("HTTP %d from %s: %s"
                              % (exc.code, request.full_url, detail[:200]))
    except socket.timeout:
        # Raised bare, not wrapped in URLError, when the read runs out.
        raise PoseUnreachable("no answer from %s within %g s -- the frame "
                              "may still be estimating; raise "
                              "--estimate-timeout"
                              % (request.full_url, timeout))
    except urllib.error.URLError as exc:
        raise PoseUnreachable("cannot reach %s: %s"
                              % (request.full_url, exc.reason))
    except OSError as exc:
        # A service killed mid-request resets the socket; urllib does not
        # wrap that, and the loop must not end on it.
        raise PoseUnreachable("%s from %s: %s"
                              % (type(exc).__name__, request.full_url, exc))
    except ValueError as exc:
        raise PoseUnreachable("%s did not return JSON: %s" % (route, exc))


def _scene_label(wire: Dict[str, Any]) -> str:
    """A short, stable name for the frame, for logs and results.

    A replay source names itself ``scene_folder:test/000003``, whose last
    component is the scene id the rest of this repository uses. A live
    camera names itself by serial, which is the same for every frame, so
    the frame id is appended to keep the label unique.
    """
    source = str(wire.get("source", "camera"))
    if "/" in source:
        return source.rsplit("/", 1)[-1]
    return "%s#%d" % (source, int(wire.get("frame_id", 0)))


def _frame_json(frame: Frame) -> Dict[str, Any]:
    """What the log keeps about the frame itself -- everything needed to
    find the pixels again and to re-project a pose onto them."""
    return {"frame_id": frame.frame_id, "source": frame.source_name,
            "timestamp_ns": frame.timestamp_ns,
            "width": frame.width, "height": frame.height,
            "depth_valid": round(frame.valid_depth_fraction(), 4),
            "K": [[float(v) for v in row] for row in frame.K],
            "depth_scale": frame.depth_scale}


def _ms_since(mark: float) -> float:
    return (time.perf_counter() - mark) * 1000.0


# -- entry point ---------------------------------------------------------

def build_planner(model_path: str, grasps_path: str) -> GraspPlanner:
    """The planner, or a fatal error naming the file that is wrong."""
    return GraspPlanner(model_path, grasps_path)


def load_hand_eye(path: str) -> Any:
    """``T_base_camera`` from a calibration file, named and validated.

    A wrist camera is refused rather than defaulted: its
    ``T_base_camera`` exists only for the flange pose the frame was
    captured at, and this loop never sees the robot.
    """
    hand_eye = HandEye.load(path)
    if hand_eye.mount != MOUNT_FIXED:
        raise CalibrationError(
            "this loop has no robot pose to compose with, so it can only "
            "use a fixed-mount calibration; %s declares mount %r"
            % (path, hand_eye.mount))
    return hand_eye.T_base_camera()


def main(argv: Optional[list] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", default="http://127.0.0.1:8081",
                        help="Camera service base URL (default: %(default)s)")
    parser.add_argument("--pose", default="http://127.0.0.1:8080",
                        help="Pose service base URL (default: %(default)s)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Cycles to run; 0 runs until interrupted")
    parser.add_argument("--once", action="store_true",
                        help="One cycle, then exit -- the smoke test")
    parser.add_argument("--out", default=None,
                        help="Append one JSON object per cycle to this file")
    parser.add_argument("--model", default=os.path.join(root, "model",
                                                        "3d_model.ply"),
                        help="CAD in the object frame (default: %(default)s)")
    parser.add_argument("--grasps", default=default_grasps_path(),
                        help="Grasp definition file (default: %(default)s)")
    parser.add_argument("--hand-eye", default=None,
                        help="Calibration file; without it the chosen grasp "
                             "is reported in the camera frame only")
    parser.add_argument("--accept-score", type=float, default=ACCEPT_SCORE,
                        help="The cell's own gate (default: %(default)s)")
    parser.add_argument("--pick-outcome", default=drift.UNKNOWN,
                        choices=list(drift.OUTCOMES),
                        help="What to report after a commanded pick. "
                             "Nothing is executed, so the honest default "
                             "is %(default)s")
    parser.add_argument("--stop-on-terminal", action="store_true",
                        help="Exit when the policy reaches EMPTY or FAULT "
                             "instead of starting the next bin")
    parser.add_argument("--frame-timeout", type=float,
                        default=DEFAULT_FRAME_TIMEOUT_S)
    parser.add_argument("--estimate-timeout", type=float,
                        default=DEFAULT_ESTIMATE_TIMEOUT_S)
    parser.add_argument("--quiet", action="store_true",
                        help="Write the JSON lines only")
    args = parser.parse_args(argv)

    # Everything that can be wrong about the configuration is wrong now,
    # before the first frame: a cell must not come up half-loaded.
    try:
        planner = build_planner(args.model, args.grasps)
        hand_eye = load_hand_eye(args.hand_eye) if args.hand_eye else None
    except (GraspConfigError, CalibrationError) as exc:
        print("cannot start: %s" % exc, file=sys.stderr)
        return EXIT_FAILED

    loop = CellLoop(args.camera, args.pose, planner,
                    PickPolicy(accept_score=args.accept_score),
                    hand_eye=hand_eye, pick_outcome=args.pick_outcome,
                    frame_timeout_s=args.frame_timeout,
                    estimate_timeout_s=args.estimate_timeout)
    if not args.quiet:
        print("cell loop: camera %s, pose %s, %d grasp(s) on %s, gate %.2f%s"
              % (args.camera, args.pose, len(planner.grasps),
                 os.path.basename(args.model), args.accept_score,
                 "" if hand_eye is None else ", hand-eye %s" % args.hand_eye))

    cycles = 1 if args.once else args.cycles
    sink = open(args.out, "a") if args.out else None
    faulted, picks, estimated = False, 0, 0
    try:
        for record in loop.run(cycles, stop_on_terminal=args.stop_on_terminal):
            if sink is not None:
                sink.write(json.dumps(record.to_json()) + "\n")
                sink.flush()        # a log read while the cell runs
            if not args.quiet:
                # Flushed: stdout to a pipe or a journal is block
                # buffered, and a cell console that shows nothing for a
                # minute is a cell console nobody watches.
                print(record.summary(), flush=True)
            if record.loop_state == LOOP_OK:
                estimated += 1
            if record.action is not None and record.action.state in (PICK,
                                                                     RETRY):
                picks += 1
            if record.action is not None and record.action.is_terminal:
                faulted = record.action.state == FAULT
                if not args.quiet:
                    print("  bin %d ended: %s" % (record.bin_index,
                                                  record.action.reason))
    except KeyboardInterrupt:
        if not args.quiet:
            print("\ninterrupted")
    finally:
        if sink is not None:
            sink.close()
    if not args.quiet:
        print("%d cycle(s) (%d estimated), %d bin(s), %d pick(s) commanded"
              % (loop.cycle, estimated, loop.bin_index, picks))
    if faulted and args.stop_on_terminal:
        return EXIT_FAULT
    # A run in which no cycle ever reached the estimator did not pass: a
    # smoke test that returns 0 with the camera unplugged is a smoke test
    # nobody can gate an install on.
    return EXIT_OK if estimated else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
