"""The pick cycle as an explicit state machine.

A cell's behaviour is usually implied: it lives in the order of a few
ifs in the controller, and nobody can say what happens on the third
consecutive miss without reading them all. This module writes it down
instead, so it can be reviewed, argued with, and changed in one place.

States and what causes each transition::

    SCAN   look at the bin
             viable grasp, no recent failure ......... PICK
             viable grasp, after a failure ........... RETRY
             poses verified but no grasp reachable ... SHAKE
             parts proposed but nothing verified ..... SCAN, then SHAKE,
                                                       then FAULT
             nothing proposed at all ................. SCAN, then SHAKE,
                                                       then EMPTY
             drift verdict = recalibrate ............. FAULT
    PICK   execute the top grasp; the cell reports the outcome
             success ................................. SCAN
             slip or miss ............................ SCAN (counter up)
    RETRY  same, on a grasp away from the one that just failed
    SHAKE  disturb the bin, then SCAN. Exhausting the shake budget with
           parts still present is a FAULT; exhausting it with nothing
           proposed is an EMPTY.
    EMPTY  the bin is done; ask for the next one. Terminal for this bin.
    FAULT  stop and raise. Terminal until an operator clears it.

Two gates decide almost everything. The **score gate** is 0.7: on
cross-validated train scenes a pose at or above it carries ~0.99
precision at 5 mm (analysis/score_calibration.md), and the top-scoring
pose of every scene landed within 3.2 mm. The **drift verdict** comes
from :mod:`deploy.cell.drift`; a cell that keeps picking on a
calibration known to be outside its budget is manufacturing scrap
quietly, which is the failure mode this whole package exists to prevent.

The distinction the cell has to keep straight is *nothing proposed* from
*nothing verified*. No proposals means the segmenters saw no part: an
empty bin, or a domain the models do not recognise. Proposals that
nothing verifies means parts are there and geometry refused them: a pile
that needs disturbing, or a genuine estimation failure. The first ends
in EMPTY, the second in FAULT, and confusing them either stops a running
line or leaves it grinding at an empty bin.

This is a policy, not an executor: it chooses, it does not move
anything. The controller runs the action and reports back.

Run ``python -m deploy.cell.policy`` for a simulated sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import drift
from .grasp import GraspCandidate

SCAN = "scan"
PICK = "pick"
RETRY = "retry"
SHAKE = "shake"
EMPTY = "empty"
FAULT = "fault"

STATES = (SCAN, PICK, RETRY, SHAKE, EMPTY, FAULT)

#: Terminal states: the cycle does not continue on its own from here.
TERMINAL = (EMPTY, FAULT)

#: The cell's accept gate. At >= 0.7 a pose carries ~0.99 precision at
#: 5 mm on cross-validated data (analysis/score_calibration.md); below
#: it, a confident-looking pose is as likely to be a near-symmetric flip
#: as the truth, and a flip puts the tool on the wrong end of the part.
#: This is the same number the pose service gates on
#: (deploy/pose_service/config.py, accept_score) -- restated rather than
#: imported so a cell running against a different service still has the
#: threshold its own decisions were reviewed against.
ACCEPT_SCORE = 0.7

#: Rescans allowed before the cell stops re-looking and starts
#: disturbing the bin. One rescan covers a frame spoiled by motion,
#: a reflection or a dropped depth image; a second rescan of an
#: undisturbed bin returns the same answer, so there is no third.
MAX_RESCANS = 2

#: Shakes allowed before the cell gives up on the bin. Three is what a
#: pile of these parts takes to change its top layer; past that the bin
#: is either empty or the cell is not seeing what is in it.
MAX_SHAKES = 3

#: Consecutive failed picks before the bin gets shaken rather than
#: retried. Two retries on different grasps is enough evidence that the
#: part is not going to come out of the pile as it lies.
MAX_RETRIES = 2

#: Frames with no proposals at all before the bin is treated as possibly
#: empty. Two, because one frame can be spoiled.
EMPTY_CONFIRM_FRAMES = 2

#: A retry must not aim at the grasp that just failed. Anything within
#: this of the failed grasp point counts as the same attempt, metres.
#: 15 mm is wider than the pipeline's own ~2 mm in-plane spread
#: (report.md, Limitations) and narrower than the part, so a retry lands
#: on a different feature of the same part or on a different part.
RETRY_EXCLUSION_M = 0.015

#: Expected wall-clock for each action, seconds -- what the cell budgets,
#: not what this module spends. SCAN is the pose service in pick mode
#: (0.7 s mean, 2.1 s max on GPU; 1.6-2.4 s CPU-only on 4 cores,
#: analysis/runtime.md) plus ~0.15 s of grasp planning. PICK sits in the
#: middle of the target context's 4-8 s production cycle and is
#: dominated by the robot move. SHAKE is a fixed programme.
BUDGET_S = {SCAN: 2.5, PICK: 5.0, RETRY: 5.0, SHAKE: 8.0,
            EMPTY: 0.0, FAULT: 0.0}


@dataclass
class CellFrame:
    """One frame, as the cell sees it: poses, grasps, and what failed.

    ``poses`` accepts the pose service's ``PoseEstimateDTO``, the
    estimator's ``PoseEstimate`` or plain dicts -- only ``score`` is read
    here, because the geometry was already consumed by the planner.
    """

    scene_id: str = ""
    poses: Sequence[Any] = ()
    grasps: Sequence[GraspCandidate] = ()
    #: Masks the segmenters proposed before registration. Zero means the
    #: models saw no part at all, which is a different situation from
    #: proposals that failed to verify.
    n_proposals: int = 0
    #: Set when the frame itself failed (a timeout, a dropped image).
    error: Optional[str] = None
    #: Why the planner rejected what it rejected, for the log.
    rejections: Sequence[str] = ()

    @classmethod
    def from_service(cls, frame_result: Any,
                     grasps: Sequence[GraspCandidate] = (),
                     rejections: Sequence[str] = ()) -> "CellFrame":
        """Adapt a ``deploy.pose_service.schema.FrameResult``."""
        return cls(scene_id=frame_result.scene_id, poses=frame_result.poses,
                   grasps=grasps, n_proposals=frame_result.n_proposals,
                   error=frame_result.error, rejections=rejections)

    def verified(self, gate: float = ACCEPT_SCORE) -> List[Any]:
        """Poses at or above the cell's gate."""
        return [p for p in self.poses if _score(p) >= gate]


@dataclass
class Action:
    """What the cell should do next, and why."""

    state: str
    reason: str
    cycle_budget_s: float
    grasp: Optional[GraspCandidate] = None
    #: Verdict carried through even when it did not change the decision,
    #: so a WATCH is visible in the cell's log from the first cycle.
    drift_verdict: str = drift.OK

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "reason": self.reason,
                "cycle_budget_s": self.cycle_budget_s,
                "drift_verdict": self.drift_verdict,
                "grasp": None if self.grasp is None else self.grasp.to_dict()}

    def __repr__(self) -> str:
        return "%-6s %-4.1fs  %s" % (self.state.upper(), self.cycle_budget_s,
                                     self.reason)


@dataclass
class PickHistory:
    """The counters the transitions read. One bin's worth.

    "Progress" means a pick that came out: it resets every escalation,
    because a bin that is yielding parts is not a bin in trouble.
    """

    picks: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    rescans_since_progress: int = 0
    shakes_since_progress: int = 0
    consecutive_barren: int = 0        # frames with nothing verified
    consecutive_no_proposals: int = 0
    #: Grasp point of the pick that just failed, camera frame, so a retry
    #: can aim somewhere else.
    last_failed_point: Optional[List[float]] = None
    #: Latest verdict from deploy.cell.drift, set by the cell.
    drift_verdict: str = drift.OK
    states: List[str] = field(default_factory=list)

    def record(self, action: Action) -> None:
        """Note the action the cell is about to run."""
        self.states.append(action.state)
        if action.state == SCAN:
            self.rescans_since_progress += 1
        elif action.state == SHAKE:
            self.shakes_since_progress += 1
            self.rescans_since_progress = 0
            self.consecutive_barren = 0
            self.consecutive_no_proposals = 0
        elif action.state in (PICK, RETRY):
            self.picks += 1
            self.last_failed_point = (
                None if action.grasp is None
                else [float(v) for v in action.grasp.T_camera_grasp.t])

    def report_outcome(self, outcome: str) -> None:
        """What the cell's success sensor said about the last pick."""
        if outcome not in drift.OUTCOMES:
            raise ValueError("outcome must be one of %s, got %r"
                             % (drift.OUTCOMES, outcome))
        if outcome == drift.SUCCESS:
            self.successes += 1
            self.consecutive_failures = 0
            self.rescans_since_progress = 0
            self.shakes_since_progress = 0
            self.consecutive_barren = 0
            self.consecutive_no_proposals = 0
            self.last_failed_point = None
        elif outcome in (drift.SLIP, drift.MISS):
            self.consecutive_failures += 1


class PickPolicy:
    """Chooses the next action. Holds no state -- the history does.

    The loop a cell runs is two calls, plus one after a pick::

        action = policy.next_action(frame, history)
        history.record(action)
        ...                                   # the controller executes it
        history.report_outcome(drift.SUCCESS)  # only after PICK/RETRY

    ``next_action`` folds the *frame's* own evidence into the history
    (the barren and no-proposal counters, which only a frame can tell
    you about); ``record`` folds in the *action taken*; and
    ``report_outcome`` folds in what the cell's success sensor said. Each
    counter has exactly one writer.
    """

    def __init__(self, accept_score: float = ACCEPT_SCORE,
                 max_rescans: int = MAX_RESCANS,
                 max_shakes: int = MAX_SHAKES,
                 max_retries: int = MAX_RETRIES,
                 stop_on_recalibrate: bool = True):
        """Args:
            accept_score: The gate a pose must clear to be picked.
            max_rescans: Rescans before the bin is disturbed instead.
            max_shakes: Shakes before the bin is given up on.
            max_retries: Failed picks before shaking rather than retrying.
            stop_on_recalibrate: Stop the cell when the drift monitor
                says recalibrate. The alternative -- carrying on with a
                warning -- keeps the line running and makes scrap
                silently, so it has to be chosen deliberately.
        """
        if not 0.0 < accept_score <= 1.0:
            raise ValueError("accept_score must be in (0, 1], got %r"
                             % (accept_score,))
        self.accept_score = accept_score
        self.max_rescans = max_rescans
        self.max_shakes = max_shakes
        self.max_retries = max_retries
        self.stop_on_recalibrate = stop_on_recalibrate

    def next_action(self, frame_result: CellFrame,
                    history: PickHistory) -> Action:
        """Decide, from one frame and the counters so far.

        Args:
            frame_result: The frame the cell just took, with the grasps
                the planner returned for it.
            history: Counters for the current bin. Updated by the caller
                through :meth:`PickHistory.record` and
                :meth:`PickHistory.report_outcome`.
        """
        verdict = history.drift_verdict
        if verdict == drift.RECALIBRATE and self.stop_on_recalibrate:
            return self._act(FAULT, "drift monitor says recalibrate: the "
                                    "camera-to-robot transform is outside "
                                    "the accuracy budget", verdict)

        if frame_result.error:
            return self._barren(frame_result, history, verdict,
                                "frame failed (%s)" % frame_result.error)

        verified = frame_result.verified(self.accept_score)
        if not verified:
            if frame_result.n_proposals == 0:
                return self._no_proposals(history, verdict)
            return self._barren(
                frame_result, history, verdict,
                "%d proposals, none verified at score >= %.2f"
                % (frame_result.n_proposals, self.accept_score))

        grasp = self._choose(frame_result, history)
        if grasp is None:
            return self._shake_or_fault(
                history, verdict,
                "%d pose(s) verified but no grasp is reachable%s"
                % (len(verified), _first_reason(frame_result)))

        if history.consecutive_failures == 0:
            return self._act(PICK, "top of %d candidate(s): %s on pose %d, "
                                   "score %.3f, rank %.3f"
                             % (len(frame_result.grasps), grasp.grasp_name,
                                grasp.pose_index, grasp.score, grasp.rank),
                             verdict, grasp)
        if history.consecutive_failures < self.max_retries:
            return self._act(RETRY, "%d failed pick(s); trying %s on pose "
                                    "%d, %.0f mm from the last attempt"
                             % (history.consecutive_failures,
                                grasp.grasp_name, grasp.pose_index,
                                1000.0 * _distance(grasp,
                                                   history.last_failed_point)),
                             verdict, grasp)
        return self._shake_or_fault(
            history, verdict,
            "%d consecutive failed picks (limit %d)"
            % (history.consecutive_failures, self.max_retries))

    # -- the branches ----------------------------------------------------

    def _choose(self, frame_result: CellFrame,
                history: PickHistory) -> Optional[GraspCandidate]:
        """Best candidate, skipping the one that just failed.

        The planner already ranked them; the only thing the policy adds
        is not aiming at the grasp that has already been tried, because
        the bin has barely changed since.
        """
        for candidate in frame_result.grasps:
            if not candidate.accepted:
                continue
            if history.consecutive_failures and \
                    _distance(candidate, history.last_failed_point) < \
                    RETRY_EXCLUSION_M:
                continue
            return candidate
        return None

    def _no_proposals(self, history: PickHistory, verdict: str) -> Action:
        """The segmenters saw no part. Look again, disturb, then declare."""
        history.consecutive_no_proposals += 1
        if history.consecutive_no_proposals < EMPTY_CONFIRM_FRAMES:
            return self._act(SCAN, "no proposals; rescanning to confirm "
                                   "(%d of %d)"
                             % (history.consecutive_no_proposals,
                                EMPTY_CONFIRM_FRAMES), verdict)
        if history.shakes_since_progress < self.max_shakes:
            return self._act(SHAKE, "no proposals in %d frames; shaking in "
                                    "case parts are held in a corner"
                             % history.consecutive_no_proposals, verdict)
        return self._act(EMPTY, "no proposals in %d frames and %d shakes "
                                "produced none: the bin is empty"
                         % (history.consecutive_no_proposals,
                            history.shakes_since_progress), verdict)

    def _barren(self, frame_result: CellFrame, history: PickHistory,
                verdict: str, why: str) -> Action:
        """Parts are there; nothing came back the cell will act on."""
        history.consecutive_barren += 1
        if history.rescans_since_progress < self.max_rescans:
            return self._act(SCAN, "%s; rescanning (%d of %d)"
                             % (why, history.rescans_since_progress + 1,
                                self.max_rescans), verdict)
        return self._shake_or_fault(history, verdict, why)

    def _shake_or_fault(self, history: PickHistory, verdict: str,
                        why: str) -> Action:
        if history.shakes_since_progress < self.max_shakes:
            return self._act(SHAKE, "%s; shaking the bin (%d of %d)"
                             % (why, history.shakes_since_progress + 1,
                                self.max_shakes), verdict)
        return self._act(FAULT, "%s, and %d shakes did not help: stopping "
                                "and raising" % (why,
                                                 history.shakes_since_progress),
                         verdict)

    def _act(self, state: str, reason: str, verdict: str,
             grasp: Optional[GraspCandidate] = None) -> Action:
        if verdict == drift.WATCH:
            reason += " [drift: watch]"
        return Action(state=state, reason=reason,
                      cycle_budget_s=BUDGET_S[state], grasp=grasp,
                      drift_verdict=verdict)


def _score(pose: Any) -> float:
    if isinstance(pose, dict):
        return float(pose.get("score", pose.get("submission_score", 0.0)))
    return float(getattr(pose, "score",
                         getattr(pose, "submission_score", 0.0)))


def _distance(candidate: GraspCandidate,
              point: Optional[Sequence[float]]) -> float:
    if point is None:
        return float("inf")
    return float(np.linalg.norm(candidate.T_camera_grasp.t
                                - np.asarray(point, dtype=np.float64)))


def _first_reason(frame_result: CellFrame) -> str:
    if not frame_result.rejections:
        return ""
    return " (%s)" % frame_result.rejections[0]


def _self_check() -> int:
    """Drive the machine through every transition and print the trace."""
    import os

    from .grasp import GraspPlanner, default_grasps_path

    here = os.path.dirname(os.path.abspath(__file__))
    cad = os.path.join(os.path.dirname(os.path.dirname(here)),
                       "model", "3d_model.ply")
    planner = GraspPlanner(cad, default_grasps_path())

    def frame_with_grasps(scene_id, score, n_poses=2):
        """Real candidates from the real CAD, with the plate presented
        to the camera and nothing in the way."""
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        poses = [{"R": R.tolist(), "t": [0.02 * i, 0.0, 0.70 + 0.005 * i],
                  "score": score} for i in range(n_poses)]
        grasps = planner.plan(poses, np.zeros((0, 3)))
        return CellFrame(scene_id=scene_id, poses=poses, grasps=grasps,
                         n_proposals=len(poses) * 2)

    barren = CellFrame(scene_id="barren", poses=[{"score": 0.42}],
                       grasps=[], n_proposals=4,
                       rejections=["pose 0 / suction_plate_top: score 0.42 "
                                   "below the gate"])
    nothing = CellFrame(scene_id="empty", poses=[], grasps=[],
                        n_proposals=0)
    unreachable = CellFrame(
        scene_id="buried", poses=[{"score": 0.91}], grasps=[],
        n_proposals=3,
        rejections=["pose 0 / suction_plate_top: approach blocked: a "
                    "measured point 2.1 mm off the axis"])

    #: (frame, outcome the cell reports after a PICK/RETRY, expected state)
    script = [
        (frame_with_grasps("000001", 0.93), drift.SUCCESS, PICK),
        (frame_with_grasps("000002", 0.88), drift.MISS, PICK),
        (frame_with_grasps("000003", 0.88), drift.SLIP, RETRY),
        (frame_with_grasps("000004", 0.88), None, SHAKE),
        (barren, None, SCAN),
        (barren, None, SCAN),
        (barren, None, SHAKE),
        (unreachable, None, SHAKE),
        (nothing, None, SCAN),
        (nothing, None, EMPTY),
    ]

    print("policy.py self-check\n")
    print("  a bin worked through to empty (every cycle begins with a\n"
          "  scan -- that is what produces the frame; the SCAN *state*\n"
          "  means scan again without acting):")
    policy = PickPolicy()
    history = PickHistory()
    observed = []
    for frame, outcome, _ in script:
        action = policy.next_action(frame, history)
        observed.append(action.state)
        print("    %-9s %r" % (frame.scene_id, action))
        history.record(action)
        if outcome is not None and action.state in (PICK, RETRY):
            history.report_outcome(outcome)
            print("               -> cell reports %s" % outcome)
    expected = [state for _, _, state in script]
    failures = 0
    if observed != expected:
        print("      FAIL: got %s" % observed)
        print("            expected %s" % expected)
        failures += 1
    else:
        print("    sequence %s" % " -> ".join(observed))
    print("    %d picks, %d successes, %d shakes"
          % (history.picks, history.successes,
             history.states.count(SHAKE)))

    print("\n  parts present that never verify: rescan, shake, then stop")
    history = PickHistory()
    trace = []
    for _ in range(12):
        action = policy.next_action(barren, history)
        trace.append(action.state)
        history.record(action)
        if action.is_terminal:
            print("    %s" % " -> ".join(trace))
            print("    final: %s" % action.reason)
            break
    if trace[-1] != FAULT:
        print("      FAIL: expected FAULT, got %s" % trace[-1])
        failures += 1

    print("\n  drift verdicts")
    history = PickHistory()
    good = frame_with_grasps("000010", 0.95)
    for verdict, expect in ((drift.OK, PICK), (drift.WATCH, PICK),
                            (drift.RECALIBRATE, FAULT)):
        history.drift_verdict = verdict
        action = policy.next_action(good, history)
        print("    %-12s -> %r" % (verdict, action))
        if action.state != expect:
            print("      FAIL: expected %s" % expect)
            failures += 1
    history.drift_verdict = drift.RECALIBRATE
    action = PickPolicy(stop_on_recalibrate=False).next_action(good, history)
    print("    %-12s -> %r   (stop_on_recalibrate=False)"
          % ("recalibrate", action))
    if action.state != PICK:
        print("      FAIL: expected the cell to keep picking")
        failures += 1

    print("\n  a retry must not aim at the grasp that just failed")
    history = PickHistory()
    frame = frame_with_grasps("000020", 0.90)
    first = policy.next_action(frame, history)
    history.record(first)
    history.report_outcome(drift.MISS)
    second = policy.next_action(frame, history)
    moved = _distance(second.grasp, history.last_failed_point)
    print("    first  %s on pose %d" % (first.grasp.grasp_name,
                                        first.grasp.pose_index))
    print("    retry  %s on pose %d, %.1f mm away"
          % (second.grasp.grasp_name, second.grasp.pose_index,
             1000.0 * moved))
    if second.state != RETRY or moved < RETRY_EXCLUSION_M:
        print("      FAIL: retry did not move off the failed grasp")
        failures += 1

    print("\n  cycle budgets: %s"
          % ", ".join("%s %.1fs" % (k, v) for k, v in BUDGET_S.items()))
    if abs(BUDGET_S[SCAN] + BUDGET_S[PICK] - 7.5) > 1e-9:
        failures += 1
    print("    a scan-and-pick cycle budgets %.1f s, inside the target "
          "context's 4-8 s" % (BUDGET_S[SCAN] + BUDGET_S[PICK]))

    print("\n  %d failure(s)" % failures)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _self_check() else 0)
