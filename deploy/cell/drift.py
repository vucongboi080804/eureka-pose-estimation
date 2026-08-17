"""Watch for camera-to-robot calibration drift, the failure with no fault.

A hand-eye calibration does not break, it creeps: a camera mount settles,
a bracket takes a knock nobody logs, a temperature cycle moves a post.
The vision system keeps reporting poses of the same quality, the service
keeps returning score 0.9, and the picks slowly start catching edges.
There is no exception to catch and no metric in the pose pipeline that
moves, which is why the target context calls it the most insidious field
failure -- and why the watch has to live out here, between the pose and
the pick, rather than inside either.

**What is observable, and by what.**

*With robot feedback* -- a touch-off, a force-controlled seating move, a
fixture that reports where the part actually ended up -- the cell learns
where the part really was, and the difference from where vision said it
was is measured directly. Its *systematic* part is the calibration
error. That is the only signal here that sees the camera-to-robot
transform itself, and it is why per-axis **medians**, not magnitudes,
are what this module accumulates: the median of a zero-mean measurement
error tends to zero however noisy it is, while a real offset survives.
Averaging magnitudes would turn noise into a trend.

*Without robot feedback* the cell can still be watched, but be clear
about what is being watched. The verification ``support`` and the pose
polish residual are camera-to-scene quantities: they detect the camera
degrading (mount vibration changing focus, contamination, a lighting
change, a knock large enough to move the intrinsics) and they say
nothing at all about where the robot base is. A camera that shifts
rigidly on its mount without changing its optics moves *every* pose by
the same amount and leaves both signals flat while the cell quietly
mis-picks. Vision-only monitoring is therefore an early warning for a
correlated class of faults, not a substitute for feedback -- a cell that
needs the drift verdict to mean something must give the monitor
corrections from somewhere.

**Structure.** Two horizons, both bounded. A rolling window of recent
observations gives the level; a ring of per-shift aggregates gives the
trend, because at a 4-8 s cycle a shift is thousands of picks and no
count-bounded window of raw picks reaches back far enough to see a
tenth of a millimetre per day. The trend is a Theil-Sen fit over the
shift medians -- a median of pairwise slopes, which needs no model of the
noise and does not chase one bad shift. Not a Kalman filter: there is no
process model worth writing down for a bracket creeping, and a filter
that is tuned wrong hides exactly the signal it was installed to find.

Everything is also appended to JSONL, so the RAM footprint is fixed and
the history is not: at ~250 bytes an observation and a 6 s cycle, a
month of two-shift operation is about 30 MB.

Run ``python -m deploy.cell.drift`` for the injected-drift and
false-positive runs.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

#: The cell is inside its accuracy budget; keep picking.
OK = "ok"
#: Still picking, but the calibration is drifting; schedule maintenance.
WATCH = "watch"
#: Outside the budget. Re-run the hand-eye before the next shift.
RECALIBRATE = "recalibrate"

VERDICTS = (OK, WATCH, RECALIBRATE)

#: Pick outcomes a cell reports. ``unknown`` is for cells with no
#: success sensor: the observation still carries the vision signals.
SUCCESS = "success"
SLIP = "slip"
MISS = "miss"
UNKNOWN = "unknown"
OUTCOMES = (SUCCESS, SLIP, MISS, UNKNOWN)

#: Systematic offset at which the calibration stops being the smaller
#: error, millimetres. The target context's rule of thumb puts
#: end-to-end accuracy near 0.5% of the container: a 0.2 m tray of these
#: parts allows about 1 mm in total. At 1 mm of *systematic* offset that
#: whole allowance is spent on calibration alone, before the pipeline's
#: own ~2 mm in-plane spread (report.md, Limitations) is added -- so this
#: is where an engineer should be told, not where the cell should stop.
WATCH_BIAS_MM = 1.0

#: ...and where it should stop. At 2 mm the systematic offset equals the
#: pipeline's entire random in-plane error, so calibration has become the
#: dominant term and every pick is being aimed with a known bias.
RECALIBRATE_BIAS_MM = 2.0

#: How far ahead the trend is projected when deciding to warn. A week is
#: the horizon a maintenance slot is booked in.
PROJECTION_DAYS = 7.0

#: Below this many feedback observations the medians are anecdotes and
#: the monitor reports ``ok`` with a stated reason rather than a verdict.
MIN_FEEDBACK_SAMPLES = 20

#: ...and below this many shifts there is no trend to fit.
MIN_TREND_SHIFTS = 4

#: Drop in the median verification ``support`` from the commissioning
#: baseline that is worth a look. Support is the fraction of the posed
#: model's silhouette that the depth map agrees with (src/verify.py); it
#: sits near 0.8-0.9 on clean scenes, and a tenth is well outside its
#: scene-to-scene wobble. Vision-side only -- see the module docstring.
SUPPORT_DROP = 0.10

#: Pick failure rate that is itself evidence something is wrong. The
#: target context states 98-99.9% pick success, so 2% failures is the
#: floor of specified behaviour.
MAX_FAILURE_RATE = 0.02

#: Rise in the median pose-polish residual worth a look, millimetres.
#: The polish stage pins the in-plane shift against the through-hole
#: centres (report.md, Polish); its residual is a camera-to-scene
#: quantity that grows when the optics or the depth map degrade.
POLISH_RISE_MM = 1.0

#: Raw values kept per shift for its medians. A shift at a 6 s cycle is
#: thousands of picks; 2000 already pins a median to ~0.01 mm and caps
#: what one long shift can hold in RAM. The pick *counts* are exact
#: regardless -- only the median inputs are capped.
MAX_PER_SHIFT = 2000


class DriftError(ValueError):
    """An observation the monitor cannot make sense of."""


@dataclass
class PickObservation:
    """One pick cycle, as the cell saw it.

    ``correction_mm`` is the only field that observes the camera-to-robot
    transform: the offset the robot had to apply, in the **robot base
    frame**, to actually reach the part -- from a touch-off, a seating
    move, or a fixture that measured where the part ended up. Sign
    convention: ``measured - commanded``, so a positive X means the part
    was further along base +X than vision said. Cells without such a
    move leave it None and the monitor falls back to the vision-only
    signals, with the limits the module docstring sets out.
    """

    timestamp: float = field(default_factory=time.time)
    scene_id: str = ""
    shift: str = ""
    outcome: str = UNKNOWN
    #: The pose the grasp came from: score, and the verification support
    #: behind it (PoseEstimateDTO.depth_verification).
    score: Optional[float] = None
    support: Optional[float] = None
    #: In-plane correction the polish stage applied to this pose, mm.
    polish_residual_mm: Optional[float] = None
    grasp: str = ""
    #: Where the tool was sent, robot base frame, metres. Kept so a bad
    #: run can be replayed against the poses that produced it.
    commanded_xyz_m: Optional[List[float]] = None
    #: measured - commanded, robot base frame, millimetres.
    correction_mm: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise DriftError("outcome must be one of %s, got %r"
                             % (OUTCOMES, self.outcome))
        if self.correction_mm is not None:
            vector = np.asarray(self.correction_mm, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise DriftError("correction_mm must be 3 finite numbers in "
                                 "the robot base frame, got %r"
                                 % (self.correction_mm,))
            if float(np.abs(vector).max()) > 100.0:
                raise DriftError(
                    "correction of %.0f mm is not calibration drift -- a "
                    "correction that large is a failed pick or a units "
                    "mistake, and folding it into the median would hide a "
                    "real trend" % float(np.abs(vector).max()))
            self.correction_mm = [float(v) for v in vector]
        if not self.shift:
            self.shift = shift_of(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": float(self.timestamp),
                "scene_id": self.scene_id, "shift": self.shift,
                "outcome": self.outcome, "score": self.score,
                "support": self.support,
                "polish_residual_mm": self.polish_residual_mm,
                "grasp": self.grasp,
                "commanded_xyz_m": self.commanded_xyz_m,
                "correction_mm": self.correction_mm}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PickObservation":
        return cls(timestamp=float(payload.get("timestamp", 0.0)),
                   scene_id=str(payload.get("scene_id", "")),
                   shift=str(payload.get("shift", "")),
                   outcome=str(payload.get("outcome", UNKNOWN)),
                   score=_optional_float(payload.get("score")),
                   support=_optional_float(payload.get("support")),
                   polish_residual_mm=_optional_float(
                       payload.get("polish_residual_mm")),
                   grasp=str(payload.get("grasp", "")),
                   commanded_xyz_m=payload.get("commanded_xyz_m"),
                   correction_mm=payload.get("correction_mm"))


@dataclass
class ShiftSummary:
    """One shift, reduced to the medians the trend is fitted on."""

    shift: str
    t_mid: float
    n: int = 0
    n_feedback: int = 0
    n_success: int = 0
    n_failed: int = 0
    n_scored: int = 0
    median_correction_mm: Optional[List[float]] = None
    median_support: Optional[float] = None
    median_polish_mm: Optional[float] = None

    @property
    def bias_mm(self) -> Optional[float]:
        if self.median_correction_mm is None:
            return None
        return float(np.linalg.norm(self.median_correction_mm))

    @property
    def failure_rate(self) -> Optional[float]:
        judged = self.n_success + self.n_failed
        return None if judged == 0 else self.n_failed / float(judged)


@dataclass
class DriftStats:
    """What the monitor concluded, and why."""

    n_observations: int
    n_shifts: int
    n_feedback: int
    verdict: str
    reasons: List[str]
    #: Systematic offset now, from the robust fit, mm. None without
    #: robot feedback -- and that is the honest answer, not zero.
    bias_mm: Optional[float] = None
    bias_axes_mm: Optional[List[float]] = None
    #: Window median of the same quantity: the level, without the fit.
    window_bias_mm: Optional[float] = None
    trend_mm_per_day: Optional[float] = None
    projected_bias_mm: Optional[float] = None
    failure_rate: Optional[float] = None
    support_median: Optional[float] = None
    support_baseline: Optional[float] = None
    polish_median_mm: Optional[float] = None
    polish_trend_mm_per_day: Optional[float] = None
    #: True when the verdict rests on vision-only signals, which cannot
    #: see the camera-to-robot transform.
    vision_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["reasons"] = list(self.reasons)
        return payload


class DriftMonitor:
    """Accumulates pick observations and reports a calibration verdict.

    Bounded in memory by construction: ``window`` raw observations and
    ``shifts`` per-shift aggregates, whatever the cell's uptime. The
    JSONL file is the unbounded part, and it is append-only so a crash
    mid-shift loses one line, not the history.
    """

    def __init__(self, path: Optional[str] = None, window: int = 500,
                 shifts: int = 90,
                 support_baseline: Optional[float] = None,
                 polish_baseline_mm: Optional[float] = None):
        """Args:
            path: JSONL file to append to. None keeps it in memory only.
            window: Raw observations kept for the level estimate.
            shifts: Shift aggregates kept for the trend. 90 shifts is a
                quarter of two-shift operation -- long enough that a
                tenth of a millimetre a day is unmistakable.
            support_baseline: Median verification support measured at
                commissioning. Without it the support signal has nothing
                to be a drop *from*, and the monitor says so rather than
                inventing a reference from its own first shift.
            polish_baseline_mm: The same for the polish residual.
        """
        if window < 1 or shifts < 1:
            raise DriftError("window and shifts must be positive")
        self.path = path
        self.window = deque(maxlen=window)      # type: deque
        self._max_shifts = shifts
        self._pending = OrderedDict()           # type: OrderedDict
        self._shifts_cache = OrderedDict()      # type: OrderedDict
        self._dirty = set()                     # type: set
        self.support_baseline = support_baseline
        self.polish_baseline_mm = polish_baseline_mm
        self.n_observations = 0

    # -- accumulation ----------------------------------------------------

    def observe(self, observation: PickObservation) -> None:
        """Record one pick. Appends to the JSONL if one was configured."""
        if not isinstance(observation, PickObservation):
            raise DriftError("expected a PickObservation, got %r"
                             % type(observation).__name__)
        self.window.append(observation)
        self.n_observations += 1
        self._fold(observation)
        if self.path is not None:
            self._append(observation)

    def _append(self, observation: PickObservation) -> None:
        """One line, flushed. A monitoring file that loses the tail on a
        power cut is monitoring the wrong thing."""
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(observation.to_dict(),
                                    sort_keys=True) + "\n")

    def _fold(self, observation: PickObservation) -> None:
        """Accumulate into the shift the observation belongs to.

        O(1) per observation: the medians are recomputed only when
        someone asks for them, and only for the shifts that changed.
        Recomputing on every pick would make a shift quadratic in its own
        length, which at a 6 s cycle is 5000 picks.
        """
        key = observation.shift
        bucket = self._pending.get(key)
        if bucket is None:
            bucket = {"t": [], "n": 0, "success": 0, "failed": 0,
                      "scored": 0, "correction": [], "support": [],
                      "polish": []}
            self._pending[key] = bucket
        bucket["n"] += 1
        if observation.outcome == SUCCESS:
            bucket["success"] += 1
        elif observation.outcome in (SLIP, MISS):
            bucket["failed"] += 1
        if observation.score is not None:
            bucket["scored"] += 1
        # Only the first MAX_PER_SHIFT values of a shift feed its
        # medians. Within one shift the drift this module looks for moves
        # under 0.05 mm, and 2000 samples already pin a median to ~0.01
        # mm, so the cap costs nothing and bounds the memory a long shift
        # can take.
        if bucket["n"] <= MAX_PER_SHIFT:
            bucket["t"].append(observation.timestamp)
            if observation.correction_mm is not None:
                bucket["correction"].append(observation.correction_mm)
            if observation.support is not None:
                bucket["support"].append(observation.support)
            if observation.polish_residual_mm is not None:
                bucket["polish"].append(observation.polish_residual_mm)
        self._dirty.add(key)
        while len(self._pending) > self._max_shifts:
            self._pending.popitem(last=False)
        self._shifts_cache.pop(key, None)

    def _summaries(self) -> List[ShiftSummary]:
        """Shift aggregates, oldest first, recomputed only where dirty."""
        for key in self._dirty:
            if key in self._pending:
                self._shifts_cache[key] = _summarise(key, self._pending[key])
        self._dirty.clear()
        for key in list(self._shifts_cache):
            if key not in self._pending:
                self._shifts_cache.pop(key)
        return sorted(self._shifts_cache.values(), key=lambda s: s.t_mid)

    @property
    def shifts(self) -> Dict[str, ShiftSummary]:
        """The per-shift table, oldest first."""
        return OrderedDict((s.shift, s) for s in self._summaries())

    # -- reading ---------------------------------------------------------

    def stats(self) -> DriftStats:
        """Verdict and the numbers behind it."""
        summaries = self._summaries()
        feedback = [o for o in self.window if o.correction_mm is not None]
        stats_out = DriftStats(n_observations=self.n_observations,
                               n_shifts=len(summaries),
                               n_feedback=len(feedback), verdict=OK,
                               reasons=[])

        with_correction = [s for s in summaries
                           if s.median_correction_mm is not None]
        if feedback:
            window_axes = np.median(
                np.array([o.correction_mm for o in feedback]), axis=0)
            stats_out.window_bias_mm = float(np.linalg.norm(window_axes))
        if with_correction:
            axes, trend = _fit_axes(with_correction)
            stats_out.bias_axes_mm = [float(v) for v in axes]
            stats_out.bias_mm = float(np.linalg.norm(axes))
            stats_out.trend_mm_per_day = float(np.linalg.norm(trend))
            stats_out.projected_bias_mm = float(
                np.linalg.norm(axes + trend * PROJECTION_DAYS))

        # The failure rate is read off the rolling window: it is a rate,
        # so it needs volume rather than recency, and 500 picks pin 2% to
        # about half a point. Trends come from the shift ring, the only
        # horizon long enough to fit one.
        judged = [o for o in self.window
                  if o.outcome in (SUCCESS, SLIP, MISS)]
        if judged:
            stats_out.failure_rate = sum(
                1 for o in judged if o.outcome != SUCCESS) / float(len(judged))
        # The two vision-only signals are read off the most recent shift,
        # which is the horizon they are useful on: a per-shift median of
        # thousands of picks is stable, and a step between shifts is
        # exactly the shape a camera taking a knock leaves.
        recent = summaries[-1] if summaries else None
        if recent is not None:
            stats_out.support_median = recent.median_support
            stats_out.polish_median_mm = recent.median_polish_mm
        stats_out.support_baseline = self.support_baseline
        polish = [(s.t_mid, s.median_polish_mm) for s in summaries
                  if s.median_polish_mm is not None]
        if len(polish) >= MIN_TREND_SHIFTS:
            stats_out.polish_trend_mm_per_day = _theil_sen(
                np.array([t for t, _ in polish]),
                np.array([p for _, p in polish]))

        self._judge(stats_out)
        return stats_out

    def _judge(self, out: DriftStats) -> None:
        """Apply the thresholds. Every escalation states its number."""
        out.vision_only = out.bias_mm is None
        if out.bias_mm is not None:
            if out.n_feedback < MIN_FEEDBACK_SAMPLES:
                out.reasons.append(
                    "only %d corrections in the window (need %d) -- level "
                    "reported, no verdict taken from it"
                    % (out.n_feedback, MIN_FEEDBACK_SAMPLES))
            elif out.bias_mm >= RECALIBRATE_BIAS_MM:
                out.verdict = RECALIBRATE
                out.reasons.append(
                    "systematic offset %.2f mm >= %.2f mm: calibration is "
                    "now the dominant error" % (out.bias_mm,
                                                RECALIBRATE_BIAS_MM))
            elif out.bias_mm >= WATCH_BIAS_MM:
                out.verdict = WATCH
                out.reasons.append(
                    "systematic offset %.2f mm >= %.2f mm"
                    % (out.bias_mm, WATCH_BIAS_MM))
            elif (out.projected_bias_mm is not None
                  and out.n_shifts >= MIN_TREND_SHIFTS
                  and out.projected_bias_mm >= RECALIBRATE_BIAS_MM):
                out.verdict = WATCH
                out.reasons.append(
                    "offset %.2f mm trending %.3f mm/day reaches %.2f mm "
                    "within %.0f days" % (out.bias_mm, out.trend_mm_per_day,
                                          RECALIBRATE_BIAS_MM,
                                          PROJECTION_DAYS))
        else:
            out.reasons.append(
                "no robot feedback: the signals below are camera-to-scene "
                "and cannot see the camera-to-robot transform")

        if out.failure_rate is not None and out.failure_rate > MAX_FAILURE_RATE:
            out.verdict = _escalate(out.verdict, WATCH)
            out.reasons.append("pick failure rate %.1f%% above the %.1f%% "
                               "the cell is specified for"
                               % (100.0 * out.failure_rate,
                                  100.0 * MAX_FAILURE_RATE))
        if (out.support_median is not None and out.support_baseline is not None
                and out.support_baseline - out.support_median >= SUPPORT_DROP):
            out.verdict = _escalate(out.verdict, WATCH)
            out.reasons.append(
                "verification support %.3f is %.3f below the %.3f baseline"
                % (out.support_median,
                   out.support_baseline - out.support_median,
                   out.support_baseline))
        if (out.polish_median_mm is not None
                and self.polish_baseline_mm is not None
                and out.polish_median_mm - self.polish_baseline_mm
                >= POLISH_RISE_MM):
            out.verdict = _escalate(out.verdict, WATCH)
            out.reasons.append(
                "pose polish residual %.2f mm is %.2f mm above the %.2f mm "
                "baseline" % (out.polish_median_mm,
                              out.polish_median_mm - self.polish_baseline_mm,
                              self.polish_baseline_mm))
        if not out.reasons:
            out.reasons.append("inside the budget")

    def report(self, shifts: int = 10) -> str:
        """The table an engineer looks at, most recent shifts last."""
        out = self.stats()
        lines = ["drift monitor: %s" % out.verdict.upper()]
        for reason in out.reasons:
            lines.append("  - " + reason)
        lines.append("")
        lines.append("  %-16s %6s %6s %7s %9s %9s %8s %8s"
                     % ("shift", "picks", "fdbk", "fail%", "bias mm",
                        "dx,dy,dz", "support", "polish"))
        recent = self._summaries()[-shifts:]
        for summary in recent:
            axes = summary.median_correction_mm
            lines.append(
                "  %-16s %6d %6d %7s %9s %9s %8s %8s"
                % (summary.shift, summary.n, summary.n_feedback,
                   "-" if summary.failure_rate is None
                   else "%.1f" % (100.0 * summary.failure_rate),
                   "-" if summary.bias_mm is None
                   else "%.2f" % summary.bias_mm,
                   "-" if axes is None
                   else "%+.1f%+.1f%+.1f" % tuple(axes),
                   "-" if summary.median_support is None
                   else "%.3f" % summary.median_support,
                   "-" if summary.median_polish_mm is None
                   else "%.2f" % summary.median_polish_mm))
        lines.append("")
        lines.append("  offset now      %s   (watch %.1f, recalibrate %.1f)"
                     % ("-" if out.bias_mm is None else "%.2f mm"
                        % out.bias_mm, WATCH_BIAS_MM, RECALIBRATE_BIAS_MM))
        lines.append("  trend           %s"
                     % ("-" if out.trend_mm_per_day is None
                        else "%.3f mm/day, %.2f mm in %.0f days"
                        % (out.trend_mm_per_day, out.projected_bias_mm,
                           PROJECTION_DAYS)))
        if out.vision_only:
            lines.append("  VISION ONLY: these signals watch the camera, "
                         "not the camera-to-robot transform.")
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str, **kwargs: Any) -> "DriftMonitor":
        """Replay a history file into a fresh monitor.

        A malformed line is skipped rather than fatal: the file is
        append-only and the last line of a crashed shift is routinely
        half-written. Everything before it is still evidence.
        """
        monitor = cls(path=None, **kwargs)
        skipped = 0
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    monitor.observe(PickObservation.from_dict(
                        json.loads(line)))
                except (ValueError, KeyError, TypeError):
                    skipped += 1
        monitor.path = path
        monitor.skipped_lines = skipped
        return monitor


def shift_of(timestamp: float) -> str:
    """Default shift key: the UTC date. Cells running two shifts a day
    should pass their own key instead -- the monitor only needs it to be
    stable and to sort chronologically."""
    return time.strftime("%Y-%m-%d", time.gmtime(timestamp))


def _summarise(key: str, bucket: Dict[str, Any]) -> ShiftSummary:
    correction = (np.median(np.array(bucket["correction"]), axis=0).tolist()
                  if bucket["correction"] else None)
    return ShiftSummary(
        shift=key, t_mid=float(np.median(bucket["t"])) if bucket["t"] else 0.0,
        n=bucket["n"], n_feedback=len(bucket["correction"]),
        n_success=bucket["success"], n_failed=bucket["failed"],
        n_scored=bucket["scored"], median_correction_mm=correction,
        median_support=(float(np.median(bucket["support"]))
                        if bucket["support"] else None),
        median_polish_mm=(float(np.median(bucket["polish"]))
                          if bucket["polish"] else None))


def _fit_axes(summaries: Sequence[ShiftSummary]
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-axis robust fit over the shift medians.

    Returns ``(offset_now, slope_per_day)``. Fitting and evaluating at
    the newest shift, rather than reporting the window median, removes
    the half-window lag a rolling median has against a steady trend --
    which is exactly the regime this module exists to catch.
    """
    t = np.array([s.t_mid for s in summaries], dtype=np.float64)
    y = np.array([s.median_correction_mm for s in summaries],
                 dtype=np.float64)
    if len(summaries) < MIN_TREND_SHIFTS or float(t.max() - t.min()) <= 0.0:
        return np.median(y, axis=0), np.zeros(3)
    days = (t - t.max()) / 86400.0          # 0 at the newest shift
    offset, slope = np.zeros(3), np.zeros(3)
    for axis in range(3):
        result = stats.theilslopes(y[:, axis], days)
        slope[axis] = result[0]
        offset[axis] = result[1]            # the fit evaluated at days = 0
    return offset, slope


def _theil_sen(t: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(t) < MIN_TREND_SHIFTS or float(t.max() - t.min()) <= 0.0:
        return None
    return float(stats.theilslopes(y, (t - t.max()) / 86400.0)[0])


def _escalate(current: str, level: str) -> str:
    return level if VERDICTS.index(level) > VERDICTS.index(current) \
        else current


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


# -- synthetic verification ----------------------------------------------

def simulate(days: int, picks_per_day: int, drift_mm_per_day: float,
             noise_mm: float, seed: int = 0, t0: float = 0.0,
             first_day: int = 0,
             axis: Optional[Sequence[float]] = None,
             monitor: Optional[DriftMonitor] = None,
             support_baseline: float = 0.86) -> DriftMonitor:
    """Feed a monitor a synthetic history with a known drift.

    The correction a robot reports is modelled as a systematic part that
    grows linearly (the camera creeping on its mount) plus zero-mean
    measurement noise (the touch-off's own repeatability). That is the
    whole model, and it is the one the thresholds were chosen against.
    """
    rng = np.random.default_rng(seed)
    direction = np.asarray(axis if axis is not None else (0.6, -0.7, 0.39),
                           dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    monitor = monitor or DriftMonitor(window=500, shifts=120,
                                      support_baseline=support_baseline)
    for day in range(first_day, first_day + days):
        for k in range(picks_per_day):
            timestamp = t0 + day * 86400.0 + k * (28800.0 / picks_per_day)
            elapsed = timestamp - t0
            systematic = direction * drift_mm_per_day * elapsed / 86400.0
            monitor.observe(PickObservation(
                timestamp=timestamp, scene_id="sim", outcome=SUCCESS,
                score=0.9, support=support_baseline,
                polish_residual_mm=0.8,
                correction_mm=(systematic
                               + rng.normal(0.0, noise_mm, 3)).tolist()))
    return monitor


def _first_days(days: int, picks_per_day: int, drift: float, noise: float,
                seed: int) -> Tuple[Optional[int], Optional[int]]:
    """Day on which the monitor first says watch, then recalibrate."""
    monitor = DriftMonitor(window=500, shifts=120, support_baseline=0.86)
    origin = time.time() - days * 86400.0
    first_watch, first_recal = None, None
    for day in range(days):
        simulate(1, picks_per_day, drift, noise, seed=seed * 1000 + day,
                 t0=origin, first_day=day, monitor=monitor)
        verdict = monitor.stats().verdict
        if verdict != OK and first_watch is None:
            first_watch = day
        if verdict == RECALIBRATE and first_recal is None:
            first_recal = day
            break
    return first_watch, first_recal


def _self_check() -> int:
    """Injected drift, then pure noise. Both are the evidence."""
    failures = 0
    print("drift.py self-check")

    def check(name, ok, detail=""):
        nonlocal failures
        print("  %-52s %s%s" % (name, "ok" if ok else "FAIL",
                                "" if ok else "  " + detail))
        if not ok:
            failures += 1

    #: Picks a synthetic day carries. Real shifts are thousands; the
    #: medians are already pinned at this many and the run stays short.
    picks_per_day = 40
    #: Touch-off repeatability, mm, 1-sigma per axis. 0.5 mm is a
    #: pessimistic force-seating move on a part of this size.
    noise_mm = 0.5

    # 1. Injected drift. At 0.1 mm/day the thresholds predict watch on
    #    day 10 (1.0 mm) and recalibrate on day 20 (2.0 mm).
    print("\n  injected drift, %d picks/day, %.1f mm/axis noise:"
          % (picks_per_day, noise_mm))
    print("    %-10s %-14s %-14s %s"
          % ("mm/day", "watch on day", "recalibrate", "expected"))
    for drift in (0.1, 0.05, 0.2):
        watch_days, recal_days = [], []
        for seed in range(5):
            watch, recal = _first_days(60, picks_per_day, drift, noise_mm,
                                       seed)
            watch_days.append(watch)
            recal_days.append(recal)
        expected_watch = WATCH_BIAS_MM / drift
        expected_recal = RECALIBRATE_BIAS_MM / drift
        print("    %-10.2f %-14s %-14s %.0f / %.0f"
              % (drift, _spread(watch_days), _spread(recal_days),
                 expected_watch, expected_recal))
        if drift == 0.1:
            ok = all(w is not None and abs(w - expected_watch) <= 2
                     for w in watch_days) and \
                 all(r is not None and abs(r - expected_recal) <= 3
                     for r in recal_days)
            check("0.1 mm/day is caught within 2 days of the threshold", ok,
                  "%s / %s" % (watch_days, recal_days))

    # 2. Pure noise must not trip it.
    print("\n  pure noise, no drift:")
    runs = 100
    for noise in (0.5, 1.0):
        tripped, worst = 0, 0.0
        for seed in range(runs):
            monitor = simulate(40, picks_per_day, 0.0, noise,
                               seed=10000 + seed,
                               t0=time.time() - 40 * 86400.0)
            out = monitor.stats()
            worst = max(worst, out.bias_mm or 0.0)
            if out.verdict != OK:
                tripped += 1
        rate = 100.0 * tripped / runs
        print("    %.1f mm/axis noise, %d runs x 40 days: %d tripped "
              "(%.0f%%), worst offset seen %.2f mm"
              % (noise, runs, tripped, rate, worst))
        check("noise at %.1f mm does not trip the monitor" % noise,
              tripped == 0, "%d/%d tripped" % (tripped, runs))

    # 3. No robot feedback: the verdict must say what it cannot see.
    print("")
    blind = DriftMonitor(window=500, shifts=120, support_baseline=0.86,
                         polish_baseline_mm=0.8)
    t0 = time.time() - 22 * 86400.0
    for day in range(20):
        for k in range(picks_per_day):
            blind.observe(PickObservation(
                timestamp=t0 + day * 86400.0 + k * 600.0, outcome=SUCCESS,
                score=0.9, support=0.86, polish_residual_mm=0.8))
    out = blind.stats()
    check("no feedback: verdict ok, and it says why it is blind",
          out.verdict == OK and out.vision_only
          and "cannot see the camera-to-robot" in out.reasons[0],
          str(out.reasons))
    for k in range(picks_per_day * 3):      # a later shift, support down
        blind.observe(PickObservation(timestamp=t0 + 21 * 86400.0 + k * 200.0,
                                      outcome=SUCCESS, score=0.7,
                                      support=0.70, polish_residual_mm=0.8))
    out = blind.stats()
    check("a support drop raises watch without any feedback",
          out.verdict == WATCH and any("support" in r for r in out.reasons),
          str(out.reasons))

    failing = DriftMonitor(window=500, shifts=120)
    origin = time.time() - 5 * 86400.0
    for day in range(5):
        for k in range(picks_per_day):
            failing.observe(PickObservation(
                timestamp=origin + day * 86400.0 + k * 600.0,
                outcome=MISS if k % 20 == 0 else SUCCESS))
    out = failing.stats()
    check("a 5% pick failure rate raises watch on its own",
          out.verdict == WATCH
          and any("failure rate" in r for r in out.reasons),
          str(out.reasons))

    # 4. Persistence.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history", "picks.jsonl")
        live = DriftMonitor(path=path, window=500, shifts=120,
                            support_baseline=0.86)
        simulate(30, 20, 0.1, noise_mm, seed=42,
                 t0=time.time() - 30 * 86400.0, monitor=live)
        with open(path, "a") as handle:      # a half-written last line
            handle.write('{"timestamp": 1.0, "correc')
        replay = DriftMonitor.from_jsonl(path, window=500, shifts=120,
                                         support_baseline=0.86)
        a, b = live.stats(), replay.stats()
        check("JSONL replay reproduces the verdict and the offset",
              a.verdict == b.verdict
              and abs(a.bias_mm - b.bias_mm) < 1e-9
              and replay.skipped_lines == 1,
              "%s %.4f vs %s %.4f, %d skipped"
              % (a.verdict, a.bias_mm, b.verdict, b.bias_mm,
                 replay.skipped_lines))
        size = os.path.getsize(path) / float(live.n_observations)
        print("  %d observations, %.0f bytes each on disk"
              % (live.n_observations, size))
        print("\n" + "\n".join("  " + line
                               for line in live.report(shifts=6).splitlines()))

    # 5. Observations the monitor must refuse.
    for label, kwargs in (
            ("an unknown outcome", {"outcome": "maybe"}),
            ("a 2-vector correction", {"correction_mm": [1.0, 2.0]}),
            ("a NaN correction", {"correction_mm": [1.0, float("nan"), 0.0]}),
            ("a 150 mm correction", {"correction_mm": [150.0, 0.0, 0.0]})):
        try:
            PickObservation(**kwargs)
            check("refuses %s" % label, False, "accepted")
        except DriftError as exc:
            print("  refuses %-28s %s..." % (label + ":", str(exc)[:44]))

    print("  %d failure(s)" % failures)
    return failures


def _spread(values: Sequence[Optional[int]]) -> str:
    known = [v for v in values if v is not None]
    if not known:
        return "never"
    return ("%d" % known[0] if min(known) == max(known)
            else "%d-%d" % (min(known), max(known)))


if __name__ == "__main__":
    raise SystemExit(1 if _self_check() else 0)
