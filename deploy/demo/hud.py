"""One cycle of the cell, drawn: the frame, the poses, and the decision.

The overlay is the released one. Silhouettes come from ``score.silhouette``
-- the scoring script's own rasteriser, the same call
``scripts/run_pipeline.py`` makes to write ``pred_test`` -- and poses are
drawn with ``cv2.drawFrameAxes`` in ``visualize.py``'s convention (X red,
Y green, Z blue), so a frame out of this module and a frame out of
``overlays_test/`` are the same picture of the same pose. What is added
is what a cell knows and a batch overlay does not: which instance the
cell chose, whether it would pick or rescan, and what the cycle cost.

The panel is the point. A demo that shows only pretty silhouettes proves
the segmenter runs; a demo that shows the score next to the gate next to
the state next to the latency proves the *cell* runs, and it is what a
customer watching a recorded session is actually trying to read.

    from deploy.demo.hud import FrameHud, HudFrame, hardware_line
    hud = FrameHud("model/3d_model.ply", hardware=hardware_line())
    image = hud.render(HudFrame.from_cycle(record))

cv2 drawing only, no matplotlib: this runs on the board, where the
matplotlib import alone costs more than the frame it would draw.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..pick.drift import UNKNOWN as OUTCOME_UNKNOWN
from ..pick.policy import EMPTY, FAULT, PICK
from ..pick.runner import LOOP_OK
from ..pose.schema import GATE_PICK

#: Rendered size. 1280x720 because that is the smallest frame in which a
#: 0.4-scale Hershey line is still legible on a laptop at a normal
#: viewing distance, and it keeps an MP4 of a few hundred cycles small.
HUD_SIZE = (1280, 720)

#: Width of the side panel, pixels. Wide enough for the longest line the
#: panel writes ("base x +0.000 y +0.000 z +0.000 m") at the panel font.
PANEL_W = 380

#: BGR, the way OpenCV takes them.
BG = (26, 24, 22)               # panel ground, near-black so text carries
FG = (238, 238, 238)
DIM = (150, 148, 145)
RULE = (70, 66, 62)
OK = (120, 220, 120)            # picking
WARN = (60, 190, 250)           # rescanning, shaking, retrying
BAD = (70, 70, 240)             # fault, or a service that is down

#: Predicted instances, in visualize.py's mask colour, so this overlay
#: and overlays_test/ read the same.
POSE_COLOUR = (0, 255, 255)     # yellow
#: The instance the cell chose. Magenta because nothing in this scene --
#: orange parts, a grey tray, yellow silhouettes -- is anywhere near it.
CHOSEN_COLOUR = (255, 0, 255)
FILL_ALPHA = 0.25               # visualize.py's

#: Pose axis length, metres, and line weight: visualize.py's numbers, so
#: the axes are the same size on the same part.
AXIS_LENGTH = 0.015
AXIS_THICKNESS = 3

#: Drawn length of the approach arrow, metres. Every grasp in
#: deploy/pick/grasps.part.json declares a 50 mm approach standoff, and
#: the dict a candidate serialises does not carry it, so the arrow states
#: that standoff rather than inventing a length.
APPROACH_DRAW_M = 0.05

#: Panel typography. One font, three sizes: the panel has to stay
#: readable after an MP4 encoder has been over it.
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 0.62
BODY_SCALE = 0.44
SMALL_SCALE = 0.38
LINE_H = 20

#: Where the board identity is read from inside a container.
#: /proc/device-tree is not visible in one, so the Jetson runbook
#: bind-mounts the model file to this path; the same path is read here so
#: a HUD rendered on the board says Jetson Nano and not "aarch64".
CONTAINER_DEVICE_MODEL = "/etc/device-tree-model"

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


class PartSilhouette:
    """The posed CAD rasterised into the image, as the scorer does it.

    ``score.silhouette`` is imported here rather than at module import so
    that the failure names the file a deployment forgot to ship, instead
    of breaking the import of a module a caller may only want the panel
    from.
    """

    def __init__(self, model_path: str):
        # The scoring script sits at the repository root, which is on the
        # path already when this package was imported from there; the
        # insert is for a caller that started somewhere else.
        sys.path.insert(0, _REPO_ROOT)
        try:
            import score as scorer     # the released scoring script
            import trimesh
        except ImportError as exc:
            raise RuntimeError(
                "the HUD draws silhouettes with score.silhouette and reads "
                "the CAD with trimesh; neither is importable here (%s)"
                % exc)
        mesh = trimesh.load(model_path, force="mesh")
        if mesh.vertices is None or not len(mesh.vertices):
            raise RuntimeError("CAD %r has no vertices" % model_path)
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(mesh.faces)
        self._render = scorer.silhouette

    def mask(self, R: np.ndarray, t: np.ndarray, K: np.ndarray,
             shape: Tuple[int, int]) -> np.ndarray:
        """Front faces of the posed model, 1 inside, 0 outside."""
        return self._render(self.vertices, self.faces, R, t, K, shape)


@dataclass
class HudFrame:
    """One cycle in the plain types the renderer draws from.

    Both a live cycle and a line of a saved ``cycles.jsonl`` reduce to
    this, so the drawing code has one input shape and the
    ``--annotate-only`` path cannot drift from the live one.
    """

    rgb: np.ndarray                             # (H, W, 3) BGR
    K: np.ndarray                               # (3, 3)
    poses: List[Dict[str, Any]] = field(default_factory=list)
    #: ``GraspCandidate.to_dict()`` for the grasp the cell chose.
    grasp: Optional[Dict[str, Any]] = None
    scene: str = ""
    source: str = ""
    frame_id: int = 0
    cycle: int = 0
    bin_index: int = 1
    n_proposals: int = 0
    gate: str = ""
    state: str = ""
    reason: str = ""
    loop_state: str = LOOP_OK
    stage_ms: Dict[str, float] = field(default_factory=dict)
    service_ms: Dict[str, float] = field(default_factory=dict)
    #: The policy's counters for this bin, and the drift monitor's
    #: verdict -- what the cell believes about itself, not about the frame.
    history: Dict[str, int] = field(default_factory=dict)
    drift: str = ""
    #: Digest of the pose service configuration that produced the poses.
    #: On the frame because a pick has to be traceable to a model and a
    #: threshold months later (deploy/ARCHITECTURE.md).
    config_digest: str = ""
    #: What the cell's success sensor reported after a commanded pick,
    #: or None when nothing was commanded. Drawn, not assumed: a green
    #: PICK next to a rising counter reads as a part in the gripper.
    outcome: Optional[str] = None
    error: Optional[str] = None

    @property
    def chosen_index(self) -> Optional[int]:
        """Which pose the chosen grasp hangs off, if any."""
        if not self.grasp:
            return None
        return int(self.grasp.get("pose_index", -1))

    @property
    def top_score(self) -> Optional[float]:
        return float(self.poses[0]["score"]) if self.poses else None

    @classmethod
    def from_cycle(cls, record: Any) -> "HudFrame":
        """Adapt a :class:`deploy.pick.runner.CycleRecord`."""
        result, action = record.result, record.action
        return cls(
            rgb=record.frame.rgb, K=record.frame.K,
            poses=[] if result is None else [p.to_dict()
                                             for p in result.poses],
            grasp=None if record.chosen is None else record.chosen.to_dict(),
            scene=record.scene_id, source=record.frame.source_name,
            frame_id=record.frame.frame_id, cycle=record.cycle,
            bin_index=record.bin_index,
            n_proposals=0 if result is None else result.n_proposals,
            gate="" if result is None else result.gate,
            state="" if action is None else action.state,
            reason="" if action is None else action.reason,
            loop_state=record.loop_state, stage_ms=dict(record.stage_ms),
            service_ms={} if result is None else dict(result.timings_ms),
            history=dict(record.history),
            drift="" if action is None else action.drift_verdict,
            config_digest="" if result is None else result.config_digest,
            outcome=record.outcome, error=record.error)

    @classmethod
    def from_json(cls, entry: Dict[str, Any], rgb: np.ndarray) -> "HudFrame":
        """Rebuild from one line of ``cycles.jsonl`` plus its pixels.

        The log carries the poses and the intrinsics, so re-rendering a
        long board run needs the frames back but not the pipeline.
        """
        frame = entry.get("frame") or {}
        return cls(
            rgb=rgb, K=np.asarray(frame.get("K"), dtype=np.float64),
            poses=list(entry.get("poses") or []), grasp=entry.get("grasp"),
            scene=str(entry.get("scene", "")),
            source=str(frame.get("source", "")),
            frame_id=int(frame.get("frame_id", 0)),
            cycle=int(entry.get("cycle", 0)),
            bin_index=int(entry.get("bin", 1)),
            n_proposals=int(entry.get("n_proposals", 0)),
            gate=str(entry.get("gate") or ""),
            state=str(entry.get("state") or ""),
            reason=str(entry.get("reason") or ""),
            loop_state=str(entry.get("loop", LOOP_OK)),
            stage_ms=dict(entry.get("stage_ms") or {}),
            service_ms=dict(entry.get("service_ms") or {}),
            history=dict(entry.get("history") or {}),
            drift=str(entry.get("drift") or ""),
            config_digest=str(entry.get("config_digest") or ""),
            outcome=entry.get("outcome"), error=entry.get("error"))


class FrameHud:
    """Renders cycles at a fixed size. Loads the CAD once.

    Stateless per frame: two renders of the same cycle produce the same
    pixels, which is what lets ``--annotate-only`` re-render a run
    without anyone wondering whether the picture changed.
    """

    def __init__(self, model_path: str, hardware: str = "",
                 size: Tuple[int, int] = HUD_SIZE, panel_w: int = PANEL_W,
                 accept_score: float = 0.7):
        """Args:
            model_path: The CAD, metres, object frame.
            hardware: The line naming what ran this -- see
                :func:`hardware_line`. Empty draws no hardware line,
                which is worse than a wrong one only in that it says
                nothing.
            size: ``(width, height)`` of the rendered frame.
            panel_w: Width of the side panel inside ``size``.
            accept_score: The cell's gate, drawn on the score bar.
        """
        self.silhouette = PartSilhouette(model_path)
        self.hardware = hardware
        self.width, self.height = int(size[0]), int(size[1])
        self.panel_w = int(panel_w)
        self.accept_score = float(accept_score)

    def render(self, hud_frame: HudFrame) -> np.ndarray:
        """One (height, width, 3) BGR image, ready for a video writer."""
        canvas = np.full((self.height, self.width, 3), BG, dtype=np.uint8)
        image = self.draw_overlay(hud_frame)
        _paste_fit(canvas, image, 0, 0, self.width - self.panel_w,
                   self.height)
        self._draw_panel(canvas, hud_frame)
        return canvas

    # -- the image -------------------------------------------------------

    def draw_overlay(self, hud_frame: HudFrame) -> np.ndarray:
        """The colour frame with every pose on it, chosen one distinct."""
        canvas = np.ascontiguousarray(hud_frame.rgb).copy()
        shape = canvas.shape[:2]
        K = np.asarray(hud_frame.K, dtype=np.float64)
        chosen = hud_frame.chosen_index
        # Chosen last so its outline is never buried under a neighbour's.
        order = [i for i in range(len(hud_frame.poses)) if i != chosen]
        order += [i for i in (chosen,) if i is not None
                  and 0 <= i < len(hud_frame.poses)]
        for index in order:
            pose = hud_frame.poses[index]
            colour = CHOSEN_COLOUR if index == chosen else POSE_COLOUR
            R = np.asarray(pose["R"], dtype=np.float64)
            t = np.asarray(pose["t"], dtype=np.float64)
            self._draw_pose(canvas, R, t, K, shape, colour,
                            index == chosen)
            _label(canvas, R, t, K, "%d  %.2f" % (index,
                                                  float(pose["score"])))
        if hud_frame.grasp:
            self._draw_grasp(canvas, hud_frame.grasp, K)
        return canvas

    def _draw_pose(self, canvas: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray, shape: Tuple[int, int],
                   colour: Sequence[int], chosen: bool) -> None:
        mask = self.silhouette.mask(R, t, K, shape) > 0
        if mask.any():
            canvas[mask] = ((1 - FILL_ALPHA) * canvas[mask]
                            + FILL_ALPHA * np.array(colour)).astype(np.uint8)
            cv2.drawContours(canvas, _contours(mask), -1, colour,
                             3 if chosen else 1)
        # visualize.py's convention exactly: rvec/tvec straight from the
        # pose, zero distortion (camera.json states none for this data).
        rvec, _ = cv2.Rodrigues(R)
        cv2.drawFrameAxes(canvas, K, np.zeros(5), rvec, t, AXIS_LENGTH,
                          AXIS_THICKNESS)

    def _draw_grasp(self, canvas: np.ndarray, grasp: Dict[str, Any],
                    K: np.ndarray) -> None:
        """The tool pose: where the TCP goes and which way it comes in."""
        T = np.asarray(grasp["T_camera_grasp"], dtype=np.float64)
        contact = T[:3, 3]
        approach = np.asarray(grasp.get("approach_camera", T[:3, 2]),
                              dtype=np.float64)
        start = _project(contact - approach * APPROACH_DRAW_M, K)
        end = _project(contact, K)
        if start is not None and end is not None:
            cv2.arrowedLine(canvas, start, end, (0, 0, 0), 6, cv2.LINE_AA,
                            tipLength=0.25)
            cv2.arrowedLine(canvas, start, end, CHOSEN_COLOUR, 3,
                            cv2.LINE_AA, tipLength=0.25)
        if end is not None:
            cv2.circle(canvas, end, 11, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(canvas, end, 11, CHOSEN_COLOUR, 2, cv2.LINE_AA)
            cv2.circle(canvas, end, 2, CHOSEN_COLOUR, -1, cv2.LINE_AA)

    # -- the panel -------------------------------------------------------

    def _draw_panel(self, canvas: np.ndarray, f: HudFrame) -> None:
        x0 = self.width - self.panel_w
        cv2.rectangle(canvas, (x0, 0), (self.width, self.height), BG, -1)
        cv2.line(canvas, (x0, 0), (x0, self.height), RULE, 1)
        panel = _Panel(canvas, x0 + 16, self.panel_w - 32, 34)

        panel.line("cycle %d  bin %d" % (f.cycle, f.bin_index), FG,
                   TITLE_SCALE)
        panel.line("scene %s   frame %d" % (f.scene or "-", f.frame_id))
        panel.wrap(f.source or "-", DIM, SMALL_SCALE)

        if f.loop_state != LOOP_OK:
            panel.rule()
            panel.line(f.loop_state.upper().replace("_", " "), BAD,
                       TITLE_SCALE)
            panel.wrap(f.error or "", DIM, SMALL_SCALE, limit=4)
            self._draw_hardware(panel)
            return

        panel.rule()
        panel.line("VISION", DIM, SMALL_SCALE)
        panel.line("%d pose(s) from %d proposal(s)" % (len(f.poses),
                                                       f.n_proposals))
        top = f.top_score
        # Two thresholds, and they are not the same one: ``gate`` is the
        # pose service's verdict at its own accept score, the tick on the
        # bar is this cell's. Colouring the bar by the service's would
        # paint a green bar beside a RESCAN every time the cell is set
        # stricter than the service it calls.
        clears = top is not None and top >= self.accept_score
        panel.line("top score %s   service %s"
                   % ("-" if top is None else "%.3f" % top,
                      (f.gate or "-").upper()),
                   OK if f.gate == GATE_PICK else WARN)
        panel.score_bar(top, self.accept_score, OK if clears else WARN)
        if f.error:
            # The service answered and said the frame failed. Without
            # this the panel shows an empty pose list, which is what a
            # bin with nothing in it looks like.
            panel.wrap("service error: %s" % f.error, BAD, SMALL_SCALE,
                       limit=2)

        panel.rule()
        panel.line("DECISION", DIM, SMALL_SCALE)
        panel.line((f.state or "-").upper(), _state_colour(f.state),
                   TITLE_SCALE)
        panel.wrap(f.reason, DIM, SMALL_SCALE, limit=4)

        if f.grasp:
            grasp = f.grasp
            panel.rule()
            panel.line("GRASP  %s" % grasp.get("grasp", "-"), CHOSEN_COLOUR)
            clearance = grasp.get("clearance_m")
            panel.line("%s  pose %d  rank %.2f  clear %s"
                       % (grasp.get("type", "-"),
                          int(grasp.get("pose_index", 0)),
                          float(grasp.get("rank", 0.0)),
                          "free" if clearance is None
                          else "%.0f mm" % (1000.0 * clearance)),
                       DIM, SMALL_SCALE)
            panel.line(_xyz("cam ", grasp.get("T_camera_grasp")))
            base = grasp.get("T_base_grasp")
            # Only with a hand-eye file: without one the base frame is not
            # unknown, it is undefined, and a zero would read as a number.
            panel.line(_xyz("base", base) if base else
                       "base  no hand-eye given", FG if base else DIM,
                       BODY_SCALE if base else SMALL_SCALE)

        panel.rule()
        panel.line("LATENCY  %.0f ms cycle" % f.stage_ms.get("cycle_ms", 0.0),
                   DIM, SMALL_SCALE)
        panel.bars(_latency_rows(f))

        panel.rule()
        panel.line("CELL", DIM, SMALL_SCALE)
        for row in _counters(f):
            panel.line(row, FG, SMALL_SCALE)
        if f.outcome:
            panel.line(_outcome(f.outcome), DIM, SMALL_SCALE)
        # Provenance, the way every response carries it: which model and
        # which thresholds produced this pose, months from now.
        panel.line("drift %s   config %s" % (f.drift or "-",
                                             f.config_digest or "-"),
                   DIM, SMALL_SCALE)
        self._draw_hardware(panel)

    def _draw_hardware(self, panel: "_Panel") -> None:
        if not self.hardware:
            return
        panel.to_bottom(len(self.hardware.split("\n")) * LINE_H + 10)
        panel.rule()
        for line in self.hardware.split("\n"):
            panel.line(line, DIM, SMALL_SCALE)


class _Panel:
    """A text cursor down the side panel. Draws, then moves down."""

    def __init__(self, canvas: np.ndarray, x: int, width: int, y: int):
        self.canvas = canvas
        self.x = x
        self.width = width
        self.y = y

    def line(self, text: str, colour: Sequence[int] = FG,
             scale: float = BODY_SCALE) -> None:
        cv2.putText(self.canvas, text, (self.x, self.y), FONT, scale,
                    colour, 1, cv2.LINE_AA)
        self.y += LINE_H + (6 if scale >= TITLE_SCALE else 0)

    def wrap(self, text: str, colour: Sequence[int] = DIM,
             scale: float = SMALL_SCALE, limit: int = 3) -> None:
        """Break on spaces to the panel width, at most ``limit`` lines."""
        budget = max(8, int(self.width / (scale * 19.0)))
        words, row, rows = text.split(), "", []
        for word in words:
            candidate = (row + " " + word).strip()
            if len(candidate) > budget and row:
                rows.append(row)
                row = word
            else:
                row = candidate
        if row:
            rows.append(row)
        for row in rows[:limit]:
            self.line(row, colour, scale)

    def rule(self) -> None:
        self.y += 4
        cv2.line(self.canvas, (self.x, self.y),
                 (self.x + self.width, self.y), RULE, 1)
        self.y += 16

    def score_bar(self, score: Optional[float], gate: float,
                  colour: Sequence[int]) -> None:
        """The score against the gate, so "0.68" is visibly a rescan."""
        top, height = self.y - 10, 8
        cv2.rectangle(self.canvas, (self.x, top),
                      (self.x + self.width, top + height), RULE, -1)
        if score:
            filled = int(self.width * min(1.0, max(0.0, score)))
            cv2.rectangle(self.canvas, (self.x, top),
                          (self.x + filled, top + height), colour, -1)
        at = self.x + int(self.width * gate)
        cv2.line(self.canvas, (at, top - 3), (at, top + height + 3), FG, 1)
        self.y += LINE_H

    def bars(self, rows: Sequence[Tuple[str, float]]) -> None:
        """Named milliseconds as bars, all scaled to the largest."""
        if not rows:
            return
        longest = max(value for _, value in rows) or 1.0
        for name, value in rows:
            cv2.putText(self.canvas, name, (self.x, self.y), FONT,
                        SMALL_SCALE, DIM, 1, cv2.LINE_AA)
            left, right = self.x + 74, self.x + self.width - 56
            filled = int((right - left) * value / longest)
            cv2.rectangle(self.canvas, (left, self.y - 8),
                          (left + max(1, filled), self.y - 1), FG, -1)
            cv2.putText(self.canvas, "%6.0f" % value,
                        (right + 4, self.y), FONT, SMALL_SCALE, DIM, 1,
                        cv2.LINE_AA)
            self.y += 16

    def to_bottom(self, height: int) -> None:
        """Jump the cursor to a block of ``height`` at the panel's foot."""
        self.y = max(self.y, self.canvas.shape[0] - height)


# -- hardware ------------------------------------------------------------

def hardware_line(bench_path: Optional[str] = None,
                  override: Optional[str] = None) -> str:
    """What ran this, in two lines: the machine, then its size.

    A demo frame that does not say which machine produced its latencies
    is a demo frame that will be quoted about the wrong machine. The
    order of preference is the order of trust: what the caller states,
    then a bench fingerprint recorded on the machine that ran
    (``results/bench/*.json``), then this host as it describes itself.

    Args:
        bench_path: A bench record whose ``host`` block describes the
            machine the numbers came from.
        override: Text to use verbatim, e.g. for a run whose frames were
            estimated somewhere other than where they are being drawn.
    """
    if override:
        return override
    host = None                     # type: Optional[Dict[str, Any]]
    if bench_path:
        with open(bench_path) as handle:
            host = json.load(handle).get("host")
    if host is None:
        host = _local_host()
    name = host.get("device_tree_model") or host.get("cpu_model") \
        or host.get("machine") or "unknown machine"
    parts = [str(host.get("machine") or platform.machine())]
    cores = host.get("cpu_affinity") or host.get("cpu_count")
    if cores:
        parts.append("%d cores" % int(cores))
    if host.get("total_ram_mb"):
        parts.append("%d MB" % int(host["total_ram_mb"]))
    return "%s\n%s" % (str(name).strip()[:44], "  ".join(parts))


def _local_host() -> Dict[str, Any]:
    """This machine, probed the way deploy/board/bench.py probes it.

    The same two files decide it: a Tegra is the only thing that puts a
    model name in /proc/device-tree/model, and the runbook bind-mounts
    that name into a container as CONTAINER_DEVICE_MODEL.
    """
    model = _read_text("/proc/device-tree/model") \
        or _read_text(CONTAINER_DEVICE_MODEL)
    return {"device_tree_model": model.strip("\x00").strip() if model
            else None,
            "cpu_model": _cpu_model(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
            "total_ram_mb": _total_ram_mb()}


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as handle:
            return handle.read()
    except (IOError, OSError, UnicodeDecodeError):
        return None


def _cpu_model() -> Optional[str]:
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    for key in ("model name", "Model", "Hardware"):
        found = re.search(r"^%s\s*:\s*(.+)$" % key, cpuinfo, re.M)
        if found:
            return found.group(1).strip()
    return None


def _total_ram_mb() -> Optional[int]:
    found = re.search(r"^MemTotal:\s+(\d+) kB",
                      _read_text("/proc/meminfo") or "", re.M)
    return int(found.group(1)) // 1024 if found else None


# -- drawing helpers -----------------------------------------------------

def _latency_rows(f: HudFrame) -> List[Tuple[str, float]]:
    """The loop's stages, with the service's own split folded in.

    ``estimate`` is the round trip; ``segment`` and ``register`` are the
    service's report of what it spent inside it, indented because they
    are a decomposition of the row above and not additional time.
    """
    rows = [("camera", f.stage_ms.get("camera_ms", 0.0)),
            ("decode", f.stage_ms.get("decode_ms", 0.0)),
            ("estimate", f.stage_ms.get("estimate_ms", 0.0))]
    for name, key in ((" segment", "segment_ms"),
                      (" register", "register_ms")):
        if key in f.service_ms:
            rows.append((name, float(f.service_ms[key])))
    rows.append(("plan", f.stage_ms.get("plan_ms", 0.0)))
    return rows


def _counters(f: HudFrame) -> List[str]:
    """The policy's bin counters, and what each is counted since.

    ``picks`` runs for the whole bin; the three the policy escalates on
    are reset the moment the cell makes progress. Lining all four up on
    one row made "rescans 0" appear directly after two rescans, so the
    reset ones are labelled with what resets them.
    """
    history = f.history
    return ["picks %d this bin" % history.get("picks", 0),
            "since progress: %d rescan  %d shake  %d fail"
            % (history.get("rescans_since_progress", 0),
               history.get("shakes_since_progress", 0),
               history.get("consecutive_failures", 0))]


def _outcome(outcome: str) -> str:
    """What the cell's success sensor said about the pick just commanded.

    ``unknown`` is the honest answer from a loop with no robot behind it
    (deploy/pick/runner.py) and it belongs on the frame: a green PICK
    beside a rising pick counter otherwise reads as a part in the gripper.
    """
    if outcome == OUTCOME_UNKNOWN:
        return "pick commanded, outcome unknown (no success sensor)"
    return "pick outcome %s" % outcome


def _state_colour(state: str) -> Sequence[int]:
    """Green picks, red stops, grey is done, amber is everything the cell
    does instead of picking."""
    if state == PICK:
        return OK
    if state == FAULT:
        return BAD
    if state == EMPTY:
        return DIM
    return WARN


def _xyz(label: str, matrix: Optional[Sequence[Sequence[float]]]) -> str:
    if not matrix:
        return "%s  -" % label
    t = np.asarray(matrix, dtype=np.float64)[:3, 3]
    return "%s x%+.3f y%+.3f z%+.3f m" % (label, t[0], t[1], t[2])


def _project(point: Sequence[float],
             K: np.ndarray) -> Optional[Tuple[int, int]]:
    """Pixel of a camera-frame point, or None if it is behind the camera."""
    p = np.asarray(point, dtype=np.float64)
    if p[2] <= 1e-6:
        return None
    return (int(round(K[0, 0] * p[0] / p[2] + K[0, 2])),
            int(round(K[1, 1] * p[1] / p[2] + K[1, 2])))


def _contours(mask: np.ndarray) -> Sequence[np.ndarray]:
    """Outer and inner contours, so the part's through holes are drawn."""
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                             cv2.CHAIN_APPROX_SIMPLE)
    return found[0] if len(found) == 2 else found[1]


def _label(canvas: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray,
           text: str) -> None:
    """Index and score at the pose origin, outlined so it reads on any
    colour -- visualize.py's trick."""
    at = _project(t, K)
    if at is None:
        return
    at = (at[0] + 10, at[1] - 10)
    cv2.putText(canvas, text, at, FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, at, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _paste_fit(canvas: np.ndarray, image: np.ndarray, x: int, y: int,
               width: int, height: int) -> None:
    """Letterbox ``image`` into the box, keeping its aspect ratio.

    Aspect is kept because the overlay is a measurement: a stretched
    frame would put the axes at angles the pose never had.
    """
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    size = (max(1, int(image.shape[1] * scale)),
            max(1, int(image.shape[0] * scale)))
    # INTER_AREA is the only downscale that does not alias a 1 px contour
    # into a dotted line.
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    at_x = x + (width - size[0]) // 2
    at_y = y + (height - size[1]) // 2
    canvas[at_y:at_y + size[1], at_x:at_x + size[0]] = resized
