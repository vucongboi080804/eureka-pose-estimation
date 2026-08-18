"""Record the cell running: an MP4 of the cycles, and a contact sheet.

Two ways in, one renderer:

    # live -- drives the loop against the two services and records it
    python -m deploy.demo.render_demo --cycles 8 --out out/demo.mp4 \
        --log out/cycles.jsonl --sheet out/sheet.png

    # again, from the log -- no camera, no pose service, no pipeline
    python -m deploy.demo.render_demo --annotate-only \
        --log out/cycles.jsonl --frames . --out out/demo.mp4

``--annotate-only`` is the reason the runner's log carries the poses and
the intrinsics as well as the decision: a long run on the board is
expensive and is done once, and every later version of the overlay is a
re-render costing a few seconds a frame on a laptop. It also means the
video can be checked against the log it claims to show, line by line.

Nothing is smoothed, interpolated or re-ordered. One cycle is one held
still frame, in the order the cycles happened, and ``--hold 0`` holds
each for the wall-clock the cycle actually took -- which is the only
setting in which the video's pace is a measurement rather than a
decision.

Codec: ``mp4v`` (MPEG-4 Part 2) by default, because it is compiled into
every OpenCV build this ships against -- the board's 4.10 included --
whereas ``avc1`` needs an H.264 encoder the opencv-python wheels leave
out and fails by writing an empty file rather than by raising. Pass
``--codec avc1`` on a machine that has one; the output is perhaps three
times smaller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..cell.calibration import CalibrationError
from ..cell.grasp import GraspConfigError
from ..cell.policy import ACCEPT_SCORE, PickPolicy
from ..cell.runner import CellLoop, LOOP_OK, build_planner, load_hand_eye
from .hud import FrameHud, HudFrame, hardware_line

EXIT_OK = 0
EXIT_FAILED = 1

#: Video frame rate. 10 is the lowest rate every player still scrubs
#: smoothly, and a cycle is held for many frames, so a higher rate would
#: only multiply identical frames.
DEFAULT_FPS = 10.0

#: How long one cycle is shown, seconds. 1.5 s is about what it takes to
#: read the panel; 0 means "as long as the cycle took", which turns the
#: video into a timing record.
DEFAULT_HOLD_S = 1.5

#: Distinct outcomes kept in memory for the contact sheet. The pool also
#: holds the most and least confident pick and ``--sheet-n`` fillers, so
#: the ceiling is ``SHEET_POOL + 2 + sheet_n`` rendered frames -- at
#: 1280x720x3 = 2.6 MB each, 53 MB with the defaults, and the same
#: whether the run is ten cycles or a thousand.
SHEET_POOL = 12


class DemoVideo:
    """An MP4 that holds each cycle for a while. Closes to a report.

    Opens the writer on the first frame, when the size is known, and
    refuses to leave an empty file behind: a 0-byte MP4 from a codec that
    was not really there is the failure this class exists to make loud.
    """

    def __init__(self, path: str, fps: float = DEFAULT_FPS,
                 codec: str = "mp4v"):
        self.path = path
        self.fps = float(fps)
        self.codec = codec
        self._writer = None         # type: Optional[cv2.VideoWriter]
        self._stats = None          # type: Optional[Dict[str, Any]]
        self.size = (0, 0)
        self.frames = 0
        self.cycles = 0

    def write(self, image: np.ndarray, hold_s: float) -> None:
        """Add one cycle, repeated for ``hold_s`` of video."""
        if self._writer is None:
            self._open(image.shape[1], image.shape[0])
        for _ in range(max(1, int(round(hold_s * self.fps)))):
            self._writer.write(image)
            self.frames += 1
        self.cycles += 1

    def _open(self, width: int, height: int) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        writer = cv2.VideoWriter(self.path, _fourcc(self.codec), self.fps,
                                 (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV cannot write %r with codec %r; mp4v is the codec "
                "every build has" % (self.path, self.codec))
        self._writer, self.size = writer, (width, height)

    def close(self) -> Dict[str, Any]:
        """Release the writer and describe what was written. Idempotent,
        so a failure path and the report can both call it."""
        if self._stats is not None:
            return self._stats
        if self._writer is None:
            self._stats = {"path": self.path, "frames": 0, "cycles": 0}
            return self._stats
        self._writer.release()
        self._stats = {"path": os.path.abspath(self.path),
                       "cycles": self.cycles,
                       "frames": self.frames, "codec": self.codec,
                       "fps": self.fps, "size": "%dx%d" % self.size,
                       "seconds": round(self.frames / self.fps, 1),
                       "bytes": os.path.getsize(self.path)}
        return self._stats

    def report(self) -> str:
        stats = self.close()
        if not stats["frames"]:
            return "no frames written to %s" % self.path
        return ("%s  %s  %.1f s at %g fps  %d cycles  %.1f MB  (%s)"
                % (stats["path"], stats["size"], stats["seconds"],
                   stats["fps"], stats["cycles"],
                   stats["bytes"] / 1048576.0, stats["codec"]))


class ContactSheet:
    """A bounded pool of the frames a README should show.

    A reader learns from variety, not from the first N frames: one frame
    of every distinct outcome the run produced (each policy state, each
    way the loop went down), then the most and least confident picks,
    then evenly spaced fillers. The pool is capped so a thousand-cycle
    run costs the same memory as a ten-cycle one.
    """

    def __init__(self, count: int = 6, cols: int = 3):
        self.count = max(1, int(count))
        self.cols = max(1, int(cols))
        #: kind -> (cycle, image); (cycle, image, score) for the picks.
        self._first = OrderedDict()     # type: Dict[str, Tuple[int, Any]]
        self._best = None               # type: Optional[Tuple]
        self._worst = None              # type: Optional[Tuple]
        self._fillers = []              # type: List[Tuple[int, Any]]
        self._seen = 0

    def offer(self, hud_frame: HudFrame, image: np.ndarray) -> None:
        self._seen += 1
        kind = (hud_frame.loop_state if hud_frame.loop_state != LOOP_OK
                else hud_frame.state or "unknown")
        if kind not in self._first and len(self._first) < SHEET_POOL:
            self._first[kind] = (hud_frame.cycle, image)
        score = hud_frame.top_score
        if hud_frame.grasp and score is not None:
            if self._best is None or score > self._best[2]:
                self._best = (hud_frame.cycle, image, score)
            if self._worst is None or score < self._worst[2]:
                self._worst = (hud_frame.cycle, image, score)
        if len(self._fillers) < self.count and self._seen % 3 == 1:
            self._fillers.append((hud_frame.cycle, image))

    def save(self, path: str) -> Optional[str]:
        """Write the grid; returns the line to print, or None if empty."""
        chosen = OrderedDict()          # cycle -> image, insertion ordered
        for source in (list(self._first.values()),
                       [p[:2] for p in (self._best, self._worst) if p],
                       self._fillers):
            for cycle, image in source:
                if len(chosen) < self.count:
                    chosen.setdefault(cycle, image)
        if not chosen:
            return None
        tiles = [chosen[c] for c in sorted(chosen)]
        rows = (len(tiles) + self.cols - 1) // self.cols
        cell_w = tiles[0].shape[1] // 2
        cell_h = tiles[0].shape[0] // 2
        sheet = np.zeros((rows * cell_h, self.cols * cell_w, 3), np.uint8)
        for index, tile in enumerate(tiles):
            top = (index // self.cols) * cell_h
            left = (index % self.cols) * cell_w
            sheet[top:top + cell_h, left:left + cell_w] = cv2.resize(
                tile, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(sheet, (left, top),
                          (left + cell_w - 1, top + cell_h - 1),
                          (60, 58, 55), 1)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        cv2.imwrite(path, sheet)
        return ("%s  %dx%d  %d frame(s), cycles %s"
                % (os.path.abspath(path), sheet.shape[1], sheet.shape[0],
                   len(tiles), ", ".join(str(c) for c in sorted(chosen))))


# -- the two sources of cycles -------------------------------------------

def live_cycles(args: argparse.Namespace
                ) -> Iterator[Tuple[HudFrame, Dict[str, Any]]]:
    """Run the loop against the services and yield what it produced.

    The loop is the runner's, unmodified: this module renders cycles, it
    does not have a second opinion about how one is run.
    """
    planner = build_planner(args.model, args.grasps)
    hand_eye = load_hand_eye(args.hand_eye) if args.hand_eye else None
    loop = CellLoop(args.camera, args.pose, planner,
                    PickPolicy(accept_score=args.accept_score),
                    hand_eye=hand_eye)
    sink = open(args.log, "a") if args.log else None
    try:
        for record in loop.run(args.cycles):
            entry = record.to_json()
            if sink is not None:
                sink.write(json.dumps(entry) + "\n")
                sink.flush()
            if record.frame is None:
                # The camera never answered, so there are no pixels to
                # draw. It is in the log; the video simply has no frame
                # for it.
                print("cycle %d: %s" % (record.cycle, record.error))
                continue
            yield HudFrame.from_cycle(record), entry
    finally:
        if sink is not None:
            sink.close()


def logged_cycles(args: argparse.Namespace
                  ) -> Iterator[Tuple[HudFrame, Dict[str, Any]]]:
    """Re-read a run from its log, pairing each line with its pixels."""
    missing = 0
    with open(args.log) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError as exc:
                print("%s:%d is not JSON: %s" % (args.log, number, exc))
                continue
            if args.scenes and entry.get("scene") not in args.scenes:
                continue
            rgb = _load_frame_pixels(args.frames, entry)
            if rgb is None:
                # A log line whose frame is not where the log says is a
                # skipped frame, not a failed render.
                missing += 1
                continue
            yield HudFrame.from_json(entry, rgb), entry
    if missing:
        print("%d cycle(s) skipped: no rgb.png under %s"
              % (missing, os.path.abspath(args.frames)))


def _load_frame_pixels(root: str,
                       entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """The colour frame a log line refers to, from a folder of scenes.

    A replay source names itself ``scene_folder:test/000003``; the part
    after the colon is the path under ``root``. The scene id alone is
    tried next, which is what makes a run served from a recorded session
    (``session:test40/000003``) re-renderable against the split it was
    recorded off: ``--frames test``.
    """
    frame = entry.get("frame") or {}
    source = str(frame.get("source", ""))
    relative = source.split(":", 1)[1] if ":" in source else source
    for candidate in (relative, str(entry.get("scene", ""))):
        if not candidate:
            continue
        path = os.path.join(root, candidate, "rgb.png")
        image = cv2.imread(path)
        if image is not None:
            return image
    return None


def _hardware(args: argparse.Namespace) -> str:
    """The panel's hardware line: the machine that ESTIMATED the frames.

    Live, that is this host, and probing it is right. Re-rendering a log
    it is whatever ran the services then, which the log does not record
    -- so rather than stamp the laptop doing the re-render onto a board's
    latencies, the panel says the machine is not known and names the two
    flags that would say it.
    """
    if args.hardware or args.bench:
        return hardware_line(args.bench, args.hardware)
    if args.annotate_only:
        return ("re-rendered from %s\nmachine not recorded -- pass --bench "
                "or --hardware" % os.path.basename(args.log))
    return hardware_line()


# -- entry point ---------------------------------------------------------

def _fourcc(codec: str) -> int:
    """FourCC across OpenCV majors: 4.x exposes the free function, 5.x
    also hangs it off the class."""
    factory = getattr(cv2, "VideoWriter_fourcc", None) \
        or cv2.VideoWriter.fourcc
    return int(factory(*codec))


def main(argv: Optional[list] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo.mp4", help="MP4 to write")
    parser.add_argument("--log", default=None,
                        help="cycles.jsonl: written in live mode, read in "
                             "--annotate-only")
    parser.add_argument("--sheet", default=None,
                        help="Also write a contact sheet PNG here")
    parser.add_argument("--sheet-n", type=int, default=6)
    parser.add_argument("--sheet-cols", type=int, default=3)
    parser.add_argument("--annotate-only", action="store_true",
                        help="Re-render from --log instead of running the "
                             "pipeline again")
    parser.add_argument("--frames", default=root,
                        help="Root under which --annotate-only finds the "
                             "frames the log names. A folder-replayed run "
                             "re-renders from the release root; a run served "
                             "from a recorded session re-renders from the "
                             "split it was recorded off, because the log "
                             "keeps the scene id (default: %(default)s)")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Only these scene ids. --annotate-only only: "
                             "live, the camera service decides which scenes "
                             "it serves (CAM_SCENES)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD_S,
                        help="Seconds each cycle is shown; 0 holds it for "
                             "the wall-clock the cycle took")
    parser.add_argument("--codec", default="mp4v",
                        help="FourCC for cv2.VideoWriter (default: "
                             "%(default)s)")
    parser.add_argument("--camera", default="http://127.0.0.1:8081")
    parser.add_argument("--pose", default="http://127.0.0.1:8080")
    parser.add_argument("--cycles", type=int, default=10,
                        help="Cycles to run in live mode; 0 runs until "
                             "interrupted")
    parser.add_argument("--model", default=os.path.join(root, "model",
                                                        "3d_model.ply"))
    parser.add_argument("--grasps", default=None,
                        help="Grasp definition file (default: the one "
                             "shipped beside deploy/cell)")
    parser.add_argument("--hand-eye", default=None)
    parser.add_argument("--accept-score", type=float, default=ACCEPT_SCORE)
    parser.add_argument("--hardware", default=None,
                        help="Text for the panel's hardware line; it must "
                             "name the machine that ESTIMATED the frames, "
                             "which is not this one when re-rendering a "
                             "board run")
    parser.add_argument("--bench", default=None,
                        help="Bench record whose host block names that "
                             "machine (results/bench/*.json)")
    args = parser.parse_args(argv)
    if args.grasps is None:
        from ..cell.grasp import default_grasps_path
        args.grasps = default_grasps_path()
    if args.annotate_only and not args.log:
        print("--annotate-only needs the --log it re-renders",
              file=sys.stderr)
        return EXIT_FAILED

    try:
        hud = FrameHud(args.model, hardware=_hardware(args),
                       accept_score=args.accept_score)
    except (RuntimeError, OSError) as exc:
        print("cannot render: %s" % exc, file=sys.stderr)
        return EXIT_FAILED

    video = DemoVideo(args.out, args.fps, args.codec)
    sheet = ContactSheet(args.sheet_n, args.sheet_cols) if args.sheet else None
    try:
        cycles = (logged_cycles(args) if args.annotate_only
                  else live_cycles(args))
        for hud_frame, entry in cycles:
            image = hud.render(hud_frame)
            hold = args.hold if args.hold > 0 else \
                hud_frame.stage_ms.get("cycle_ms", 1000.0) / 1000.0
            video.write(image, hold)
            if sheet is not None:
                sheet.offer(hud_frame, image)
            print("cycle %-4d %-10s %-6s %-6s %6.0f ms  ->  video %.1f s"
                  % (hud_frame.cycle, hud_frame.scene or "-",
                     hud_frame.gate or hud_frame.loop_state,
                     hud_frame.state or "-",
                     hud_frame.stage_ms.get("cycle_ms", 0.0),
                     video.frames / video.fps), flush=True)
    except (GraspConfigError, CalibrationError, RuntimeError, OSError) as exc:
        print("render failed: %s" % exc, file=sys.stderr)
        video.close()
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\ninterrupted; closing the video")

    print(video.report())
    if sheet is not None:
        line = sheet.save(args.sheet)
        print(line or "no frames for the contact sheet")
    # A render that drew nothing -- every cycle lost the camera, or the
    # log named frames that are not under --frames -- is a failure, not
    # an empty success: a CI job or an operator has to be able to tell.
    return EXIT_OK if video.cycles else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
