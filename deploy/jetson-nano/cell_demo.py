"""The whole cell in one process, for the board.

`deploy/ARCHITECTURE.md` splits the cell into three processes -- camera,
vision, decision -- because that is how a real line is built: the camera is
a device, the model is a service, the controller owns motion, and a failure
in one is attributable in the logs of one. That split costs a loopback hop
and a second Python interpreter, which is nothing on a workstation.

On a 4 GB board it is not nothing, and for a demonstration it is not
useful. This is the same pipeline with the seams removed: one interpreter,
one model load, frames pulled straight from a recorded session or a scene
folder, and every cycle drawn to a video as it happens. It is the file to
copy onto a Jetson when the question is "does it work here", and it shares
its estimator, planner, policy and renderer with the service path rather
than reimplementing them, so the two cannot drift.

    python deploy/jetson-nano/cell_demo.py --root /data --split test \
        --seg-model weights/part-seg-nano.pt --imgsz 640 --out /out/demo

Reads a recorded session instead with `--session sessions/test40`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from deploy.cell.grasp import GraspPlanner, default_grasps_path
from deploy.cell.policy import CellFrame, PickHistory, PickPolicy
from deploy.demo.hud import FrameHud, HudFrame, hardware_line
from src.detect import part_pixel_mask
from src.detect_seg import detect_scene_hybrid
from src.model_cloud import load_model_cloud
from src.register import PoseEstimator
from src.scene_io import Scene, list_scenes, load_scene

#: Frames per second of the rendered video. The board takes seconds per
#: cycle, so this is a playback rate, not a capture rate -- one cycle is
#: one frame and the panel carries the real latency.
VIDEO_FPS = 2.0


class Capture:
    """Frames from a recorded session or a release folder, one interface.

    A session is what a cell records in production; a folder is what a
    release ships. The loop should not care which it is looking at.
    """

    def __init__(self, root: str = ".", split: str = "test",
                 scenes: Optional[List[str]] = None,
                 session: Optional[str] = None):
        self.session_path = session
        self.root, self.split = root, split
        self._reader = None
        if session:
            from deploy.camera_service.session import SessionReader
            self._reader = SessionReader(session).open()
            self.ids = [f["scene_id"] for f in self._reader.records]
            self.source = "session:%s" % os.path.basename(session.rstrip("/"))
        else:
            self.ids = scenes or list_scenes(root, split)
            self.source = "folder:%s" % split

    def __len__(self) -> int:
        return len(self.ids)

    def frames(self) -> Iterator[Tuple[str, Scene, np.ndarray]]:
        """Yield (scene_id, Scene, rgb) in capture order."""
        for i, scene_id in enumerate(self.ids):
            if self._reader is not None:
                rec = self._reader.read(i)
                depth = rec.depth_raw.astype(np.float32) * rec.depth_scale
                scene = Scene(scene_id=rec.scene_id, rgb=rec.rgb,
                              depth=depth, K=np.asarray(rec.K, float))
            else:
                scene = load_scene(self.root, self.split, scene_id)
            yield scene_id, scene, scene.rgb

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()


def _stage(timings: Dict[str, float], name: str, t0: float) -> float:
    """Record a stage's wall time and return the new mark."""
    now = time.perf_counter()
    timings[name] = round((now - t0) * 1000.0, 1)
    return now


def run(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    hardware = hardware_line(args.bench, args.hardware)

    load0 = time.perf_counter()
    from ultralytics import YOLO
    cloud = load_model_cloud(args.cad)
    seg = YOLO(args.seg_model)
    extra = YOLO(args.extra_seg_model) if args.extra_seg_model else None
    planner = GraspPlanner(args.cad, args.grasps or default_grasps_path())
    policy = PickPolicy(accept_score=args.accept_score)
    hud = FrameHud(args.cad, hardware=hardware, accept_score=args.accept_score)
    load_s = time.perf_counter() - load0
    print("loaded in %.1f s  |  %s" % (load_s, hardware.replace("\n", " | ")))

    capture = Capture(args.root, args.split, args.scenes, args.session)
    history = PickHistory()
    writer = None
    log = open(os.path.join(args.out, "cycles.jsonl"), "w")
    picks = rescans = 0
    cycle_times: List[float] = []

    try:
        for cycle, (scene_id, scene, rgb) in enumerate(capture.frames(), 1):
            if args.cycles and cycle > args.cycles:
                break
            t0 = time.perf_counter()
            timings: Dict[str, float] = {}
            error = None
            poses: List[Any] = []
            grasps: List[Any] = []
            try:
                estimator = PoseEstimator(cloud, scene.depth, scene.K,
                                          part_mask=part_pixel_mask(scene.rgb))
                mark = _stage(timings, "prepare", t0)
                poses = detect_scene_hybrid(scene, estimator, seg,
                                            extra_model=extra,
                                            conf=args.conf, pick=args.pick,
                                            imgsz=args.imgsz)
                mark = _stage(timings, "estimate", mark)
                grasps = planner.plan(poses, scene.depth, K=scene.K)
                _stage(timings, "grasp", mark)
            except Exception as exc:                      # one bad frame
                error = "%s: %s" % (type(exc).__name__, exc)

            frame = CellFrame(scene_id=scene_id, poses=poses, grasps=grasps,
                              n_proposals=len(poses), error=error)
            action = policy.next_action(frame, history)
            history.record(action)
            cycle_s = time.perf_counter() - t0
            cycle_times.append(cycle_s)
            picks += action.state == "pick"
            rescans += action.state in ("scan", "rescan")

            chosen = action.grasp
            hf = HudFrame(
                rgb=rgb, K=scene.K,
                poses=[{"R": np.asarray(p.R).tolist(),
                        "t": np.asarray(p.t).tolist(),
                        "score": float(p.submission_score)} for p in poses],
                grasp=None if chosen is None else chosen.to_dict(),
                scene=scene_id, source=capture.source, frame_id=cycle,
                cycle=cycle, n_proposals=len(poses),
                gate="pick" if action.state == "pick" else "rescan",
                state=action.state, reason=action.reason,
                stage_ms=timings, history=history.__dict__.copy(),
                drift=action.drift_verdict, error=error)
            log.write(json.dumps({
                "cycle": cycle, "scene": scene_id, "state": action.state,
                "reason": action.reason, "n_poses": len(poses),
                "top_score": None if not poses else round(
                    float(poses[0].submission_score), 4),
                "cycle_s": round(cycle_s, 3), "stage_ms": timings,
                "error": error}) + "\n")
            log.flush()

            print("%3d  %-7s %-5s %2d poses  top %-5s  %5.2f s  %s"
                  % (cycle, scene_id, action.state, len(poses),
                     "-" if not poses else "%.2f" % poses[0].submission_score,
                     cycle_s, action.reason[:44]))

            if not args.no_video:
                canvas = hud.render(hf)
                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        os.path.join(args.out, "cell_demo.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("cannot open the video writer -- "
                                           "this build of OpenCV has no mp4v")
                writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
        log.close()
        capture.close()

    if not cycle_times:
        print("no cycles ran")
        return 1
    order = sorted(cycle_times)
    summary = {
        "hardware": hardware, "source": capture.source,
        "cycles": len(cycle_times), "picks": picks, "rescans": rescans,
        "model_load_s": round(load_s, 2),
        "cycle_s": {"mean": round(sum(order) / len(order), 3),
                    "median": round(order[len(order) // 2], 3),
                    "max": round(order[-1], 3)},
        "config": {"seg_model": os.path.basename(args.seg_model),
                   "extra_seg_model": args.extra_seg_model, "imgsz": args.imgsz,
                   "conf": args.conf, "pick": args.pick,
                   "accept_score": args.accept_score},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n%d cycles: %d picked, %d rescanned | %.2f s median, %.2f s max"
          % (len(cycle_times), picks, rescans,
             summary["cycle_s"]["median"], summary["cycle_s"]["max"]))
    print("wrote %s" % args.out)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_argument_group("where the frames come from")
    src.add_argument("--root", default=".", help="Release folder")
    src.add_argument("--split", default="test")
    src.add_argument("--scenes", nargs="*", help="Subset, in this order")
    src.add_argument("--session", help="A recorded session directory instead")
    mdl = p.add_argument_group("what estimates them")
    mdl.add_argument("--cad", default=None, help="CAD ply (default <root>/model)")
    mdl.add_argument("--seg-model", default="weights/part-seg-nano.pt")
    mdl.add_argument("--extra-seg-model", default=None)
    mdl.add_argument("--imgsz", type=int, default=640)
    mdl.add_argument("--conf", type=float, default=0.25)
    mdl.add_argument("--grasps", default=None, help="cell-grasps json")
    cel = p.add_argument_group("how the cell decides")
    cel.add_argument("--pick", action="store_true", default=True,
                     help="Stop each frame at the first confident pose")
    cel.add_argument("--full-sweep", dest="pick", action="store_false",
                     help="Rank every instance instead (slow on a board)")
    cel.add_argument("--accept-score", type=float, default=0.7)
    out = p.add_argument_group("what comes out")
    out.add_argument("--out", default="out_demo")
    out.add_argument("--fps", type=float, default=VIDEO_FPS)
    out.add_argument("--cycles", type=int, default=0, help="0 = every frame")
    out.add_argument("--no-video", action="store_true")
    out.add_argument("--bench", default=None,
                     help="Bench record naming the machine, for the panel")
    out.add_argument("--hardware", default=None, help="Hardware line verbatim")
    args = p.parse_args()
    if args.cad is None:
        args.cad = os.path.join(args.root, "model", "3d_model.ply")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
