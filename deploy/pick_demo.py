"""One robot cycle, end to end, without a camera.

Reads one scene's arrays the way a camera SDK would hand them over (colour
frame, raw depth frame, intrinsics, depth scale) and runs the live adapter's
pick path: the pose the cell would grab, with the confidence it would gate
on. Use it to check a new machine -- or a new camera's calibration -- before
wiring the SDK in.

    .venv/bin/python deploy/pick_demo.py --root . --split test --scene 000001
    .venv/bin/python deploy/pick_demo.py --root . --split test --scene 000001 --all

``--all`` runs the full sweep (every instance, ranked) instead of stopping at
the first confident pick.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.live_adapter import estimate_scene, load_models, scene_from_arrays


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=".", help="Release folder (model/ + splits)")
    p.add_argument("--split", default="test")
    p.add_argument("--scene", required=True)
    p.add_argument("--all", action="store_true",
                   help="Full sweep instead of the first confident pick")
    p.add_argument("--weights", default="weights/part-seg.pt")
    p.add_argument("--extra-weights", default="weights/part-seg-synthetic.pt")
    args = p.parse_args()

    d = os.path.join(args.root, args.split, args.scene)
    # Exactly what a camera hands over: BGR frame, raw depth frame, and the
    # calibration that turns depth ticks into metres.
    rgb = cv2.imread(os.path.join(d, "rgb.png"))
    depth_raw = cv2.imread(os.path.join(d, "depth.png"), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(d, "camera.json")) as fh:
        cam = json.load(fh)

    t0 = time.time()
    model_cloud, seg, extra = load_models(
        os.path.join(args.root, "model", "3d_model.ply"),
        args.weights, args.extra_weights or None)
    load_s = time.time() - t0

    scene = scene_from_arrays(rgb, depth_raw, cam["K"], cam["depth_scale"],
                              scene_id=args.scene)
    t0 = time.time()
    poses = estimate_scene(scene, model_cloud, seg, extra, pick=not args.all)
    cycle_s = time.time() - t0

    print("models loaded in %.1f s (once per process)" % load_s)
    print("%s: %d pose(s) in %.2f s%s"
          % (args.scene, len(poses), cycle_s, "" if args.all else "  [pick mode]"))
    for i, pose in enumerate(poses):
        t = pose["t"]
        print("  %d  score %.3f  t = [%7.4f %7.4f %7.4f] m"
              % (i, pose["score"], t[0], t[1], t[2]))
    if poses:
        best = poses[0]
        print("\ngrasp pose (T_camera_object), score %.3f:" % best["score"])
        for row in best["R"]:
            print("  [%8.5f %8.5f %8.5f]" % tuple(row))
        print("  t = [%.5f %.5f %.5f] m" % tuple(best["t"]))
        print("\nA cell would grab this one, or rescan if the score is below "
              "its gate (0.7 keeps precision 1.00 at 5 mm on held-out scenes).")
    else:
        print("\nNothing verified: rescan or shake the bin.")


if __name__ == "__main__":
    main()
