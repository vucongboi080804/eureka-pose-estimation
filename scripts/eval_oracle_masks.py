"""Registration ceiling: estimate poses from the *ground-truth* masks.

Segmentation is the other half of the problem; this script removes it. Feeding
the labelled masks straight into registration measures how much score the
pose estimator alone can reach — the ceiling any detector-driven pipeline is
chasing. Run score.py on the output:

    .venv/bin/python scripts/eval_oracle_masks.py --root .
    .venv/bin/python score.py --release . --split train --submission oracle_masks.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect import part_pixel_mask
from src.scene_io import backproject, list_scenes, load_scene
from src.model_cloud import load_model_cloud
from src.register import PoseEstimator

_MODEL = None

#: Peel the mask edge before back-projection: boundary pixels carry depth
#: blended between the part and whatever lies behind it.
ERODE_PX = 2


def _init_worker(ply_path):
    global _MODEL
    _MODEL = load_model_cloud(ply_path)


def eroded(mask, px=ERODE_PX, keep_at_least=300):
    """Shrink the mask away from its unreliable boundary, unless that would
    leave too little of a small instance to register."""
    slim = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                     iterations=px)
    return slim.astype(bool) if slim.sum() >= keep_at_least else mask


def _run_scene(args):
    root, split, scene_id = args
    scene = load_scene(root, split, scene_id)
    estimator = PoseEstimator(_MODEL, scene.depth, scene.K,
                              part_mask=part_pixel_mask(scene.rgb))
    preds = []
    t0 = time.time()
    for mask_path in scene.mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        points = backproject(scene.depth, scene.K, eroded(mask))
        est = estimator.estimate(points)
        if est is None:
            continue
        preds.append({
            "R": est.R.tolist(),
            "t": est.t.tolist(),
            "score": est.confidence,
        })
    return scene_id, preds, len(scene.mask_paths), time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=".")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="oracle_masks.json")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    ply = os.path.join(args.root, "model", "3d_model.ply")
    scene_ids = list_scenes(args.root, args.split)
    jobs = [(args.root, args.split, s) for s in scene_ids]

    submission = {}
    if args.workers <= 1:
        _init_worker(ply)
        results = map(_run_scene, jobs)
    else:
        # spawn, not fork: Open3D's OpenMP pools deadlock in forked children.
        # OMP_NUM_THREADS must be set here, in the parent: libgomp reads it
        # once when Open3D loads, before a child initializer could run.
        os.environ.setdefault(
            "OMP_NUM_THREADS", str(max(1, os.cpu_count() // args.workers)))
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   mp_context=mp.get_context("spawn"),
                                   initializer=_init_worker, initargs=(ply,))
        results = pool.map(_run_scene, jobs)
    for scene_id, preds, n_masks, dt in results:
        submission[scene_id] = preds
        fits = sorted(round(p["score"], 2) for p in preds)
        print("%s  %d/%d registered in %4.1fs  fitness %s"
              % (scene_id, len(preds), n_masks, dt, fits), flush=True)

    with open(args.out, "w") as fh:
        json.dump(submission, fh)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
