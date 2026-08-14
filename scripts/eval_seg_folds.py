"""Honest train-split evaluation of the learned-mask pipeline.

Each fold's model predicts only the scenes it never saw; stitching the
folds together covers the whole train split with held-out predictions.

    .venv/bin/python scripts/eval_seg_folds.py --root . --runs seg_runs \
        --out seg_train.json [--union pipeline_train.json]
    .venv/bin/python score.py --release . --split train --submission seg_train.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect import NMS_ANGLE_DEG, NMS_DIST, part_pixel_mask
from src.detect_seg import detect_from_masks, masks_from_model
from src.model_cloud import load_model_cloud
from src.register import PoseEstimator
from src.scene_io import load_scene


def merge_nms(rows: list) -> list:
    """NMS over submission rows: same place AND same orientation = dup."""
    kept = []
    for row in sorted(rows, key=lambda r: -r["score"]):
        R, t = np.array(row["R"]), np.array(row["t"])
        dup = False
        for k in kept:
            if np.linalg.norm(t - np.array(k["t"])) > NMS_DIST:
                continue
            cos = (np.trace(R.T @ np.array(k["R"])) - 1.0) / 2.0
            if np.degrees(np.arccos(np.clip(cos, -1, 1))) < NMS_ANGLE_DEG:
                dup = True
                break
        if not dup:
            kept.append(row)
    return kept

FOLD_VAL_SCENES = {
    "fold0": ["000007", "000014", "000021", "000033", "000047"],
    "fold1": ["000008", "000019", "000022", "000040", "000054"],
    "fold2": ["000009", "000020", "000026", "000041", "000058"],
    "fold3": ["000010", "000011", "000023", "000030", "000059"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=".")
    p.add_argument("--runs", default="seg_runs")
    p.add_argument("--out", default="seg_train.json")
    p.add_argument("--union", default=None,
                   help="Optional geometric-pipeline submission to merge "
                        "with (NMS dedupes)")
    args = p.parse_args()

    from ultralytics import YOLO

    geometric = {}
    if args.union:
        with open(args.union) as fh:
            geometric = json.load(fh)

    model_cloud = load_model_cloud(
        os.path.join(args.root, "model", "3d_model.ply"))
    submission = {}
    for fold, scene_ids in sorted(FOLD_VAL_SCENES.items()):
        weights = os.path.join(args.runs, fold, "weights", "best.pt")
        model = YOLO(weights)
        for sid in scene_ids:
            scene = load_scene(args.root, "train", sid)
            estimator = PoseEstimator(model_cloud, scene.depth, scene.K,
                                      part_mask=part_pixel_mask(scene.rgb))
            masks = masks_from_model(model, scene.rgb)
            found = detect_from_masks(scene, estimator, masks)
            rows = [{"R": e.R.tolist(), "t": e.t.tolist(),
                     "score": round(e.confidence, 4)} for e in found]
            rows = merge_nms(rows + geometric.get(sid, []))
            submission[sid] = rows
            print("%s %s: %d masks -> %d predictions"
                  % (fold, sid, len(masks), len(submission[sid])),
                  flush=True)

    with open(args.out, "w") as fh:
        json.dump(submission, fh)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
