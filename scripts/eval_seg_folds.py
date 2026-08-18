"""Honest train-split evaluation of the learned-mask pipeline.

Each fold's model predicts only the scenes it never saw; stitching the
folds together covers the whole train split with held-out predictions.

    .venv/bin/python scripts/eval_seg_folds.py --root . --runs seg_runs \
        --out seg_train.json [--union pipeline_train.json]
    .venv/bin/python score.py --release . --split train --submission seg_train.json

``--ablate`` switches one stage off at a time (flip rivals, rotation-grid
fallback, polish, part-colour gate, own-mask check, RGB hole cue) so its
contribution can be measured against the full configuration.
"""

import argparse
import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.masks import part_pixel_mask
from src.nms import is_duplicate
from src.detect_seg import SEG_IMGSZ, detect_from_masks, masks_from_model
from src.model_cloud import load_model_cloud
from src.register import PoseEstimator
from src.scene_io import load_scene


def merge_nms(rows: list) -> list:
    """NMS over submission rows, by the detector's own duplicate rule."""
    kept = []
    for row in sorted(rows, key=lambda r: -r["score"]):
        if not any(is_duplicate(row["R"], row["t"], k["R"], k["t"]) for k in kept):
            kept.append(row)
    return kept

#: Stages ``--ablate`` can switch off; ``none`` is the full configuration.
ABLATIONS = ("none", "no_flips", "no_grid", "no_polish", "no_gate",
             "no_own_mask", "no_hole_cue")

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
    p.add_argument("--weights", default=None,
                   help="Evaluate ONE model on every train scene instead "
                        "of stitching folds. Only honest for a model that "
                        "never saw the train split (e.g. synthetic-only).")
    p.add_argument("--extra-weights", default=None,
                   help="Second segmenter whose masks join the proposal "
                        "pool (e.g. the synthetic-only model).")
    p.add_argument("--extra-runs", default=None,
                   help="Second per-fold runs dir whose fold models also "
                        "join the proposal pool.")
    p.add_argument("--attempts", type=int, default=3,
                   help="RANSAC restart budget per mask")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Segmentation confidence floor for proposals")
    p.add_argument("--imgsz", type=int, default=SEG_IMGSZ,
                   help="Segmenter input side; lower it to price a "
                        "memory-constrained board (analysis/nano_profile.md)")
    p.add_argument("--ablate", action="append", default=[],
                   metavar="|".join(ABLATIONS),
                   help="Switch stages off (repeatable, or comma-separated)")
    args = p.parse_args()
    ablate = {a for arg in args.ablate for a in arg.split(",") if a}
    unknown = ablate - set(ABLATIONS)
    if unknown:
        p.error("unknown --ablate value(s): %s" % ", ".join(sorted(unknown)))
    ablate.discard("none")

    from ultralytics import YOLO

    geometric = {}
    if args.union:
        with open(args.union) as fh:
            geometric = json.load(fh)

    model_cloud = load_model_cloud(
        os.path.join(args.root, "model", "3d_model.ply"))
    submission = {}
    if args.weights:
        all_scenes = sorted(s for fold in FOLD_VAL_SCENES.values()
                            for s in fold)
        plan = {"single": all_scenes}
    else:
        plan = FOLD_VAL_SCENES
    extra = YOLO(args.extra_weights) if args.extra_weights else None
    for fold, scene_ids in sorted(plan.items()):
        weights = args.weights or os.path.join(args.runs, fold,
                                               "weights", "best.pt")
        model = YOLO(weights)
        model2 = None
        if args.extra_runs:
            model2 = YOLO(os.path.join(args.extra_runs, fold,
                                       "weights", "best.pt"))
        for sid in scene_ids:
            scene = load_scene(args.root, "train", sid)
            estimator = PoseEstimator(model_cloud, scene.depth, scene.K,
                                      part_mask=part_pixel_mask(scene.rgb),
                                      flips="no_flips" not in ablate,
                                      grid="no_grid" not in ablate,
                                      polish="no_polish" not in ablate,
                                      hole_cue="no_hole_cue" not in ablate)
            masks = masks_from_model(model, scene.rgb, conf=args.conf,
                                     imgsz=args.imgsz)
            if extra is not None:
                masks += masks_from_model(extra, scene.rgb, conf=args.conf,
                                          imgsz=args.imgsz)
            if model2 is not None:
                masks += masks_from_model(model2, scene.rgb, conf=args.conf,
                                          imgsz=args.imgsz)
            found = detect_from_masks(
                scene, estimator, masks, attempts=args.attempts,
                colour_gate="no_gate" not in ablate,
                own_mask_check="no_own_mask" not in ablate)
            rows = [{"R": e.R.tolist(), "t": e.t.tolist(),
                     "score": round(e.submission_score, 4)} for e in found]
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
