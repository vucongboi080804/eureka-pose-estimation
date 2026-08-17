"""Run the full detection + pose pipeline over a split.

Writes the submission JSON, and optionally a labels folder in the dataset's
own format -- poses.json plus rendered silhouette masks per scene -- so that
the released visualize.py can draw the predictions unchanged:

    .venv/bin/python scripts/run_pipeline.py --root . --split test \
        --out submission.json --labels-out pred_test
    .venv/bin/python visualize.py --root . --split test \
        --labels pred_test --save overlays_test/
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect import detect_scene, part_pixel_mask
from src.detect_seg import PICK_SCORE, detect_scene_hybrid
from src.model_cloud import load_model_cloud
from src.register import PoseEstimator
from src.scene_io import list_scenes, load_scene

_MODEL = None
_MESH = None
_SEG = None
_SEG_EXTRA = None


def _init_worker(ply_path, seg_weights=None, extra_weights=None):
    global _MODEL, _MESH, _SEG, _SEG_EXTRA
    _MODEL = load_model_cloud(ply_path)
    import trimesh
    m = trimesh.load(ply_path, force="mesh")
    _MESH = (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces))
    if seg_weights:
        from ultralytics import YOLO
        _SEG = YOLO(seg_weights)
        if extra_weights:
            _SEG_EXTRA = YOLO(extra_weights)


def _run_scene(args):
    root, split, scene_id, labels_out, passes, pick, hole_cue = args
    t0 = time.time()
    try:
        scene = load_scene(root, split, scene_id)
        estimator = PoseEstimator(_MODEL, scene.depth, scene.K,
                                  part_mask=part_pixel_mask(scene.rgb),
                                  hole_cue=hole_cue)
        if _SEG is not None:
            found = detect_scene_hybrid(scene, estimator, _SEG,
                                        extra_model=_SEG_EXTRA, pick=pick)
        else:
            found = detect_scene(scene, estimator, passes=passes)
        preds = [{"R": e.R.tolist(), "t": e.t.tolist(),
                  "score": round(e.submission_score, 4)} for e in found]
        if labels_out:
            _write_labels(labels_out, scene_id, found, scene)
    except Exception:
        # One unreadable or degenerate scene must not take the whole
        # submission down: report it and submit an empty list for it.
        print("%s  FAILED, submitting no poses" % scene_id, file=sys.stderr)
        traceback.print_exc()
        return scene_id, [], time.time() - t0
    return scene_id, preds, time.time() - t0


def _write_labels(labels_out, scene_id, found, scene):
    """Predictions in the dataset's own labels format, for visualize.py."""
    import score as scorer   # the released scoring script, for silhouette()

    verts, faces = _MESH
    scene_dir = os.path.join(labels_out, scene_id)
    os.makedirs(os.path.join(scene_dir, "masks"), exist_ok=True)
    rows = []
    for i, est in enumerate(found):
        sil = scorer.silhouette(verts, faces, est.R, est.t, scene.K,
                                scene.depth.shape)
        mask_rel = os.path.join("masks", "%03d.png" % i)
        cv2.imwrite(os.path.join(scene_dir, mask_rel), sil * 255)
        rows.append({"mask": mask_rel, "R": est.R.tolist(),
                     "t": est.t.tolist(),
                     "score": round(est.submission_score, 4)})
    with open(os.path.join(scene_dir, "poses.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=".")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="submission.json")
    p.add_argument("--labels-out", default=None,
                   help="Also write predictions as a dataset-format labels "
                        "folder for visualize.py")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--passes", type=int, default=2,
                   help="Independent detection sweeps to union; RANSAC "
                        "luck differs per sweep, the union keeps what any "
                        "sweep finds")
    p.add_argument("--seg-model", default=None,
                   help="Ultralytics segmentation weights; when given, "
                        "learned masks replace the geometric pile "
                        "splitting (registration stack unchanged)")
    p.add_argument("--extra-seg-model", default=None,
                   help="Second segmenter joining the proposal pool "
                        "(e.g. weights/part-seg-synthetic.pt)")
    p.add_argument("--no-hole-cue", dest="hole_cue", action="store_false",
                   help="Drop the RGB hole-consistency objection. On by "
                        "default: a predicted through-hole filled with "
                        "solid part colour at or in front of its own rim "
                        "is material the pose claims is empty, which is "
                        "how a half-turn hides from the depth verifier")
    p.add_argument("--pick", action="store_true",
                   help="Stop each scene at the first pose with score >= "
                        "%.1f -- deployment latency mode (learned-mask "
                        "path); the submission uses the full sweep"
                        % PICK_SCORE)
    args = p.parse_args()

    ply = os.path.join(args.root, "model", "3d_model.ply")
    scene_ids = list_scenes(args.root, args.split)
    if args.scenes:
        scene_ids = [s for s in scene_ids if s in set(args.scenes)]
    jobs = [(args.root, args.split, s, args.labels_out, args.passes,
             args.pick, args.hole_cue) for s in scene_ids]

    submission = {}
    if args.workers <= 1:
        _init_worker(ply, args.seg_model, args.extra_seg_model)
        results = map(_run_scene, jobs)
    else:
        # Share the cores between workers. Must be set in the parent before
        # spawning: libgomp reads OMP_NUM_THREADS once when Open3D loads,
        # so setting it inside the child initializer would come too late.
        os.environ.setdefault(
            "OMP_NUM_THREADS", str(max(1, os.cpu_count() // args.workers)))
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   mp_context=mp.get_context("spawn"),
                                   initializer=_init_worker,
                                   initargs=(ply, args.seg_model, args.extra_seg_model))
        results = pool.map(_run_scene, jobs)
    for scene_id, preds, dt in results:
        submission[scene_id] = preds
        print("%s  %2d found in %5.1fs  scores %s"
              % (scene_id, len(preds), dt,
                 [round(p["score"], 2) for p in preds]), flush=True)

    # Every scene of the split appears in the submission, empty when the
    # pipeline found nothing -- the scorer treats absent scenes as empty
    # anyway, but an explicit list keeps the file self-describing.
    for scene_id in scene_ids:
        submission.setdefault(scene_id, [])

    with open(args.out, "w") as fh:
        json.dump(submission, fh)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
