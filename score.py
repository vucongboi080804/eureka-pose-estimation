"""Score predicted poses against ground truth.

This is the exact script used to grade submissions. Run it on the training
split to measure yourself before you submit::

    python score.py --release . --split train --submission my_train_poses.json

Its labels come from beside the images by default; ``--gt`` points it at a
folder of per-scene labels held elsewhere.

**Metric: MSSD** (maximum symmetry-aware surface distance),

    e = min over symmetries S of  max over model vertices x of
        || R_est x + t_est  -  (R_gt (S x) + t_gt) ||

The maximum deviation of any point on the model, taking each vertex to its
counterpart, minimised over the object's symmetry transforms. This object is
asymmetric, so that set contains only the identity and the expression reduces to

    e = max over x of  || (R_est - R_gt) x + (t_est - t_gt) ||

``||Ax + b||`` is convex in ``x``, so its maximum over a point set is attained
at a vertex of that set's convex hull. Only hull vertices are evaluated and the
result is exact rather than approximate.

**Matching.** Predictions are taken in order of descending ``score`` and each
claims the closest unclaimed ground-truth instance whose error is under the
threshold. Every instance can be claimed once. An unmatched prediction counts as
a false positive unless it lands on an ignore region, or it matched an instance
whose visible fraction is below ``--visib-min``; in both of those cases it is
discarded instead.

**Visible fraction** is computed here, as visible mask area over the area the
model covers when projected without occluders. Instances below ``--visib-min``
are optional: finding them counts for nothing and missing them costs nothing.

**Reported.** Recall and precision at each MSSD threshold, their mean (AR), and
the fraction of scenes whose highest-``score`` prediction is within
``TOP1_THRESHOLD_MM`` of some instance. A scene with instances but no prediction
counts against that fraction.

Needs ``numpy``, ``opencv-python`` and ``trimesh``.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
import typing as T

import numpy as np

logger = logging.getLogger("pose_annot.score_assignment")

#: MSSD thresholds in millimetres.
DEFAULT_THRESHOLDS_MM = (2.0, 4.0, 6.0, 8.0, 10.0)

#: Threshold in millimetres for the top-1 decision.
TOP1_THRESHOLD_MM = 5.0

#: Instances at least this visible are required; the rest are optional.
DEFAULT_VISIB_MIN = 0.8

#: A prediction whose projection is at least this covered by an ignore region is
#: discarded rather than counted as a false positive.
IGNORE_OVERLAP = 0.5

M_TO_MM = 1000.0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score predicted poses against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--release", required=True,
                   help="Dataset root (holds model/ and the split folders)")
    p.add_argument("--split", default="test", help="Split the submission covers")
    p.add_argument("--gt", default=None,
                   help="Folder of per-scene labels; defaults to reading them "
                        "from beside the images in the split")
    p.add_argument("--submission", help="Predictions: {scene: [{R, t, score}]}")
    p.add_argument("--thresholds", type=float, nargs="*",
                   default=list(DEFAULT_THRESHOLDS_MM),
                   help="MSSD thresholds in millimetres")
    p.add_argument("--visib-min", type=float, default=DEFAULT_VISIB_MIN,
                   help="Below this visible fraction an instance is optional")
    p.add_argument("--tsv", action="store_true",
                   help="Also print a tab-separated summary line, for pasting "
                        "into a spreadsheet")
    p.add_argument("--tsv-header", action="store_true",
                   help="Print the column names above the --tsv line")
    p.add_argument("--label", default=None,
                   help="Name used in the --tsv line; defaults to the "
                        "submission's filename")
    p.add_argument("--selftest", action="store_true",
                   help="Score predictions synthesised from the ground truth, "
                        "as a check that the scorer runs and responds")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_hull_vertices(model_path: str) -> np.ndarray:
    """Convex-hull vertices of the model, in the model's own units (metres).

    The hull suffices because MSSD maximises a convex function of the vertex
    position, and such a maximum over a point set is attained at a hull vertex.
    """
    import trimesh

    mesh = trimesh.load(model_path, force="mesh")
    hull = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    logger.info("Model %s: %d vertices -> %d hull vertices",
                os.path.basename(model_path), len(mesh.vertices), len(hull))
    return hull


def mssd_mm(
    R_est: np.ndarray, t_est: np.ndarray,
    R_gt: np.ndarray, t_gt: np.ndarray,
    hull: np.ndarray, symmetries: T.Optional[T.Sequence[np.ndarray]] = None,
) -> float:
    """MSSD between two poses, in millimetres.

    Args:
        R_est, t_est: Predicted pose.
        R_gt, t_gt: Ground-truth pose.
        hull: ``(N, 3)`` hull vertices in model units (metres).
        symmetries: Optional 4x4 model symmetry transforms. ``None`` means the
            object is asymmetric, which is the case for this dataset.

    Returns:
        Maximum vertex deviation, millimetres.
    """
    best = np.inf
    for S in (symmetries or [np.eye(4)]):
        x = hull @ np.asarray(S)[:3, :3].T + np.asarray(S)[:3, 3]
        delta = (hull @ np.asarray(R_est).T + np.asarray(t_est)) - (
            x @ np.asarray(R_gt).T + np.asarray(t_gt))
        best = min(best, float(np.linalg.norm(delta, axis=1).max()))
    return best * M_TO_MM


# --------------------------------------------------------------------------- #
# Projection, for the visible fraction and for ignore handling
# --------------------------------------------------------------------------- #


def silhouette(verts: np.ndarray, faces: np.ndarray, R, t, K, shape) -> np.ndarray:
    """Rasterize the posed model's front faces, ignoring occluders.

    Triangles are filled rather than points splatted, so the model's through
    holes appear in the result.
    """
    import cv2

    pc = verts @ np.asarray(R).T + np.asarray(t)
    normals = np.cross(pc[faces[:, 1]] - pc[faces[:, 0]],
                       pc[faces[:, 2]] - pc[faces[:, 0]])
    front = np.einsum("ij,ij->i", normals, -pc[faces].mean(axis=1)) > 0
    if not front.any():
        front = np.ones(len(faces), dtype=bool)
    z = np.clip(pc[:, 2], 1e-9, None)
    uv = np.column_stack([K[0, 0] * pc[:, 0] / z + K[0, 2],
                          K[1, 1] * pc[:, 1] / z + K[1, 2]])
    out = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(out, np.rint(uv[faces[front]]).astype(np.int32), 1)
    return out


class SceneData:
    """Ground truth for one scene, and what is needed to judge predictions."""

    def __init__(self, scene, image_dir, gt_dir, mesh, K):
        import cv2

        self.scene = scene
        self.K = K
        verts, faces = mesh
        with open(os.path.join(gt_dir, "poses.json")) as fh:
            poses = json.load(fh)

        self.R = [np.asarray(p["R"], dtype=np.float64) for p in poses]
        self.t = [np.asarray(p["t"], dtype=np.float64) for p in poses]
        self.shape = cv2.imread(
            os.path.join(image_dir, "rgb.png"), cv2.IMREAD_GRAYSCALE).shape

        # Visible mask area over the area the model covers when projected with
        # no occluders. Computed rather than shipped, so this script needs only
        # the images, the labels and the model.
        self.visib = []
        for pose, R, t in zip(poses, self.R, self.t):
            mask = cv2.imread(os.path.join(gt_dir, pose["mask"]), cv2.IMREAD_GRAYSCALE)
            amodal = silhouette(verts, faces, R, t, K, self.shape)
            denom = int(amodal.sum())
            self.visib.append(float((mask > 0).sum() / denom) if denom else 0.0)

        self.ignore = [
            cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 0
            for f in sorted(glob.glob(os.path.join(gt_dir, "ignore", "*.png")))
        ]

    def is_ignored(self, R, t, mesh) -> bool:
        """Whether a predicted pose lands on an ignore region."""
        if not self.ignore:
            return False
        sil = silhouette(mesh[0], mesh[1], R, t, self.K, self.shape) > 0
        area = int(sil.sum())
        if not area:
            return False
        return any(float((sil & ign).sum() / area) >= IGNORE_OVERLAP
                   for ign in self.ignore)


# --------------------------------------------------------------------------- #
# Matching and aggregation
# --------------------------------------------------------------------------- #


def score_scene(
    data: SceneData,
    preds: T.Sequence[T.Dict[str, T.Any]],
    hull: np.ndarray,
    mesh,
    thresholds: T.Sequence[float],
    visib_min: float,
) -> T.Dict[str, T.Any]:
    """Match one scene's predictions to its ground truth at every threshold.

    Errors are computed once and thresholded afterwards, since the pairing does
    not depend on the threshold.

    Returns:
        Per-threshold counts, plus the scene's top-1 decision.
    """
    order = sorted(range(len(preds)),
                   key=lambda i: -float(preds[i].get("score", 0.0)))
    n_gt = len(data.R)
    required = [v >= visib_min for v in data.visib]

    # errors[prediction][instance], millimetres
    errors = np.full((len(preds), n_gt), np.inf)
    for i, pred in enumerate(preds):
        R = np.asarray(pred["R"], dtype=np.float64)
        t = np.asarray(pred["t"], dtype=np.float64)
        for j in range(n_gt):
            errors[i, j] = mssd_mm(R, t, data.R[j], data.t[j], hull)

    ignored = [data.is_ignored(np.asarray(p["R"], dtype=np.float64),
                               np.asarray(p["t"], dtype=np.float64), mesh)
               for p in preds]

    out: T.Dict[str, T.Any] = {"per_threshold": {}}
    for tau in thresholds:
        claimed = [False] * n_gt
        tp = fp = 0
        for i in order:
            candidates = [j for j in range(n_gt)
                          if not claimed[j] and errors[i, j] < tau]
            if candidates:
                j = min(candidates, key=lambda j: errors[i, j])
                claimed[j] = True
                if required[j]:
                    tp += 1
                # A hit on an optional instance is neither counted nor charged.
            elif not ignored[i]:
                fp += 1
        out["per_threshold"][tau] = {
            "tp": tp, "fp": fp,
            "n_required": sum(required),
            "missed": sum(1 for j in range(n_gt) if required[j] and not claimed[j]),
        }

    # Submitting nothing for a scene that has instances counts as a top-1 miss;
    # only a scene with no instances at all is left out of the fraction.
    out["top1"] = None
    if n_gt:
        out["top1"] = bool(
            len(preds) and errors[order[0]].min() < TOP1_THRESHOLD_MM
        )
    out["n_preds"] = len(preds)
    out["visib"] = data.visib
    out["errors"] = errors
    return out


def aggregate(results: T.Dict[str, T.Dict], thresholds, visib_min) -> T.Dict[str, T.Any]:
    """Roll the per-scene results up into the reported figures.

    Shared by the printed table and the tab-separated row, so the two cannot
    disagree about what was measured.
    """
    rows = []
    for tau in thresholds:
        tp = sum(r["per_threshold"][tau]["tp"] for r in results.values())
        fp = sum(r["per_threshold"][tau]["fp"] for r in results.values())
        req = sum(r["per_threshold"][tau]["n_required"] for r in results.values())
        missed = sum(r["per_threshold"][tau]["missed"] for r in results.values())
        rows.append({
            "threshold": tau, "tp": tp, "fp": fp, "required": req, "missed": missed,
            "recall": tp / req if req else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        })
    top1 = [r["top1"] for r in results.values() if r["top1"] is not None]
    visib = np.array([v for r in results.values() for v in r["visib"]])
    return {
        "per_threshold": rows,
        "ar": float(np.mean([r["recall"] for r in rows])) if rows else 0.0,
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "top1_hits": int(sum(top1)),
        "top1_scenes": len(top1),
        "scenes": len(results),
        "predictions": sum(r["n_preds"] for r in results.values()),
        "required": int((visib >= visib_min).sum()) if visib.size else 0,
        "optional": int((visib < visib_min).sum()) if visib.size else 0,
    }


def report(results: T.Dict[str, T.Dict], thresholds, visib_min, title="") -> float:
    """Print the score table. Returns AR."""
    agg = aggregate(results, thresholds, visib_min)
    print("\n%s" % (title or "RESULTS"))
    print("  MSSD    recall   precision        TP     FP   missed")
    for row in agg["per_threshold"]:
        print("  %4.0f mm  %6.3f      %6.3f    %6d %6d   %6d"
              % (row["threshold"], row["recall"], row["precision"],
                 row["tp"], row["fp"], row["missed"]))
    print("  %-24s %.3f  (mean recall over the thresholds above)"
          % ("AR", agg["ar"]))
    if agg["top1_scenes"]:
        print("  %-24s %.3f  (%d of %d scenes, top-score pose within %.0f mm)"
              % ("top-1 per scene", agg["top1"], agg["top1_hits"],
                 agg["top1_scenes"], TOP1_THRESHOLD_MM))
    print("  %-24s %d required (visible fraction >= %.2f), %d optional"
          % ("instance pool", agg["required"], visib_min, agg["optional"]))
    return agg["ar"]


def tsv_lines(
    agg: T.Dict[str, T.Any], label: str, split: str,
    unscored: int, missing: int, header: bool = False,
) -> T.List[str]:
    """One tab-separated line of results, optionally preceded by its header.

    Tab separated rather than comma separated because tabs paste into one cell
    per column in a spreadsheet, where commas land in a single cell.

    Args:
        agg: Output of :func:`aggregate`.
        label: Name for this row.
        split: Which split was scored.
        unscored: Predictions submitted without a ``score``.
        missing: Scenes absent from the submission.
        header: Also return the column-name line.

    Returns:
        One or two lines, header first.
    """
    cols: T.List[T.Tuple[str, T.Any]] = [
        ("label", label),
        ("scored_at", time.strftime("%Y-%m-%d")),
        ("split", split),
        ("scenes", agg["scenes"]),
        ("predictions", agg["predictions"]),
        ("predictions_unscored", unscored),
        ("scenes_missing", missing),
        ("AR", "%.4f" % agg["ar"]),
        ("top1", "%.4f" % agg["top1"]),
    ]
    for row in agg["per_threshold"]:
        cols.append(("recall_%gmm" % row["threshold"], "%.4f" % row["recall"]))
        cols.append(("precision_%gmm" % row["threshold"],
                     "%.4f" % row["precision"]))
    lines = []
    if header:
        lines.append("\t".join(name for name, _ in cols))
    lines.append("\t".join(str(value) for _, value in cols))
    return lines


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def synthesise(gt_scenes: T.Dict[str, SceneData], kind: str, seed: int = 7):
    """Build predictions out of the ground truth, to show how the scorer moves.

    ``exact`` returns the labels unchanged, ``jittered`` perturbs them by about
    2 degrees and 1.5 mm, and ``rotated`` turns each pose 180 degrees about its
    own Z axis.
    """
    rng = np.random.RandomState(seed)
    half_turn_z = np.diag([-1.0, -1.0, 1.0])
    sub: T.Dict[str, T.List[T.Dict[str, T.Any]]] = {}
    for scene, data in gt_scenes.items():
        rows = []
        for R, t in zip(data.R, data.t):
            if kind == "exact":
                Rp, tp = R, t
            elif kind == "jittered":
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                ang = np.deg2rad(rng.normal(0, 2.0))
                Kx = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
                Rp = R @ (np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx)
                tp = t + rng.normal(0, 0.0015, size=3)
            elif kind == "rotated":
                Rp, tp = R @ half_turn_z, t
            else:
                raise ValueError(kind)
            rows.append({"R": Rp.tolist(), "t": np.asarray(tp).tolist(),
                         "score": float(rng.uniform(0.5, 1.0))})
        sub[scene] = rows
    return sub


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_scenes(release: str, split: str, gt_root: str, mesh) -> T.Dict[str, SceneData]:
    """Load every scene that has both images and labels."""
    split_dir = os.path.join(release, split)
    scenes: T.Dict[str, SceneData] = {}
    for gt_dir in sorted(glob.glob(os.path.join(gt_root, "*"))):
        scene = os.path.basename(gt_dir)
        image_dir = os.path.join(split_dir, scene)
        if not os.path.isdir(image_dir) or not os.path.isdir(gt_dir):
            continue
        if not os.path.exists(os.path.join(gt_dir, "poses.json")):
            continue  # a split that ships without labels
        with open(os.path.join(image_dir, "camera.json")) as fh:
            K = np.asarray(json.load(fh)["K"], dtype=np.float64)
        scenes[scene] = SceneData(scene, image_dir, gt_dir, mesh, K)
    return scenes


def main(argv: T.Optional[T.List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    import trimesh

    # Fixed preference order so the choice never depends on directory listing
    # order. Both files describe the same surface, so scores are unaffected.
    models = [p for ext in (".ply", ".glb", ".obj", ".stl")
              for p in sorted(glob.glob(os.path.join(args.release, "model", "*" + ext)))]
    if not models:
        logger.error("No model file under %s/model", args.release)
        return 2
    hull = load_hull_vertices(models[0])
    m = trimesh.load(models[0], force="mesh")
    mesh = (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces))

    gt_root = args.gt or os.path.join(args.release, args.split)
    scenes = load_scenes(args.release, args.split, gt_root, mesh)
    if not scenes:
        logger.error("No scene under %s has both images and a poses.json", gt_root)
        return 2
    logger.info("Loaded %d scenes, %d ground-truth instances",
                len(scenes), sum(len(s.R) for s in scenes.values()))

    if args.selftest:
        for kind in ("exact", "jittered", "rotated"):
            sub = synthesise(scenes, kind)
            results = {
                scene: score_scene(data, sub.get(scene, []), hull, mesh,
                                   args.thresholds, args.visib_min)
                for scene, data in scenes.items()
            }
            report(results, args.thresholds, args.visib_min,
                   title="SELFTEST -- predictions synthesised from labels (%s)" % kind)
        print()
        return 0

    if not args.submission:
        logger.error("Pass --submission or --selftest")
        return 2
    with open(args.submission) as fh:
        sub = json.load(fh)
    # Missing scores are legal and fall back to list order, but the fallback is
    # silent and costs the whole top-1 figure if the list happens to be ordered
    # badly. Warning rather than failing: a submission is still gradeable.
    n_pred = sum(len(v) for v in sub.values() if isinstance(v, list))
    n_scored = sum(1 for v in sub.values() if isinstance(v, list)
                   for p in v if "score" in p)
    if n_pred and n_scored < n_pred:
        logger.warning(
            "%d of %d predictions have no 'score'; those rank below every "
            "scored one, and equal ranks fall back to position in the list. "
            "Ranking decides the top-1 result.",
            n_pred - n_scored, n_pred,
        )

    unknown = sorted(set(sub) - set(scenes))
    if unknown:
        logger.warning("Submission has %d scene(s) with no ground truth: %s",
                       len(unknown), ", ".join(unknown[:5]))
    absent = sorted(set(scenes) - set(sub))
    if absent:
        logger.warning("Submission is missing %d scene(s); scored as empty",
                       len(absent))

    results = {
        scene: score_scene(data, sub.get(scene, []), hull, mesh,
                           args.thresholds, args.visib_min)
        for scene, data in scenes.items()
    }
    report(results, args.thresholds, args.visib_min,
           title="SUBMISSION %s" % os.path.basename(args.submission))

    print()
    if args.tsv or args.tsv_header:
        # Printed last and nothing after it, so `... --tsv | tail -1` yields
        # exactly the row.
        label = args.label or os.path.splitext(
            os.path.basename(args.submission))[0]
        for line in tsv_lines(
            aggregate(results, args.thresholds, args.visib_min),
            label=label, split=args.split, unscored=n_pred - n_scored,
            missing=len(absent), header=args.tsv_header,
        ):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
