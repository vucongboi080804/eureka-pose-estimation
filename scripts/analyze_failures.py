"""Per-instance failure analysis of a train-split submission.

Matching is score.py's own (imported and cross-checked per scene), so every
number agrees with the official scorer. Per required instance: match, MSSD,
plate tilt to the optical axis, through-holes visible, misses shared with the
other result files. Per miss: nearest unclaimed prediction (rotation to GT,
flip axis), whether the segmenters proposed the instance, and a diagnosis;
unmatched predictions are binned the same way. Tables are printed and written
between the ``<!-- tables -->`` markers of ``<out>/failure_analysis.md``
(prose outside them is kept); crops of the misses go to ``<out>/failures/``:

    .venv/bin/python scripts/analyze_failures.py --root . \
        --submission results/train_ensemble_run1.json --out analysis
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import score  # noqa: E402
from eval_seg_folds import FOLD_VAL_SCENES  # noqa: E402
from src.detect import part_pixel_mask  # noqa: E402
from src.detect_seg import MIN_MASK_PX, MIN_PART_COLOUR_FRACTION, masks_from_model  # noqa: E402
from src.model_cloud import load_model_cloud  # noqa: E402

#: Result files whose misses are compared with the analysed submission.
OTHER_RESULTS = ("train_ensemble_run2", "train_yolo11l_single",
                 "train_synthetic_only", "train_gt_masks")
THRESHOLDS = (2.0, 4.0, 6.0, 8.0, 10.0)
NEAR_MM, DUP_MM = 15.0, 5.0    # mislocalised radius; GT poses closer = one part
FLIP_DEG, CROP_KB = 150.0, 150  # nearest pose rotated more = flip; crop size cap
MARK = ("<!-- tables:start -->", "<!-- tables:end -->")
AXES = ("X (bar)", "Y (plate normal)", "Z (stem)")


def holes_visible(R, t, K, mask, model, tilt):
    """Which of the model's through-holes show in the GT mask (rim inside)."""
    if tilt > 75:
        return [False] * len(model.hole_centres)
    b1 = np.cross(model.plate_axis, [1.0, 0, 0])
    b1 /= np.linalg.norm(b1)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)[:, None]
    ring = np.cos(ang) * b1 + np.sin(ang) * np.cross(model.plate_axis, b1)
    rims = model.hole_centres[:, None] + 1.25 * model.hole_radii[:, None, None] * ring
    pc = rims.reshape(-1, 3) @ R.T + t
    u = np.rint(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    ok = (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    ok &= mask[np.clip(v, 0, mask.shape[0] - 1), np.clip(u, 0, mask.shape[1] - 1)]
    return (ok.reshape(len(model.hole_centres), -1).mean(1) >= 0.75).tolist()


def match_at(errors, order, tau):
    """score.py's greedy matching: {gt index: prediction index}."""
    claimed, assign = set(), {}
    for i in order:
        cands = [j for j in range(errors.shape[1]) if j not in claimed and errors[i, j] < tau]
        if cands:
            j = min(cands, key=lambda j: errors[i, j])
            claimed.add(j)
            assign[j] = i
    return assign


def pose_delta(R_gt, t_gt, pred):
    """Rotation (deg, dominant model axis) and translation (mm) GT -> pred."""
    rel = R_gt.T @ np.asarray(pred["R"])
    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(rel)
    axis = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    return ang, AXES[int(np.argmax(axis))], np.linalg.norm(np.asarray(pred["t"]) - t_gt) * 1e3


def scene_matches(sub, scenes, hull, mesh):
    """Errors, ignore flags and per-threshold matches per scene."""
    out = {}
    for sid, data in scenes.items():
        preds = sub.get(sid, [])
        r = score.score_scene(data, preds, hull, mesh, THRESHOLDS, score.DEFAULT_VISIB_MIN)
        order = sorted(range(len(preds)), key=lambda i: -float(preds[i].get("score", 0.0)))
        matches = {tau: match_at(r["errors"], order, tau) for tau in THRESHOLDS}
        req = [v >= score.DEFAULT_VISIB_MIN for v in data.visib]
        for tau in THRESHOLDS:   # same TP count as the official scorer
            assert sum(req[j] for j in matches[tau]) == r["per_threshold"][tau]["tp"]
        ignored = [data.is_ignored(np.asarray(p["R"]), np.asarray(p["t"]), mesh) for p in preds]
        out[sid] = {"errors": r["errors"], "matches": matches, "preds": preds, "ignored": ignored}
    return out


def best_proposals(models, rgb, gt_mask, conf):
    """Per segmenter, its best-IoU proposal for ``gt_mask``: (IoU, conf, gate)."""
    colour, out = part_pixel_mask(rgb), {}
    for name, model in models.items():
        ms = masks_from_model(model, rgb, conf=conf)
        ious = [(gt_mask & m).sum() / max((gt_mask | m).sum(), 1) for m, _ in ms]
        i = int(np.argmax(ious)) if ious else None
        out[name] = (ious[i], ms[i][1], bool(
            ms[i][0].sum() >= MIN_MASK_PX
            and colour[ms[i][0]].mean() >= MIN_PART_COLOUR_FRACTION)) if ious else (0, 0, False)
    return out


def draw_crop(rgb, data, mesh, j, pred, path):
    """RGB crop of instance ``j``: GT contour green, nearest prediction red."""
    layers = [(data.R[j], data.t[j], (0, 220, 0))] + (
        [(np.array(pred["R"]), np.array(pred["t"]), (0, 0, 255))] if pred is not None else [])
    img, ys, xs = rgb.copy(), [], []
    for R, t, colour in layers:
        sil = score.silhouette(mesh[0], mesh[1], R, t, data.K, data.shape)
        cs, _ = cv2.findContours(sil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, colour, 2)
        py, px = np.nonzero(sil)
        ys, xs = np.concatenate([ys, py]), np.concatenate([xs, px])
    crop = img[max(int(ys.min()) - 30, 0):int(ys.max()) + 30,
               max(int(xs.min()) - 30, 0):int(xs.max()) + 30]
    while True:
        cv2.imwrite(path, crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if os.path.getsize(path) <= CROP_KB * 1024 or crop.shape[1] < 80:
            return
        crop = cv2.resize(crop, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)


def stats(errs):
    errs = np.asarray(errs)
    if not len(errs):
        return "0 | - | - | - | - |"
    return "%d | %.2f | %.2f | %.2f | %.2f |" % (
        len(errs), np.median(errs), np.percentile(errs, 75), (errs < 2).mean(), (errs < 4).mean())


def tables(rows, misses, fp, submission):
    """Markdown tables for the analysis file."""
    ok = [r for r in rows if r["matched"]]
    L = ["Tables generated by `scripts/analyze_failures.py` for `%s`." % submission, "",
         "### Required instances (%d)" % len(rows), "",
         "| scene | idx | visib | tilt deg | holes | matched@10 | MSSD mm | score | "
         "also missed in |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %d | %.3f | %.0f | %d | %s | %s | %s | %s |" % (
            r["scene"], r["idx"], r["visib"], r["tilt"], r["holes"],
            "yes" if r["matched"] else "**no**", "%.1f" % r["mssd"] if r["matched"] else "-",
            "%.2f" % r["score"] if r["matched"] else "-",
            ", ".join(n.replace("train_", "") for n in r["also_missed"]) or "-"))
    L += ["", "### Missed at 10 mm (%d)" % len(misses), "",
          "| scene | idx | visib | nearest pred: MSSD (score) | rot to GT | dt mm | pred < 15 mm | "
          "fold seg IoU (conf, gate) | synthetic IoU (conf, gate) | crop | diagnosis |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in misses:
        segs = ["%.2f (%.2f, %s)" % (v[0], v[1], "pass" if v[2] else "fail")
                for v in (r["seg"].get("fold"), r["seg"].get("synthetic")) if v] or ["-", "-"]
        L.append("| %s | %d | %.3f | %.1f (%.2f) | %.0f deg about %s | %.1f | %s | %s | %s | %s | %s |" % (
            (r["scene"], r["idx"], r["visib"], r["near_mssd"], r["near_score"]) + r["near_delta"]
            + ("yes" if r["near_mssd"] < NEAR_MM else "no", segs[0], segs[-1], r["crop"], r["diag"])))
    L += ["", "### Matched at 10 mm but not at 4 mm (%d)" % sum(r["mssd"] >= 4 for r in ok), "",
          "| scene | idx | visib | tilt deg | holes | MSSD mm | rot err | dt mm | score |",
          "|---|---|---|---|---|---|---|---|"]
    L += ["| %s | %d | %.3f | %.0f | %d | %.1f | %.1f deg about %s | %.1f | %.2f |" % (
        (r["scene"], r["idx"], r["visib"], r["tilt"], r["holes"], r["mssd"])
        + r["match_delta"] + (r["score"],)) for r in ok if r["mssd"] >= 4]
    L += ["", "### Matched MSSD vs holes and tilt", "",
          "| bin | n | median mm | p75 mm | < 2 mm | < 4 mm |", "|---|---|---|---|---|---|"]
    bins = [("holes visible = %d" % k, lambda r, k=k: r["holes"] == k) for k in (2, 1, 0)]
    bins += [("tilt %s deg" % n, lambda r, a=a, b=b: a <= r["tilt"] < b)
             for n, a, b in (("0-15", 0, 15), ("15-45", 15, 45), ("> 45", 45, 91))]
    L += ["| %s | %s" % (name, stats([r["mssd"] for r in ok if sel(r)])) for name, sel in bins]
    L += ["", "### Tally", "",
          "- misses: %d duplicate labels, %d flips, %d segmenter misses, %d mislocalised, %d other" % (
              sum("duplicate" in r["diag"] for r in misses), sum("flip" in r["diag"] for r in misses),
              sum("no proposal" in r["diag"] for r in misses),
              sum("mislocalised" in r["diag"] for r in misses),
              sum("wrong pose" in r["diag"] for r in misses)),
          "- matched at 10 mm but not at 4 mm: %d of %d" % (sum(r["mssd"] >= 4 for r in ok), len(rows)),
          "- predictions: %d; unmatched at 10 mm: %d = %d on ignore regions + %d on optional "
          "instances (< 0.8 visible) + %d false positives, of which %d < %g mm from a GT pose "
          "(second pose of a found part), %d flips of a GT pose (> %g deg), %d other wrong "
          "poses of real parts" % (
              fp["n"], fp["unmatched"], fp["ignored"], fp["optional"], fp["fp"], fp["near"],
              NEAR_MM, fp["flip"], FLIP_DEG, fp["fp"] - fp["near"] - fp["flip"])]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=".")
    p.add_argument("--submission", default="results/train_ensemble_run1.json")
    p.add_argument("--out", default="analysis")
    p.add_argument("--runs", default="seg_runs_l", help="Per-fold segmenter runs")
    p.add_argument("--extra-weights", default="weights/part-seg-synthetic.pt")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "failures"), exist_ok=True)

    import trimesh
    from ultralytics import YOLO
    ply = os.path.join(args.root, "model", "3d_model.ply")
    hull = score.load_hull_vertices(ply)
    m = trimesh.load(ply, force="mesh")
    mesh = (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces))
    scenes = score.load_scenes(args.root, "train", os.path.join(args.root, "train"), mesh)
    mc = load_model_cloud(ply)
    with open(os.path.join(args.root, args.submission)) as fh:
        main_res = scene_matches(json.load(fh), scenes, hull, mesh)
    others = {n: scene_matches(json.load(open(f)), scenes, hull, mesh) for n in OTHER_RESULTS
              for f in [os.path.join(args.root, "results", n + ".json")] if os.path.exists(f)}

    rows, misses = [], []
    fp = dict.fromkeys(("n", "unmatched", "ignored", "optional", "fp", "near", "flip"), 0)
    for sid, data in scenes.items():
        res, preds = main_res[sid], main_res[sid]["preds"]
        matches = res["matches"][10.0]
        claimed = set(matches.values())
        for i, pred in enumerate(preds):     # unmatched predictions, binned like the misses
            fp["n"] += 1
            optional = i in claimed and data.visib[
                next(j for j, k in matches.items() if k == i)] < score.DEFAULT_VISIB_MIN
            fp["unmatched"] += i not in claimed or optional
            fp["optional"] += optional
            fp["ignored"] += i not in claimed and res["ignored"][i]
            if i not in claimed and not res["ignored"][i]:
                j = int(np.argmin(res["errors"][i]))
                fp["fp"] += 1
                fp["near"] += res["errors"][i, j] < NEAR_MM
                fp["flip"] += (res["errors"][i, j] >= NEAR_MM
                               and pose_delta(data.R[j], data.t[j], pred)[0] > FLIP_DEG)
        with open(os.path.join(args.root, "train", sid, "poses.json")) as fh:
            mask_files = [q["mask"] for q in json.load(fh)]
        for j, vis in enumerate(data.visib):
            if vis < score.DEFAULT_VISIB_MIN:
                continue
            R, t = data.R[j], data.t[j]
            tilt = np.degrees(np.arccos(abs((R @ mc.plate_axis)[2])))
            gt_mask = cv2.imread(os.path.join(args.root, "train", sid, mask_files[j]), 0) > 0
            hv = holes_visible(R, t, data.K, gt_mask, mc, tilt)
            i = matches.get(j)
            dup = [k for k in range(len(data.R)) if k != j and
                   score.mssd_mm(R, t, data.R[k], data.t[k], hull) < DUP_MM]
            # Nearest prediction: the closest unclaimed one (the pose that came
            # from this instance's own mask); a duplicate label's nearest is
            # its twin's match; overall nearest when nothing is unclaimed.
            free = [k for k in range(len(preds)) if k not in claimed or dup]
            near = min(free or range(len(preds)), key=lambda k: res["errors"][k, j], default=None)
            row = {"scene": sid, "idx": j, "visib": vis, "tilt": tilt, "holes": sum(hv),
                   "matched": i is not None, "dup": dup,
                   "mssd": res["errors"][i, j] if i is not None else np.nan,
                   "score": preds[i]["score"] if i is not None else np.nan,
                   "match_delta": pose_delta(R, t, preds[i]) if i is not None else None,
                   "near": preds[near] if near is not None else None,
                   "near_mssd": res["errors"][near, j] if near is not None else np.inf,
                   "near_score": preds[near]["score"] if near is not None else np.nan,
                   "near_delta": pose_delta(R, t, preds[near]) if near is not None else (np.nan, "-", np.nan),
                   "gt_mask": gt_mask,
                   "also_missed": [n for n, r in others.items() if j not in r[sid]["matches"][10.0]]}
            rows.append(row)
            if i is None:
                misses.append(row)

    models = {"synthetic": YOLO(os.path.join(args.root, args.extra_weights))} if args.extra_weights else {}
    for row in misses:
        sid, j, data = row["scene"], row["idx"], scenes[row["scene"]]
        rgb = cv2.imread(os.path.join(args.root, "train", sid, "rgb.png"))
        fold = next(f for f, ids in FOLD_VAL_SCENES.items() if sid in ids)
        models["fold"] = YOLO(os.path.join(args.root, args.runs, fold, "weights", "best.pt"))
        row["seg"] = best_proposals(models, rgb, row["gt_mask"], args.conf)
        if row["dup"]:
            twin = row["dup"][0]
            row["diag"] = "duplicate GT label of #%d (%s); one pose cannot claim both" % (
                twin, "twin matched" if twin in main_res[sid]["matches"][10.0] else "twin also missed")
        elif row["near_delta"][0] > FLIP_DEG:
            row["diag"] = "flip: nearest pose is a half-turn about model %s" % row["near_delta"][1]
        elif max(v[0] for v in row["seg"].values()) < 0.5:
            row["diag"] = "no proposal (segmenter miss)"
        elif row["near_mssd"] < NEAR_MM:
            row["diag"] = "mislocalised (in-plane drift)"
        else:
            row["diag"] = "proposal registered to a wrong pose"
        row["crop"] = "%s_inst%02d_mssd%.0fmm.png" % (sid, j, row["near_mssd"])
        draw_crop(rgb, data, mesh, j, row["near"], os.path.join(args.out, "failures", row["crop"]))

    text = tables(rows, misses, fp, args.submission)
    print(text)
    path = os.path.join(args.out, "failure_analysis.md")
    doc = open(path).read() if os.path.exists(path) else "# Failure analysis\n\n%s\n%s\n" % MARK
    if MARK[0] not in doc or MARK[1] not in doc:
        doc += "\n%s\n%s\n" % MARK
    head, rest = doc.split(MARK[0], 1)
    with open(path, "w") as fh:
        fh.write(head + MARK[0] + "\n" + text + "\n" + MARK[1] + rest.split(MARK[1], 1)[1])


if __name__ == "__main__":
    main()
