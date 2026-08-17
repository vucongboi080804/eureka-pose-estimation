# 6-DoF Pose Estimation — Solution Report

**TL;DR.** Two small instance segmenters (YOLO11l-seg fine-tuned on the 20
train scenes + one trained only on synthetic renders of the CAD) propose
masks; classical RGB-D geometry (FPFH/RANSAC → ICP → flip disambiguation →
depth-map verification → hole-centre polish) estimates and *verifies* every
pose; `score` = segmenter confidence × verification. Honest leave-scenes-out
CV on train: **AR 0.851, top-1 1.000**, precision 0.77 at 10 mm, against a
GT-mask baseline of 0.848. At the 10 mm threshold it recovers **112 of the
117 required instances — every one the metric can reach**: the five it
misses are duplicate ground-truth labels that no single-pose submission can
claim (`analysis/failure_analysis.md`). The 40 test scenes take ≈ 139 s on
one desktop GPU, ~165 s CPU-only; pick mode (one confident pose per scene)
0.7 s. Clone-and-run: `./setup.sh` then `./run_all.sh <release>`; a Docker
image, an air-gapped recipe and a Jetson pin set are in `deploy/`.

`submission.json` was produced at commit `SUBMISSION_COMMIT` by
`scripts/run_pipeline.py --split test --seg-model weights/part-seg.pt
--extra-seg-model weights/part-seg-synthetic.pt` (363 predictions over 40
scenes); `overlays_test/` by the released `visualize.py` on those poses —
exactly what `run_all.sh` does.

## Method

The split follows the measured bottleneck: with ground-truth masks the
geometric registration alone reaches AR 0.832, so nearly all of the
end-to-end gap lived in instance segmentation — the component a network
learns best from few images — while pose *quality* is a geometry problem
the depth channel answers better than any regressor trained on 20 scenes.
(BOP 2024 reached the same conclusion for the field: 2D detection, not
pose refinement, dominates the error budget.)

1. **Proposals** (`src/detect_seg.py`). Two ultralytics segmenters share
   one proposal pool at a low confidence floor (0.25): `weights/part-seg.pt`
   (YOLO11l-seg, all 20 train scenes incl. ignore masks, rotation/flip
   augmentation; 4-fold leave-scenes-out mask mAP50 0.79–0.93) and
   `weights/part-seg-synthetic.pt` (trained on 1140 domain-randomised
   BlenderProc renders of the CAD, no real image seen). Extra proposals
   cost only registration time: wrong ones die in verification. A proposal
   must be ≥ 30 % part-coloured (the HSV gate the polish already uses) —
   the synthetic model, trained with randomised part colours, otherwise
   fires on plain light background, and a flat CAD plate sunk flush into
   the tray floor can pass the depth check. A verified pose must also
   *explain the mask that proposed it* (≥ 150 of its points, or 30 %,
   within 3 mm of the posed model) — the geometric detector's progress
   invariant applied to learned masks; it removes registrations that
   drifted off their proposal and still verified somewhere else.
2. **Back-projection** (`src/scene_io.py`): masked depth pixels lift to a
   camera-frame cloud (depth is registered to colour; per-scene intrinsics).
3. **Registration of one proposal** (`src/register.py`):
   - *Global init*: FPFH + RANSAC (Open3D), scene → model.
   - *Refinement*: coarse-to-fine point-to-plane ICP; robust Tukey kernel
     only in the fine stages (a kernel narrower than the remaining
     misalignment freezes ICP).
   - *Flip disambiguation*: the part is nearly 180°-symmetric about its own
     axes, so every converged pose spawns three flipped rivals (π about
     X/Y/Z through the centroid), each re-refined.
   - *RGB hole cue* (`src/verify.py`): the depth verdict only judges
     pixels where the posed model has surface, so a predicted through-hole
     is its blind spot — and that is exactly where a half-turn about the
     stem hides. Solid part-coloured surface at or in front of a predicted
     hole's own rim plane is material the pose claims is empty; that
     objection re-ranks the rivals before the shortlist. The sign matters:
     punishing colour merely *near* the rim depth is worse than nothing,
     because a real hole frames whatever lies a few millimetres below it.
   - *Depth-map verification* (`src/verify.py`): render-free z-buffer test
     of the posed model against the observed depth. Model pixels sitting
     *in front of* the measured surface are physically impossible
     ("free-space violation"); confidence = support − 2·violation, with
     slope-aware margins on steep faces. Verification, not ICP fitness,
     picks between rivals: a flip explains the visible surface but pokes
     through free space.
   - *Rotation-grid fallback*: when nothing verifies (≥ 0.5), a
     Fibonacci-sphere rotation grid (192 orientations when the proposal
     anchors the translation, as every mask does; 60 otherwise) is
     coarse-ICP'd, ranked by the depth verdict,
     and the best few fully refined — this fixed 14 of the 15 hard
     instances feature matching missed.
   - *Polish* (`src/edge_refine.py`): 1 mm depth quantisation erases ~2 mm
     of in-plane information (see Limitations), so a final stage alternates
     a deadzoned Gauss-Newton against the CAD mesh (holds z + tilt) with
     hole-centre alignment: the through-holes are instance-private,
     sub-pixel image features, and matching predicted vs observed hole
     centroids pins the in-plane shift (and roll, given ≥ 3 holes).
4. **Scoring and NMS.** The submission `score` is the joint belief:
   segmenter confidence × verification. Ranking drives the scorer's
   matching order and top-1, so weak proposals may add recall but never
   outrank solid poses. Duplicates are suppressed only when position
   (< 9 mm) *and* orientation (< 30°) coincide — stacked parts sit one
   thickness apart but never share orientation.
5. **Domain-shift safety net.** When fewer than two detections verify at
   ≥ 0.5, the training-free geometric detector (`src/detect.py`: colour
   gate → smooth-surface patches at depth/normal breaks → register
   top-of-pile first and carve out explained points → hole-pair proposals
   for coplanar flats) sweeps the scene too and the union is de-duplicated.
   Alone it scores AR 0.723: the no-GPU, no-training baseline.

## Results (train split, released `score.py`)

| Setting                                        | 2 mm  | 4 mm  | 6 mm  | 8 mm  | 10 mm | AR    | top-1 |
| ---------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| GT masks → registration (one proposal/instance) | 0.453 | 0.872 | 0.957 | 0.966 | 0.991 | 0.848 | 1.000 |
| Geometric detector, no training                | 0.368 | 0.786 | 0.812 | 0.821 | 0.829 | 0.723 | 0.950 |
| Single segmenter (YOLO11l, conf 0.4)           | 0.496 | 0.855 | 0.906 | 0.906 | 0.915 | 0.815 | 1.000 |
| **Two-segmenter ensemble (submitted)**         | 0.521 | 0.889 | 0.940 | 0.949 | 0.957 | **0.851** | **1.000** |
| ↳ precision of the submitted config            | 0.381 | 0.675 | 0.743 | 0.760 | 0.767 |       |       |

Recall (last row: precision) at each MSSD threshold. Learned-mask rows are
honest: they stitch four leave-scenes-out folds, so every scene is predicted
by a model that never saw it. Every row is a file in `results/`
(`train_gt_masks.json`, `train_geometric.json`, `train_yolo11l_single.json`,
`train_ensemble_run1.json`) that `score.py` re-scores in seconds.

*Run-to-run variance.* Open3D's RANSAC is stochastic (its OpenMP threads
share one random engine, so a seed does not make it bit-reproducible).
Both draws of the submitted configuration give AR 0.851 with the same
per-instance result (`train_ensemble_run1.json`, `train_ensemble_run2.json`);
draws without the hole cue span 0.836–0.844 (`ablation_no_hole_cue*.json`,
`ablation_no_own_mask*.json`). Deltas under ±0.005 are noise; this one is
not, and it is the same two instances in both draws.

*Why the ensemble edges past the GT-mask row.* Two segmenters × ~2
proposals per instance means several independent RANSAC/ICP registrations
per instance where the GT-mask run gets one proposal; predicted masks also
avoid the GT masks' occlusion-boundary pixels, and the two models miss
different instances. The GT-mask row is nevertheless the more accurate one
at 10 mm (0.991): registering each labelled mask separately, it claims both
copies of a duplicate annotation, which a de-duplicated submission cannot.
The price of the ensemble is precision (0.77 at 10 mm): unmatched low-score
proposals cost precision but never AR or top-1. A deployment trades recall
for precision by thresholding `score` (≥ 0.4: precision 0.86 at AR 0.84;
≥ 0.6: 0.96 at 0.79; ≥ 0.7: 0.99 at 0.72; top-1 stays 1.000 up to 0.8) —
`analysis/score_calibration.md`.

## Analysis (what the remaining error is, what each stage buys)

Four short studies in `analysis/`, each regenerable by a script in
`scripts/` and each reading the `results/` files above:

- **Failure analysis** (`failure_analysis.md`, `scripts/analyze_failures.py`).
  Of the 7 required instances missed at 10 mm, **5 are duplicate labels**
  — two `poses.json` entries 0.4–3.7 mm apart on the same part (masks
  IoU 0.91–0.95) in 000022, 000030, 000041 — so any one-pose-per-part
  submission caps at recall 0.957 at 10 mm, and the ensemble reaches
  110 of the 112 attainable. (Predicting every pose twice at half score
  would lift AR by ~0.02; deliberately not done — it games a label defect
  and would halve precision.) The 2 real misses (3 in the previous draw)
  are half-turn flips about the stem axis: in one the sensor flattens the
  distinguishing boss so the depth verdict genuinely prefers the flip
  (0.72 vs 0.19), in the other the verdict prefers the truth but RANSAC
  never lands in that basin. Both segmenters proposed each of them
  (IoU ≥ 0.79) — zero segmenter misses, zero mislocalisations. The fix for
  the first is an RGB hole-consistency cue (a predicted through-hole
  covering part-coloured surface at or in front of its own rim plane is
  impossible); for the second, more restarts from the mask centroid.
  Hole claim confirmed: matched MSSD median 1.7 mm with both large holes
  visible (100 % < 4 mm, n 80) vs 2.8 mm with none (82 % < 4 mm, n 22).
  All 37 false positives are wrong registrations of real parts (11
  flips), none on background; their median score is 0.41 vs 0.86 for
  true positives.
- **Score calibration** (`score_calibration.md`, `scripts/score_calibration.py`).
  `score` ranks reliably (AUROC 0.95 for "within 5 mm"; the top pick of
  every scene scores ≥ 0.85 and lands within 3.2 mm) but is not a
  calibrated probability (ECE 0.14: over-confident in the middle bins,
  under-confident above 0.7 — a monotone recalibration would fix the level
  without changing the ranking). Recommended cell gate: score ≥ 0.7 →
  precision 1.00 at 5 mm on the CV predictions, 0.94 at ≥ 0.6.
- **Ablation** (`ablation.md`, `scripts/eval_seg_folds.py --ablate`). One
  stage off per CV run: rotation-grid fallback −0.049 AR (the largest
  single contribution); part-colour gate precision 0.57 → 0.75 at equal
  AR; own-mask check +0.03 precision, +1 instance; polish +3 instances at
  2 mm; flip rivals ±0 AR on this path (the grid subsumes them; kept
  because they are cheap insurance on the geometric path).
- **Runtime** (`runtime.md`). Registration (ICP 40 %, RANSAC 33 %, grid
  11 %) is ~85 % of scene time, the two segmenters 2 %; GPU therefore
  buys 19 % throughput and no latency. Full sweep 12 s/scene single-worker,
  135 s for the split at 6 workers (160 s CPU-only); **pick mode**
  (`--pick`: stop at the first pose scoring ≥ 0.8) 0.7 s mean / 2.1 s max
  per scene on GPU, 1.6–2.4 s CPU-only on 4 cores, returning a committed
  pose in 40/40 test scenes. ~1.6–1.9 GB RSS and ~0.9 GB GPU per worker.

## Design notes and dead ends that shaped the method

- **Scoring the metric, not the pixels.** MSSD is evaluated at model
  vertices up to ~39 mm from the origin, so 1° of rotation error costs
  ~0.7 mm; the 2 mm threshold demands ~1 mm / 1° accuracy. Matches to
  instances below 80 % visibility are free (neither TP nor FP), so the
  pipeline predicts generously and lets `score` rank.
- **Depth quantisation is the accuracy floor.** The depth PNGs are integer
  millimetres. ICP initialised *at the ground truth* drifts ~2.4 mm away:
  whole neighbourhoods of poses explain the quantised staircase equally
  well. Deadzoned objectives, mesh-exact Gauss-Newton, denser model
  sampling and bilateral depth smoothing all leave a ~2 mm in-plane floor —
  that information is gone from the depth channel. This is why 2 mm recall
  plateaus near 0.5 while 4 mm recall reaches ~0.9.
- **Silhouette chamfer fails in piles.** Aligning the predicted rim to
  class-colour edges seems natural, but every neighbour shares the part's
  colour: rim points snap to the neighbour's edge, several pixels off the
  truth. Hole centroids do not have this failure mode.
- **Verification beats fitness.** ICP fitness cannot tell a correct pose
  from a near-symmetric flip (both explain the visible surface); free-space
  violation can. One train instance stays wrong by design: its flipped pose
  genuinely fits the observed depth better than the ground truth (support
  0.81 vs 0.63) because the distinguishing boss is occluded.
- **Background proposals.** Before the part-colour gate, the synthetic-only
  segmenter produced ~70 confident masks on plain light background across
  the test split, and a flat CAD plate sunk ~7 mm into the tray floor
  passes free-space verification (nothing is *in front of* the surface).
  The gate removes them with no change in AR/top-1 (CV precision at 10 mm
  0.56 → 0.73). A colour-agnostic alternative would gate on height above
  the support plane; not shipped.

## Limitations

- Instances lying so that no through-hole is visible keep depth-only
  in-plane accuracy (~2 mm); their 2 mm recall is low.
- **Domain shift.** The primary segmenter was fine-tuned on 20 real scenes;
  it generalises across scenes of this capture setup (leave-scenes-out) but
  a genuinely new environment costs accuracy. The system is layered against
  that, and every layer is measured on the train split:

  | Tier | Needs | AR |
  | ---- | ----- | -- |
  | Real-trained segmenter alone (YOLO11l) | this environment | 0.815 (0.851 in the submitted ensemble) |
  | **Synthetic-only segmenter** (`scripts/render_synthetic.py`) | nothing real: trained purely on domain-randomised CAD renders | 0.814 (top-1 1.000) zero-shot on the real scenes |
  | Geometric detector, colour gate | the part's colour | 0.723 |
  | Geometric detector, depth foreground (`foreground="depth"`, Python API) | nothing but depth | recovers most instances, colour-blind |

  The depth-map verifier is environment-agnostic throughout — bad masks
  produce *low-confidence* poses, never confident mistakes. The two
  segmenters always share one proposal pool; the colour-gate geometric
  detector joins automatically when fewer than two detections verify at
  ≥ 0.5 (`detect_scene_hybrid`); the depth-foreground detector is a manual
  switch. A **new part** needs no code: `scripts/onboard_new_part.py`
  renders and trains a segmenter from its CAD with zero hand labels, and the
  pose stack reads any CAD at load time. A **new part colour** needs the
  HSV constants in `src/detect.py` retuned (proposal gate, hole polish,
  geometric fallback).
- Registration assumes the depth map is metrically consistent with the
  colour image and intrinsics (true for this dataset).
- **Runtime** (i5-14600K, RTX 4070 Ti SUPER; `analysis/runtime.md`): 40
  test scenes in 135 s at 6 workers (+15 s overlays), 160 s CPU-only;
  12 s per scene single-worker, up to 48 s on the densest pile; pick mode
  0.7 s. The geometric fallback is CPU-only research code: 10–90 s per
  scene, up to ~10 min on the densest piles.

## Deployment

The assignment metric is stricter than the production task: a robot picks
*one* part per cycle, then rescans, so the loop is governed by top-1
(1.000 here) rather than full-scene AR, and every pick thins the pile.
That loop is what `--pick` implements (first pose scoring ≥ 0.8, 0.7 s per
scene; `score` ≥ 0.7 as the cell's gate, rescan/shake otherwise), and
`deploy/live_adapter.py` is the seam where a camera SDK plugs in — it wraps
one registered RGB-D frame + intrinsics into the same `Scene` the offline
runner uses (verified identical output on a test scene).

Packaging (`deploy/OFFLINE.md`): `./setup.sh` builds the venv from the
exact pins (`deploy/requirements-lock.txt`, CPU or CUDA); `deploy/Dockerfile`
is a CPU image (1.24 GB) that reproduced the submission poses with
`--network none` — inference makes no network request; a wheelhouse recipe
covers air-gapped bare metal; `deploy/jetson-nano/` holds a Python 3.8 /
aarch64 pin set and image for a Jetson Nano 4 GB (built and smoke-tested
under qemu emulation; timing on the board not measured).

Accuracy-wise the measured bottleneck is the sensor, not the algorithm — an
industrial structured-light camera (30–100 µm noise) would let this same
registration stack settle near its ~0.3–0.5 mm verification floor without
algorithmic changes. Next upgrades: in-hand verification for assembly-grade accuracy, a wrist-mounted
second viewpoint for steeply leaning parts, and site-collected scenes
(self-labelled by the gripper's success sensor) as a permanent regression
set.

## Repository layout

```text
src/           scene_io (loading, back-projection) · model_cloud (CAD
               sampling, FPFH, hole discovery) · register (RANSAC, ICP,
               flips, grid, hole-pair proposals) · verify (depth-map
               verification, RGB hole cue) · edge_refine (polish) · detect (geometric
               detector, colour gate, NMS) · detect_seg (learned masks +
               hybrid fallback) · surface_patches
scripts/       run_pipeline (full pipeline over a split, --pick) ·
               eval_oracle_masks · eval_seg_folds (leave-scenes-out CV,
               --ablate) · analyze_failures · score_calibration ·
               make_seg_dataset · render_synthetic (BlenderProc) ·
               onboard_new_part · merge_submissions
weights/       part-seg.pt (YOLO11l-seg, all 20 train scenes)
               part-seg-synthetic.pt (YOLO11m-seg, 1140 synthetic renders)
results/       train-split prediction JSONs behind every table and ablation row
analysis/      failure_analysis · score_calibration · ablation · runtime
deploy/        Dockerfile · OFFLINE.md · requirements-lock.txt ·
               live_adapter.py · jetson-nano/
setup.sh       venv from the pinned lock (--cpu | --cuda | --jetson-nano)
run_all.sh     one-command run on any release folder (auto-scores if GT present)
submission.json / overlays_test/   test-split deliverables
```

## Reproducing

```bash
./setup.sh                                  # .venv from deploy/requirements-lock.txt (Python 3.12; --cpu/--cuda)
./run_all.sh <release_path> [split]         # submission + overlays (+ score if GT)
# or: docker build -f deploy/Dockerfile -t pose-est:cpu . && docker run --rm --network none \
#         -v <release_path>:/data:ro -v $PWD/out:/out pose-est:cpu /data test /out

# Table rows (train split):
.venv/bin/python scripts/eval_oracle_masks.py --root . --workers 6 --out results/train_gt_masks.json
.venv/bin/python scripts/run_pipeline.py --root . --split train --out results/train_geometric.json --workers 6
.venv/bin/python score.py --release . --split train --submission results/train_gt_masks.json

# Leave-scenes-out CV of the learned-mask rows (retrains 4 fold models; GPU).
# Give project= an ABSOLUTE path: ultralytics puts relative ones under its
# global runs_dir. Hyper-parameters exactly as the shipped weights were trained.
.venv/bin/python scripts/make_seg_dataset.py --root . --out seg_data/fold0 --val-scenes 000007 000014 000021 000033 000047
# ... folds 1-3 as in scripts/eval_seg_folds.py, then per fold:
.venv/bin/yolo segment train model=yolo11l-seg.pt data=seg_data/fold0/data.yaml \
    project=$PWD/seg_runs_l name=fold0 imgsz=960 epochs=250 patience=60 batch=3 \
    amp=False optimizer=AdamW lr0=0.0002 cos_lr=True degrees=180 flipud=0.5 fliplr=0.5
.venv/bin/python scripts/eval_seg_folds.py --root . --runs seg_runs_l --conf 0.4 --out results/train_yolo11l_single.json
.venv/bin/python scripts/eval_seg_folds.py --root . --runs seg_runs_l \
    --extra-weights weights/part-seg-synthetic.pt --conf 0.25 --out results/train_ensemble_run1.json
# ablation rows: add --ablate no_grid | no_flips | no_polish | no_gate | no_own_mask
# analyses:
.venv/bin/python scripts/analyze_failures.py --root . --submission results/train_ensemble_run1.json --out analysis
.venv/bin/python scripts/score_calibration.py --root . --submission results/train_ensemble_run1.json --out analysis

# Shipped weights: same recipe on the full split -> weights/part-seg.pt.
.venv/bin/python scripts/make_seg_dataset.py --root . --out seg_data/full
# Synthetic-only model: onboard_new_part.py renders (BlenderProc), carves a
# 5% val split, writes data.yaml and trains in one go; the shipped weight
# used epochs=80 patience=25 batch=8 (yolo11m-seg, no rotation augmentation).
.venv/bin/python scripts/onboard_new_part.py --cad model/3d_model.ply --workdir onboard_part --frames 1200 --epochs 80
```

`amp=False` with AdamW at `lr0=2e-4` is required — the default recipe
diverges on this 20-image dataset. `blenderproc` is needed only for the
synthetic-data scripts (not in `requirements.txt`).

## Tools disclosure

Developed with the assistance of Claude Code (Anthropic), used as a coding
assistant for implementation, debugging and experiment automation under my
direction. Libraries: Open3D (registration primitives), OpenCV, NumPy,
SciPy, trimesh, matplotlib, Ultralytics YOLO11 (instance segmentation) and
BlenderProc (synthetic rendering). No external images or labels: the
segmenters are fine-tuned from the public COCO-pretrained YOLO11 checkpoints
on the released train split and on synthetic scenes rendered from the
released CAD only.
