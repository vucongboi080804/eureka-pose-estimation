# 6-DoF Pose Estimation — Solution Report

Multi-instance 6-DoF pose estimation of a known rigid part from RGB-D
tray captures, evaluated by MSSD recall (AR over 2–10 mm thresholds).

## Method overview

The pipeline is a hybrid: a small learned segmenter proposes instances,
and classical RGB-D geometry estimates and *verifies* every pose. The
split follows the measured bottleneck — with ground-truth masks the
geometric registration alone reaches AR 0.832, so nearly all of the
end-to-end gap lived in instance segmentation, which is exactly the
component a network learns best from few images. (The BOP 2024 evaluation
reached the same conclusion for the field at large: 2D
detection/segmentation, not pose refinement, dominates the error budget.)

Per scene:

1. **Instance segmentation** (`src/detect_seg.py`). A YOLO11m-seg model
   fine-tuned on the 20 train scenes (all labelled masks plus ignore
   masks; rotation/flip augmentation) proposes instance masks. Honest
   4-fold leave-scenes-out evaluation: mask mAP50 0.82–0.93 per fold.
   A geometric fallback (`src/detect.py`) needs no GPU and no training:
   an HSV colour gate (the part is saturated orange-red; ~98% pixel
   recall) splits piles into smooth surface patches at depth steps and
   normal breaks, registers them top-of-pile first, and carves out
   explained points — it scores AR 0.723 on its own and remains the
   no-training baseline.
2. **Back-projection** (`src/scene_io.py`). Masked depth pixels lift to a
   camera-frame point cloud (depth is registered to colour; intrinsics per
   scene).
3. **Seeded multi-instance extraction** (`src/detect.py`). Within each
   component, repeatedly: take the point nearest the camera as a seed (top
   of pile = least occluded = the instances the metric requires), carve a
   55 mm sub-cloud around it, register it against the CAD model, verify,
   accept, and remove the points the accepted pose explains. Every round
   either accepts an instance or buries a dead seed, so the sweep
   terminates. Saturated cardboard patches that pass the colour gate die
   here: no pose over them ever verifies.
4. **Registration of one instance** (`src/register.py`):
   - *Global init*: FPFH features + RANSAC (Open3D), scene→model.
   - *Refinement*: coarse-to-fine point-to-plane ICP (robust Tukey kernel
     only in the fine stages — a robust kernel narrower than the remaining
     misalignment freezes ICP).
   - *Flip disambiguation*: the part is nearly 180°-symmetric about its own
     axes, so every converged pose spawns three flipped rivals (π about
     X/Y/Z through the model centroid); each is re-refined.
   - *Depth-map verification* (`src/verify.py`): render-free z-buffer test
     of the posed model against the observed depth. Pixels where the model
     sits *in front* of the measured surface are physically impossible
     ("free-space violation"); confidence = support − 2·violation.
     Verification, not ICP fitness, picks between rivals — a flip explains
     the visible surface but pokes through free space.
   - *Rotation-grid fallback*: when nothing verifies (≥ 0.5), brute-force
     60 orientations (Fibonacci directions × rolls), coarse-ICP each with
     centroids aligned, fully refine the best few. This fixed 14 of the 15
     hard instances that feature matching missed.
   - *Polish* (`src/edge_refine.py`): depth quantised to 1 mm erases ~2 mm
     of in-plane information (see Limitations); a final stage alternates a
     deadzoned point-to-plane Gauss-Newton against the actual CAD mesh
     (holds z + tilt) with hole-centre alignment — the part's through-holes
     are instance-private, sub-pixel image features; matching predicted vs
     observed hole centroids pins the in-plane shift (and roll, given ≥ 3
     holes).
5. **Scoring and NMS**. The submission `score` is the verification
   confidence, so ranking (which drives matching order and top-1) prefers
   well-verified poses. Duplicates are suppressed only when both position
   (< 9 mm) and orientation (< 30°) nearly coincide — stacked parts sit one
   thickness apart but never share orientation.

## Results (train split, released scorer)

| Setting                              | 2 mm  | 4 mm  | 6 mm  | 8 mm  | 10 mm | AR    | top-1 |
| ------------------------------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Oracle masks (registration ceiling)  | 0.436 | 0.872 | 0.940 | 0.940 | 0.974 | 0.832 | 1.000 |
| Geometric detector (no training)     | 0.368 | 0.786 | 0.812 | 0.821 | 0.829 | 0.723 | 0.950 |
| **Learned masks (submitted config)** | 0.487 | 0.855 | 0.906 | 0.906 | 0.915 | 0.814 | 1.000 |

Recall at each MSSD threshold, from the released `score.py` on the train
split. "Oracle masks" feeds the ground-truth masks to registration,
isolating pose quality from detection. The learned-mask row is an honest
number: it stitches four leave-scenes-out folds, so every scene is
predicted by a model that never saw it. Its precision at 10 mm is 0.86.

The learned-mask pipeline essentially closes the detection gap (0.814 vs
the 0.832 oracle ceiling) and even beats the oracle at 2 mm — predicted
masks avoid the ground-truth masks' occlusion-boundary pixels, giving
registration cleaner clouds. Unioning the geometric detector on top adds
only +0.003 AR while halving precision, so the submission uses learned
masks alone; the geometric pipeline stands as the training-free fallback
and the source of the verification machinery both paths share.

## Design notes and dead ends that shaped the method

- **Scoring the metric, not the pixels.** MSSD is evaluated at convex-hull
  vertices ~43 mm from the origin, so 1° of rotation error costs ~0.75 mm
  of MSSD; the 2 mm threshold therefore demands ~1 mm / 1° accuracy.
  Matches to instances below 80% visibility are free (neither TP nor FP),
  so the pipeline predicts generously and lets `score` rank.
- **Depth quantisation is the accuracy floor.** The depth PNGs are integer
  millimetres. ICP initialised *at the ground truth* drifts ~2.4 mm away:
  whole neighbourhoods of poses explain the quantised staircase equally
  well. Deadzoned objectives, mesh-exact Gauss-Newton, denser model
  sampling, and bilateral depth smoothing all still leave a ~2 mm in-plane
  floor — that information is simply gone from the depth channel. This is
  why 2 mm recall plateaus near 0.45 while 4 mm recall reaches ~0.9.
- **Silhouette chamfer fails in piles.** Aligning the predicted rim to
  class-colour edges seems natural, but every neighbour shares the part's
  colour: rim points match the neighbour's edge, and the chamfer minimum
  sits several pixels off the truth. Hole centroids do not have this
  failure mode, which is why the polish uses them instead.
- **Verification beats fitness.** ICP fitness cannot tell a correct pose
  from a near-symmetric flip (both explain the visible surface); free-space
  violation can. One instance in train remains wrong by design: its flipped
  pose genuinely fits the observed depth better than the ground truth does
  (support 0.81 vs 0.63) because the distinguishing boss is occluded.

## Limitations

- Instances lying so that no through-hole is visible keep depth-only
  in-plane accuracy (~2 mm); their 2 mm-threshold recall is low.
- **Domain shift.** The segmenter was fine-tuned on 20 scenes; it
  generalises across scenes of this capture setup (proven by
  leave-scenes-out scoring) but will degrade under a genuinely new
  environment — different lighting, backgrounds, camera, or part colour.
  Two safeguards are built in: the depth-map verifier is
  environment-agnostic, so bad masks produce *low-confidence* poses
  rather than confident mistakes; and when fewer than two detections in a
  scene verify well, the training-free geometric detector automatically
  sweeps the scene too (`detect_scene_hybrid`). Verified by simulating a
  total segmenter failure: the fallback recovers the scene at nearly full
  quality. The production-grade fix is synthetic PBR training data
  rendered from the CAD (BlenderProc) with domain randomisation.
- The geometric fallback's colour gate assumes the part stays saturated
  orange-red and the background stays dull; a different part colour needs
  the two HSV constants retuned.
- Registration assumes the depth map is metrically consistent with the
  colour image and intrinsics (true for this dataset).
- Runtime: the learned-mask path takes ~5–10 s per scene (GPU inference +
  CPU registration). The geometric fallback is CPU-only research code:
  typically 10–90 s per scene, up to ~10 minutes on the densest piles.

## Repository layout

```text
src/
  scene_io.py     scene loading, depth back-projection
  model_cloud.py  CAD sampling, FPFH prep, through-hole discovery
  register.py     single-instance registration: RANSAC, ICP, flips,
                  grid fallback, hole-pair proposals, candidate selection
  verify.py       depth-map verification (support / free-space violation)
  edge_refine.py  final polish: deadzoned depth GN + hole-centre alignment
  detect.py       geometric detector: colour gate, surface patches, NMS
  detect_seg.py   learned-mask detector (ultralytics), same registration
  surface_patches.py  smooth-surface splitting of piles
scripts/
  eval_oracle_masks.py  registration ceiling on GT masks
  run_pipeline.py       full pipeline over a split (+ labels for visualize.py)
  make_seg_dataset.py   train scenes -> YOLO-seg dataset (with folds)
  eval_seg_folds.py     honest leave-scenes-out scoring of learned masks
  merge_submissions.py  union of two submissions with NMS dedup
weights/
  part-seg.pt     YOLO11m-seg fine-tuned on all 20 train scenes
report.md         this file
submission.json   test-split predictions
overlays_test/    predicted poses drawn on every test scene
```

## Toward production deployment

The assignment metric is stricter than the production task. A robot picks
*one* part per cycle, then the scene changes: the natural loop is pick the
highest-confidence instance, rescan, repeat. That loop is governed by
top-1 (0.95–1.00 here), not full-scene AR, and every pick thins the pile
so the hard instances become easy ones. The submission `score`
(verification confidence) is designed to gate this: picks below a
threshold are skipped in favour of a rescan or another instance.

A staged rollout would follow the usual cell-integration path — calibrate
(intrinsics, depth↔RGB registration, hand–eye), collect on-site scenes as
a permanent regression set, run in shadow mode against operators, then
integrate with the grasp planner and PLC error paths (low confidence →
rescan/shake; empty tray → signal). In operation, the gripper's own
success sensor labels every pick for free, which feeds drift monitoring
and continuous evaluation.

On accuracy, the measured bottleneck is the sensor, not the algorithm:
depth here is quantised to 1 mm, and that alone erases ~2 mm of in-plane
pose information (ICP initialised at the ground truth drifts ~2.4 mm).
An industrial structured-light camera (30–100 µm noise) would let this
same registration stack settle near its ~0.3–0.5 mm verification floor
without algorithmic changes. Beyond that, the highest-value upgrades are
in-hand verification before placement when assembly-grade accuracy is
needed, a learned segmenter (trained on the self-labelled site data)
replacing the colour gate for robustness to part/background changes, and
a wrist-mounted second viewpoint for steeply leaning parts.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Validate registration alone (ground-truth masks):
.venv/bin/python scripts/eval_oracle_masks.py --root . --workers 6
.venv/bin/python score.py --release . --split train --submission oracle_masks.json

# Geometric pipeline on train (no GPU, no training), scored:
.venv/bin/python scripts/run_pipeline.py --root . --split train --out pipeline_train.json --workers 6
.venv/bin/python score.py --release . --split train --submission pipeline_train.json

# Honest leave-scenes-out scoring of the learned-mask pipeline
# (retrains four fold models; needs a GPU):
.venv/bin/python scripts/make_seg_dataset.py --root . --out seg_data/fold0 --val-scenes 000007 000014 000021 000033 000047
# ... folds 1-3 as in scripts/eval_seg_folds.py, then per fold:
.venv/bin/yolo segment train model=yolo11m-seg.pt data=seg_data/fold0/data.yaml \
    project=seg_runs name=fold0 imgsz=960 epochs=250 patience=80 batch=4 \
    amp=False optimizer=AdamW lr0=0.0002 cos_lr=True degrees=180 flipud=0.5 fliplr=0.5
.venv/bin/python scripts/eval_seg_folds.py --root . --runs seg_runs --out seg_train.json
.venv/bin/python score.py --release . --split train --submission seg_train.json

# Test submission + overlays (uses the shipped weights):
.venv/bin/python scripts/run_pipeline.py --root . --split test --out submission.json \
    --labels-out pred_test --workers 6 --seg-model weights/part-seg.pt
.venv/bin/python visualize.py --root . --split test --labels pred_test --save overlays_test/
```

## Tools disclosure

Developed with the assistance of Claude Code (Anthropic), used as a coding
assistant for implementation, debugging and experiment automation under my
direction. Libraries: Open3D (registration primitives), OpenCV, NumPy,
SciPy, trimesh, matplotlib, and Ultralytics YOLO11 (instance
segmentation, fine-tuned on the released train split only — no external
data).
