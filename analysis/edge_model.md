# Edge segmenter: a small model trained at the deployment resolution

The shipped segmenter is YOLO11l-seg at 960 px (`weights/part-seg.pt`,
55.9 MB, 27.62 M parameters), paired with a second 45.2 MB synthetic-only
model. The deployment target is a Jetson Nano 4 GB (4× Cortex-A57,
JetPack 4.6, Python 3.8, aarch64, CPU torch), where 101 MB of weights and
two 960 px forward passes per frame are the wrong shape for the budget.

Two different things are easy to confuse, so both are measured here:

- **Downscaling the big model's input** — run `part-seg.pt` at 640 instead
  of 960. The weight file does not shrink; only activations do.
- **Training a small model at the resolution it will serve at** — YOLO11n-seg
  fine-tuned at 640 on the same leave-scenes-out folds, the subject of this
  file (`weights/part-seg-nano.pt`, `results/nano_yolo11n_640.json`).

The measured answer, stated up front because it is not the expected one:
**end-to-end pose accuracy is the same either way** (AR 0.837 vs 0.838
across repeated draws), so the 9.3× smaller weight is free. What the board
actually gives up is the ~0.014 AR carried by the *second* segmenter, not
by the first one's size.

## What was trained

Same recipe as the shipped weights (`report.md`, "Reproducing"), changing
only the checkpoint, `imgsz` and `batch` — verified field-by-field against
`seg_runs_l/*/args.yaml`: `amp=False optimizer=AdamW lr0=0.0002
cos_lr=True degrees=180 flipud=0.5 fliplr=0.5 epochs=250 patience=60`.
Nothing diverged; every fold trained to a normal early stop (180–250
epochs), and the whole set took 6.5 minutes on the RTX 4070 Ti SUPER.

```bash
# per fold k = 0..3, then the deployment weight on the full split
.venv/bin/yolo segment train model=yolo11n-seg.pt data=seg_data/fold<k>/data.yaml \
    project=$PWD/seg_runs_n name=fold<k> imgsz=640 epochs=250 patience=60 batch=8 \
    amp=False optimizer=AdamW lr0=0.0002 cos_lr=True degrees=180 flipud=0.5 fliplr=0.5
.venv/bin/yolo segment train model=yolo11n-seg.pt data=seg_data/full/data.yaml \
    project=$PWD/seg_runs_n name=full imgsz=640 ...     # -> weights/part-seg-nano.pt
```

Held-out mask mAP50 per fold (each fold's own val scenes, best epoch from
`seg_runs_*/fold<k>/results.csv`):

| Fold | val scenes | YOLO11l @960 | YOLO11n @640 |
|---|---|---|---|
| fold0 | 000007 000014 000021 000033 000047 | 0.929 | 0.892 |
| fold1 | 000008 000019 000022 000040 000054 | 0.875 | 0.836 |
| fold2 | 000009 000020 000026 000041 000058 | 0.881 | 0.843 |
| fold3 | 000010 000011 000023 000030 000059 | 0.790 | 0.771 |
| **mean** | | **0.869** | **0.836** |

The small model is genuinely a worse segmenter — 0.033 mask mAP50 down, and
down on all four folds, not one. The interesting result is that this does
not reach the poses; see the next section.

`yolo11n-seg.pt` (the COCO-pretrained starting checkpoint, 6.2 MB) was
**downloaded by ultralytics at train time**. The deployment recipe is
air-gapped (`deploy/OFFLINE.md`), so the board must receive the *trained*
weight; retraining on the board is not a supported path.

## Accuracy

Leave-scenes-out CV over the 20 train scenes: every scene is predicted by
the fold model that never saw it, scored by the released `score.py`.
AR = mean recall over the five MSSD thresholds; top-1 = fraction of scenes
whose top-scored pose lands within 5 mm; prec@10 = precision at 10 mm.
All rows use `conf 0.25` except the one marked, which is `report.md`'s
original `conf 0.4` measurement and is included only so this table joins
onto that one.

**`n/inst`** is predictions in the results file ÷ the 117 required
instances — the proposals-per-instance number, listed beside AR on purpose.
Aggregate AR hides a shift toward fewer registrations per instance, and
that shift is exactly what happens when masks get coarser or a segmenter is
removed. `masks/inst` is raw segmenter masks per required instance, before
the size, colour and own-mask filters, counted from the evaluation logs.

| Configuration | `results/` file | weight(s) | params | 2 mm | 4 mm | 6 mm | 8 mm | 10 mm | prec@10 | AR | top-1 | n preds | n/inst | masks/inst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **YOLO11n @640, single (this work)** | `nano_yolo11n_640.json` | **6.0 MB** | **2.84 M** | 0.504 | 0.880 | 0.940 | 0.949 | 0.957 | **0.882** | 0.846 | 1.000 | 155 | 1.32 | 1.45 |
| ↳ same config, draw 2 (scratchpad) | – | | | 0.504 | 0.872 | 0.923 | 0.932 | 0.940 | 0.880 | 0.834 | 1.000 | 151 | 1.29 | – |
| ↳ same config, draw 3 (scratchpad) | – | | | 0.504 | 0.863 | 0.923 | 0.932 | 0.940 | 0.873 | 0.832 | 1.000 | 151 | 1.29 | – |
| YOLO11n @640 + synthetic model | `nano_yolo11n_640_dual.json` | 6.0 + 45.2 MB | 2.84 + 22.36 M | 0.487 | 0.872 | 0.932 | 0.949 | 0.957 | 0.752 | 0.839 | 1.000 | 182 | 1.56 | 3.99 |
| YOLO11l @640 (shipped weight, downscaled) | `nano_single_640.json` | 55.9 MB | 27.62 M | 0.496 | 0.863 | 0.923 | 0.923 | 0.932 | 0.845 | 0.827 | 1.000 | 157 | 1.34 | 1.59 |
| ↳ same config, draw 2 (scratchpad) | – | | | 0.513 | 0.880 | 0.940 | 0.949 | 0.957 | 0.875 | 0.848 | 1.000 | 158 | 1.35 | – |
| YOLO11l @768 (shipped weight, downscaled) | `nano_single_768.json` | 55.9 MB | 27.62 M | 0.487 | 0.863 | 0.923 | 0.932 | 0.940 | 0.866 | 0.829 | 1.000 | 156 | 1.33 | 1.63 |
| YOLO11l @960, single (native) | `nano_single_960.json` | 55.9 MB | 27.62 M | 0.479 | 0.863 | 0.932 | 0.932 | 0.940 | 0.866 | 0.829 | 1.000 | 161 | 1.38 | 1.58 |
| YOLO11l @960, single, conf 0.4 | `train_yolo11l_single.json` | 55.9 MB | 27.62 M | 0.496 | 0.855 | 0.906 | 0.906 | 0.915 | 0.863 | 0.815 | 1.000 | 152 | 1.30 | – |
| YOLO11l @640 + synthetic model | `nano_dual_640.json` | 55.9 + 45.2 MB | 27.62 + 22.36 M | 0.496 | 0.880 | 0.940 | 0.949 | 0.957 | 0.778 | 0.844 | 1.000 | 177 | 1.51 | – |
| **Shipped two-segmenter ensemble @960** | `train_ensemble_run1.json` | 55.9 + 45.2 MB | 27.62 + 22.36 M | 0.521 | 0.889 | 0.940 | 0.949 | 0.957 | 0.767 | **0.851** | 1.000 | 185 | 1.58 | – |
| ↳ second draw | `train_ensemble_run2.json` | | | 0.521 | 0.889 | 0.940 | 0.949 | 0.957 | 0.778 | 0.851 | 1.000 | 180 | 1.54 | – |
| GT masks (mask ceiling, reference) | `train_gt_masks.json` | – | – | 0.453 | 0.872 | 0.957 | 0.966 | 0.991 | 0.983 | 0.848 | 1.000 | 135 | 1.15 | – |

Exact commands behind the two new committed rows:

```bash
# results/nano_yolo11n_640.json
cd <root> && OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py \
    --root . --runs seg_runs_n --conf 0.25 --imgsz 640 \
    --out results/nano_yolo11n_640.json
.venv/bin/python score.py --release . --split train --submission results/nano_yolo11n_640.json

# results/nano_yolo11n_640_dual.json
cd <root> && OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py \
    --root . --runs seg_runs_n --conf 0.25 --imgsz 640 \
    --extra-weights weights/part-seg-synthetic.pt \
    --out results/nano_yolo11n_640_dual.json
.venv/bin/python score.py --release . --split train --submission results/nano_yolo11n_640_dual.json
```

### The noise floor is wider on these rows than on the ensemble

Open3D's RANSAC is stochastic, so one draw is not a measurement.
`analysis/ablation.md` puts the band at about ±0.005 AR, and that holds for
the *ensemble* rows (two draws of the submitted configuration both give
exactly 0.851). It does **not** hold for these single-segmenter rows:

| Configuration | AR draws | mean | spread |
|---|---|---|---|
| YOLO11n @640, single | 0.846 / 0.834 / 0.832 | **0.837** | 0.014 |
| YOLO11l @640, single | 0.827 / 0.848 | **0.838** | 0.021 |
| Shipped ensemble @960 | 0.851 / 0.851 | 0.851 | 0.000 |

A single segmenter gives ~1.3 proposals per instance where the ensemble
gives ~1.55, so several instances hang on one RANSAC draw apiece instead of
two or three, and the run-to-run swing roughly triples. Two consequences,
both of which matter for reading the table above:

1. **The committed `results/nano_yolo11n_640.json` (AR 0.846) is the
   luckiest of three draws.** The honest single-number figure for that
   configuration is **AR ≈ 0.837**, and this file quotes 0.837 wherever a
   claim rests on it. The committed file is kept as-is because it is the
   one the exact command above reproduces on a first run; the other two
   draws live in the scratchpad and are not committed.
2. **The apparent "training small beats downscaling big" effect is not
   real.** The first draw of each suggested +0.019 AR for the nano; with a
   second L@640 draw the two means are 0.837 and 0.838 — a difference of
   0.0002, far inside a 0.014–0.021 spread. Any comparison between these
   configurations built on single draws, in either direction, is noise.

### Reading the table

1. **Shrinking the segmenter 9.3× costs no measurable pose accuracy.** At
   640 px the 2.84 M nano and the 27.62 M L weight land on the same AR
   (0.837 vs 0.838) with identical top-1 (1.000), despite the nano being
   0.033 mask mAP50 worse as a segmenter. This is consistent with the
   report's own framing: the GT-mask ceiling is only AR 0.848, so mask
   quality above a fairly low bar stops being the binding constraint and
   the depth-verification stack absorbs the difference.
2. **Input resolution barely matters for the L weight either.** 960 → 768
   → 640 moves AR 0.829 → 0.829 → 0.827/0.848. The activation-memory knob
   is close to free on accuracy.
3. **What does cost accuracy is dropping the second segmenter.** Every
   single-segmenter row sits ~0.014 below the ensemble's 0.851, and that
   0.851 is the one figure in the table with a zero-spread repeat. The
   gap is small but it is the only one that survives the noise analysis.
4. **Precision moves the other way, hard.** The nano single reaches
   prec@10 0.873–0.882 against the ensemble's 0.767–0.778. One segmenter
   at a 0.25 floor emits ~155 predictions where two emit ~183; the extra
   ~28 are overwhelmingly false positives that never outrank a good pose.
5. **Top-1 is 1.000 in every configuration and every draw**, including all
   three nano draws — which is the metric the pick-one-part-per-cycle robot
   loop actually runs on.
6. **The proposals-per-instance column is where the real change is.** The
   nano single runs 1.29–1.32 registrations per required instance off
   1.45 raw masks, against the ensemble's 1.54–1.58 off 3.99. It is a
   thinner detector: cheaper, more precise, and with less redundancy left
   for a scene it reads badly. That thinness is the mechanism behind both
   the higher precision and the wider AR spread — they are the same fact
   seen from two sides.
7. **A second segmenter does not rescue the nano.** Pooling
   `part-seg-synthetic.pt` gives AR 0.839 (one draw, inside the nano's own
   spread) while precision collapses 0.882 → 0.752, and it costs 45.2 MB
   and 317 ms of extra CPU per frame. On the board it buys nothing.

## Deployment cost (CPU only)

**How measured.** `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, single
worker, pinned to four P-cores (`taskset -c 0,2,4,6`) to stand in for the
board's four Cortex-A57s. Five test scenes — 000001 000003 000015 000043
000052, the same set `analysis/runtime.md` profiles. Segmenter rows time
`masks_from_model` directly, 15 calls after a discarded warm-up; pipeline
rows time `detect_scene_hybrid` per scene; peak RSS is
`resource.getrusage(RUSAGE_SELF).ru_maxrss` in the same process. Host as
in `analysis/runtime.md` (i5-14600K, 31 GB, Python 3.12.3, torch
2.13.0+cu130, ultralytics 8.4.120), commit `332c758`.

**Machine quietness.** The gate was 1-minute load < 6 with no competing
process over 50 % CPU; it passed at **load1 3.27** and the run started
immediately. Load read 3.3–4.9 during the timed rows and 6.5 at the end —
that rise is the bench's own four pinned cores. An earlier pass of the same
bench was discarded because another agent's qemu-aarch64 job was burning
~2.7 cores through it; the numbers below are the clean repeat.

### Segmenter-only latency per frame

| Weight | imgsz | mean | median | max | RSS after load | peak RSS |
|---|---|---|---|---|---|---|
| `weights/part-seg.pt` (shipped) | 960 | 619.5 ms | 613 ms | 711 ms | 0.999 GB | 1.518 GB |
| `weights/part-seg.pt` (shipped) | 640 | 287.9 ms | 285 ms | 319 ms | 0.997 GB | 1.351 GB |
| **`weights/part-seg-nano.pt`** | 640 | **49.3 ms** | **46 ms** | 64 ms | **0.886 GB** | **1.170 GB** |
| `weights/part-seg-synthetic.pt` | 640 | 317.0 ms | 314 ms | 408 ms | 0.971 GB | 1.322 GB |

The nano is **12.6× faster than the shipped segmenter at its native 960**
and 5.8× faster than the same L weight fed 640 px — i.e. most of the win is
the model, not the resolution. The shipped *pair* costs 619.5 + ~500 ms
(the synthetic model at 960) per frame against the nano's 49.3 ms.

### Full per-scene pipeline latency and peak RSS

| Configuration | mode | mean | median | max | poses/scene | peak RSS |
|---|---|---|---|---|---|---|
| **nano single @640** | full sweep | **5.56 s** | 4.83 s | 11.94 s | 9 13 8 15 5 | **1.159 GB** |
| L single @960 | full sweep | 7.34 s | 8.27 s | 11.89 s | 9 13 7 14 5 | 1.461 GB |
| **L dual @960 (shipped)** | full sweep | 15.05 s | 15.49 s | 26.10 s | 9 14 10 16 6 | 1.619 GB |
| nano dual @640 | full sweep | 13.44 s | 12.06 s | 23.85 s | 9 13 9 17 7 | 1.427 GB |
| **nano single @640** | pick | **0.73 s** | 0.34 s | **1.46 s** | 1 1 1 1 1 | **1.172 GB** |
| L single @960 | pick | 1.37 s | 1.34 s | 1.89 s | 1 1 1 1 1 | 1.497 GB |
| **L dual @960 (shipped)** | pick | 2.00 s | 2.21 s | 2.49 s | 1 1 2 2 1 | 1.606 GB |

Against the shipped configuration the nano single is **2.7× faster in both
modes** (15.05 → 5.56 s full sweep; 2.00 → 0.73 s pick) and holds **0.43–
0.46 GB less peak RSS**. Note that against the *single* L model at 960 the
full-sweep gain is only 7.34 → 5.56 s: registration is ~85 % of scene time
(`analysis/runtime.md`), so a 570 ms/frame segmenter saving is diluted.
Most of the 2.7× therefore comes from running one model instead of two —
fewer masks proposed means fewer RANSAC/ICP registrations, which is the
same fact as the `n/inst` column. In **pick mode**, where the pipeline
stops at the first pose scoring ≥ 0.8, the segmenter is a much larger share
and the nano's 0.73 s mean / 1.46 s worst case is the number a robot cell
would feel.

### Model size and parameter count

| File | bytes | size | params |
|---|---|---|---|
| `weights/part-seg.pt` | 55 889 185 | 55.9 MB | 27.62 M |
| `weights/part-seg-synthetic.pt` | 45 188 790 | 45.2 MB | 22.36 M |
| **`weights/part-seg-nano.pt`** | **6 015 268** | **6.0 MB** | **2.84 M** |
| `weights/part-seg-nano.onnx` | 11 582 684 | 11.6 MB | – |

The board goes from 101.1 MB of weights across two models to 6.0 MB in one
— **16.9×** on disk and on resident model memory.

## Recommendation for the 4 GB board

**Ship the single `weights/part-seg-nano.pt` at `imgsz=640`, and drop the
second segmenter on this target.** In one sentence: the board gives up
about **0.014 AR** (0.837 vs 0.851) — the value of the second segmenter,
not of the big backbone, and roughly one instance at each threshold — and
in exchange gets **16.9× less weight (101.1 MB → 6.0 MB), 2.7× lower
per-scene latency (pick 2.00 s → 0.73 s CPU-only on four cores), 0.43 GB
lower peak RSS (1.606 → 1.172 GB)**, while *gaining* about 0.11 precision
at 10 mm and holding top-1 at 1.000, which is the metric the
pick-one-part-then-rescan loop actually runs on.

The configuration is the pipeline's existing learned-mask path with
`--seg-model weights/part-seg-nano.pt`, no `--extra-seg-model`, and
`imgsz=640`. Nothing else changes: the part-colour gate, own-mask check,
rotation-grid fallback, depth verification, RGB hole cue and the
training-free geometric safety net are untouched, so the safety argument in
`report.md` still holds — the verifier does not care which network proposed
a mask, and a bad mask still yields a low-confidence pose rather than a
confident mistake.

Two caveats stated plainly. First, **this rests on leave-scenes-out CV of
this capture setup**; a 2.84 M-parameter model has less capacity to absorb
a domain shift than a 27.62 M one, and nothing here measures that. Second,
**dropping `part-seg-synthetic.pt` removes one of the layered domain-shift
tiers** from the default proposal pool (`report.md`, Limitations) — the
colour-gate geometric detector still joins automatically when fewer than
two detections verify at ≥ 0.5, but the zero-shot synthetic model would no
longer be resident. A cell moved to new lighting or a new background should
re-measure before trusting the single-model configuration, and the
synthetic model remains the right thing to add back if it does.

If AR matters more than memory on some future board, the cheapest way back
to 0.851 is to re-add the second segmenter, not to re-grow the first: at
640 px the L backbone buys nothing the nano does not already deliver.

## ONNX export

```bash
cd <root> && .venv/bin/yolo export model=weights/part-seg-nano.pt \
    format=onnx imgsz=640 opset=12 simplify=False
```

`weights/part-seg-nano.onnx`, **11.58 MB** (11 582 684 bytes). Graph
validated with `onnx.checker.check_model`: opset 12, IR version 7, 389
nodes, static input `images [1,3,640,640]`, outputs `output0 [1,37,8400]`
(detection head) and `output1 [1,32,160,160]` (mask prototypes). Static
shapes throughout, which is what a TensorRT builder wants.

The export **auto-installed `onnx` 1.22.0** (plus `protobuf` 7.35.1 and
`ml-dtypes` 0.6.0) into the development venv — ultralytics fetches it on
demand and this needed network. Those packages are not in
`deploy/requirements-lock.txt`, are not needed at inference time, and
`setup.sh` rebuilds the venv from the lock file, so the shipped environment
is unaffected.

**`onnxruntime` is not installed and was not installed**, so *the ONNX file
has never been loaded or executed by any runtime* — only structurally
validated. No ONNX inference latency is claimed anywhere in this file.

**TensorRT conversion must happen on the board, and is untested here.** A
TensorRT engine is built against a specific GPU architecture, TensorRT
version and driver, so an engine built on this desktop (RTX 4070 Ti SUPER,
CUDA 13) would not load on the Nano's Maxwell GPU under JetPack 4.6 /
TensorRT 8.2. The conversion has to run on the Nano itself, and there
JetPack 4.6 ships TensorRT 8.2 with **Python 3.6 bindings only** while the
deployment stack is Python 3.8 (`deploy/jetson-nano/README.md`) — so it
would have to go through the `trtexec` CLI rather than the Python API.
None of that was attempted or verified. Note also that the GPU serves only
~2 % of scene time in this pipeline (`analysis/runtime.md`), so a TensorRT
engine would accelerate a small slice of the work.

## What was NOT measured

- **The board itself.** No number in this file was taken on a Jetson Nano.
  Every timing is an x86 i5-14600K pinned to four P-cores, which is a
  *lower bound* on Cortex-A57 time, not an estimate of it. The
  emulation-vs-native ratios measured elsewhere in this Jetson work are
  wildly non-uniform by stage (~512× on the segmenters against ~15× on
  registration), so applying any single scaling factor to these numbers
  would be wrong on at least one half of the pipeline.
- **TensorRT.** No engine was built, on the board or off it. No TensorRT
  latency, memory or accuracy figure is claimed.
- **ONNX runtime latency.** `onnxruntime` is absent by instruction; the
  `.onnx` file was validated structurally, never executed.
- **Peak RSS under a 4 GB limit.** RSS was measured on a 31 GB machine
  with no cgroup limit and no memory pressure. Behaviour under the Nano's
  zram swap and cgroup v1 limits is not characterised here.
- **Test-split accuracy for the nano.** The nano rows are train-split CV
  only. No nano submission was produced for the 40 test scenes;
  `submission.json` and `overlays_test/` are untouched.
- **Domain shift for the nano backbone**, and the cost of removing the
  synthetic segmenter from the layered fallback — see the caveats above.
- **Repeat draws for several comparison rows.** The @768, @960 and both
  dual @640 rows are single draws; given the 0.014–0.021 spread measured
  on single-segmenter configurations, differences between them of less
  than ~0.02 AR should not be read as effects.
- **Quantisation.** No INT8 or FP16 variant was trained, exported or
  scored.
- **Pick-mode accuracy for the nano.** Pick latency was measured, but
  whether the nano's picked pose agrees with the full sweep's (the
  40/40 agreement check `analysis/runtime.md` runs for the shipped
  config) was not re-run.
