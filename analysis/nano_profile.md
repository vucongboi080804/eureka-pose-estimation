# Segmenter profile for a 4 GB Jetson Nano

The board this pipeline is being ported to (Jetson Nano 4 GB, JetPack 4.6,
four Cortex-A57 cores, RAM shared with the display) is smaller than the
desktop every published number was measured on. Two knobs shrink the
pipeline without retraining anything: **how many segmenters propose**
(`weights/part-seg.pt` alone, or with `weights/part-seg-synthetic.pt`) and
**what input side they run at** (`--imgsz`). This file prices both, so the
board ships a profile with a number attached instead of a guess.

Accuracy comes from the same leave-scenes-out CV as `analysis/ablation.md`
— four folds, every scene predicted by a model that never saw it — scored
with the released `score.py`. Latency and memory come from five test scenes
run one at a time, CPU-only, on four pinned cores.

## Accuracy: what each configuration costs

| Configuration | `results/` file | 2 mm | 4 mm | 6 mm | 8 mm | 10 mm | prec@10 | AR | top-1 | TP / FP @10 mm |
|---|---|---|---|---|---|---|---|---|---|---|
| **Two segmenters, 960** (shipped) | `train_ensemble_run1.json` | 0.521 | 0.889 | 0.940 | 0.949 | 0.957 | 0.767 | **0.851** | 1.000 | 112 / 34 |
| Two segmenters, 640 | `nano_dual_640.json` | 0.496 | 0.880 | 0.940 | 0.949 | 0.957 | 0.778 | 0.844 | 1.000 | 112 / 32 |
| One segmenter, 960 | `nano_single_960.json` | 0.479 | 0.863 | 0.932 | 0.932 | 0.940 | 0.866 | 0.829 | 1.000 | 110 / 17 |
| One segmenter, 768 | `nano_single_768.json` | 0.487 | 0.863 | 0.923 | 0.932 | 0.940 | 0.866 | 0.829 | 1.000 | 110 / 17 |
| One segmenter, 640 | `nano_single_640.json` | 0.496 | 0.863 | 0.923 | 0.923 | 0.932 | 0.845 | 0.827 | 1.000 | 109 / 20 |
| One segmenter, 640, second draw | `nano_single_640_run2.json` | — | — | — | — | — | — | 0.848 | 1.000 | — |

Recall at each MSSD threshold, precision at 10 mm, AR = mean recall, top-1 =
fraction of scenes whose top-scored pose lands within 5 mm, TP / FP as
`score.py` counts them at the 10 mm threshold. The pool holds 117 required
instances, so one instance is 0.0085 of recall at a threshold and 0.0017 of
AR.

**The noise band is not the same for these rows as for the shipped one.**
`analysis/ablation.md` quotes ±0.005 from repeated draws of the two-segmenter
configuration. Single-segmenter rows swing about three times wider:
`analysis/edge_model.md` measures the same single/640 configuration at
0.827 and 0.848 across two draws (mean 0.838), and a single nano model at
0.846 / 0.834 / 0.832. The mechanism is proposal count — one segmenter puts
about 1.3 masks on each instance against the ensemble's 1.55, so fewer
independent RANSAC draws reach each one and a borderline instance flips in
or out. **Read every single-segmenter row here as ±0.015, and treat each as
one draw rather than a settled number.** The rows were measured before that
was known; the conclusions below survive it, but two of them survive with a
different strength, and each says which.

Every row, from the repository root:

```
OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py --root . \
    --runs seg_runs_l --conf 0.25 --extra-weights weights/part-seg-synthetic.pt \
    --out results/train_ensemble_run1.json
OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py --root . \
    --runs seg_runs_l --conf 0.25 --extra-weights weights/part-seg-synthetic.pt \
    --imgsz 640 --out results/nano_dual_640.json
OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py --root . \
    --runs seg_runs_l --conf 0.25 --imgsz 960 --out results/nano_single_960.json
OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py --root . \
    --runs seg_runs_l --conf 0.25 --imgsz 768 --out results/nano_single_768.json
OMP_NUM_THREADS=3 .venv/bin/python scripts/eval_seg_folds.py --root . \
    --runs seg_runs_l --conf 0.25 --imgsz 640 --out results/nano_single_640.json
```

Each is scored with
`.venv/bin/python score.py --release . --split train --submission <file>`.
The shipped row is the command `analysis/ablation.md` already documents;
`--imgsz` defaults to 960, so omitting it and passing `960` are the same run.

**Input side is nearly free — and the wider band makes this conclusion
stronger, not weaker.** 960 → 640 costs 0.002 AR with one segmenter
(0.829 → 0.827) and 0.007 with two (0.851 → 0.844). Three indistinguishable
numbers spread across a band that is itself ±0.015 is exactly what "no
effect" looks like; if the input side mattered, a difference would have to
climb out of that band, and none does. Proposal counts barely move (185, 191
and 186 masks over the 20 CV scenes at 960, 768 and 640): the part fills
enough pixels that a 640-px letterbox still resolves it. Only the 2 mm
column shows a pattern, and it moves the *wrong* way (0.479 → 0.496), which
is the clearest sign it is noise rather than lost detail — pose accuracy at
2 mm is set by depth quantisation, not by mask sharpness (`report.md`).

**The second segmenter is the real accuracy knob — direction certain,
magnitude softer than these single draws suggest.** Dropping it costs 0.022
AR at 960 and 0.017 at 640 on the rows above: 10 mm recall falls 0.957 →
0.940 and → 0.932, two and three required instances. Both deltas are
measured against single draws of the noisier arm, and 0.022 against a ±0.015
spread is not a magnitude to quote to three decimals. The *direction* is
solid on more evidence than this table holds: across every single-segmenter
draw taken since (0.846 / 0.834 / 0.832 for the nano model, 0.827 / 0.848
for the downscaled large one, `analysis/edge_model.md`), all five sit below
both draws of the two-segmenter configuration (0.851 / 0.851). Call it about
one instance, not 0.022. What it takes with it is
narrower than "recall": the false positives halve too (34 → 17 at 960), so
precision rises 0.767 → 0.866. That shape fits what
`analysis/failure_analysis.md` measured on the shipped configuration — the
primary segmenter had already put a ≥ 0.79-IoU mask on every instance that
was missed — which makes the second model's real contribution a second,
independently seeded registration of an instance the first model also
proposed, not a proposal the first model lacks. Most of those extra
registrations land on parts that are already found and are discarded; two or
three of them are the run that finally verifies.

**Pick-mode margin thins without the ensemble.** Pick mode commits to the
first pose scoring ≥ `PICK_SCORE` (0.8). On the CV predictions the shipped
configuration's per-scene best score never drops below 0.854; with one
segmenter the worst scene sits at 0.805 (960), 0.812 (640) or 0.782 (768,
the one scene in these runs that would fall through to the geometric safety
net). Two or three scenes in twenty sit below 0.85 with one segmenter
against none with two. Nothing here is broken — the safety net exists for
exactly this — but a single-segmenter cell should expect an occasional
fall-through, and the fall-through is the expensive path.

## Latency and memory: what each configuration costs the board

Five test scenes (000001, 000003, 000015, 000043, 000052), one worker, one
process per row so the peak RSS is that row's own, `CUDA_VISIBLE_DEVICES=""`,
`OMP_NUM_THREADS=4`, pinned to four physical cores (`taskset -c 0,2,4,6`) to
stand in for the Nano's four A57s. Numbers in `results/nano_runtime.json`.

| Configuration | pick: mean / max [s] | full sweep: mean / max [s] | RSS after load [GB] | peak RSS [GB] |
|---|---|---|---|---|
| **Two segmenters, 960** (shipped) | 2.07 / 2.50 | 14.80 / 25.05 | 1.05 | 1.58 |
| Two segmenters, 640 | 1.99 / 3.01 | 13.42 / 22.37 | 1.05 | 1.50 |
| One segmenter, 960 | 1.71 / 1.98 | 5.43 / 9.10 | 0.97 | 1.44 |
| One segmenter, 768 | 0.89 / 1.40 | 6.26 / 9.74 | 0.97 | 1.38 |
| One segmenter, 640 | 1.13 / 1.56 | 5.63 / 8.48 | 0.97 | 1.31 |

Model + segmenter load is a one-off 2.8–3.9 s per process in every row.

These ten rows came from a throwaway harness, not a shipped entry point;
`scripts/run_pipeline.py --imgsz` has since been added, so every row now has
a one-command form. The two 960 rows are the ones to re-run first on the
board (add `--imgsz 640` for the 640 rows):

```
.venv/bin/python scripts/run_pipeline.py --split test --workers 1 --pick \
    --scenes 000001 000003 000015 000043 000052 \
    --seg-model weights/part-seg.pt [--extra-seg-model weights/part-seg-synthetic.pt]
```

**Time follows the proposal count, not the pixels.** Registration is ~96 %
of scene time (`analysis/runtime.md`), so what matters is how many masks
reach it: two segmenters produce 483 proposals over the 20 CV scenes against
186 for one, and the full sweep duly costs 2.4–2.7× more (13.4–14.8 s per
scene vs 5.4–6.3 s). Input side moves the full sweep by less than its own
scene-to-scene spread — 5.43, 6.26 and 5.63 s at 960, 768 and 640 are one
measurement's noise, because the segmenters were only 2 % of the time to
begin with.

**Memory splits the same way but with a smaller lever.** The whole span from
the heaviest to the lightest configuration is 1.58 → 1.31 GB: the second
segmenter costs 0.08 GB resident (0.97 → 1.05 GB after load) and the input
side costs 0.13 GB of transient activation (1.44 → 1.31 GB peak at 960 vs
640). The remaining ~1 GB is Python, torch, Open3D and the CAD clouds, and
no segmenter choice touches it. Practical consequence for a 4 GB board with
a shared display: one worker fits in every configuration, two fit in none.

## What to run on the board

**One segmenter at `--imgsz 640`, in pick mode** — and since this file was
written, `analysis/edge_model.md` has trained a segmenter for that slot:
`weights/part-seg-nano.pt` reaches the same accuracy as the downscaled large
model (0.837 against 0.838, both means of multiple draws) at 6.0 MB instead
of 55.9 and 49 ms instead of 619 ms per frame. **The board should run that
weight**; `deploy/board/config.nano.json` does.

Either way the trade is the same: AR around 0.84 against the shipped 0.851 —
roughly one required instance, and inside the ±0.015 band these rows carry —
with unchanged top-1 (1.000) and *better* precision (0.845 vs 0.767), for
62 % less scene time (5.6 s vs 14.8 s per full sweep) and 17 % less peak RSS
(1.31 GB vs 1.58 GB). The price is a thinner pick margin (worst scene 0.812
against 0.854) that will occasionally drop a cycle into the geometric safety
net.

If the cell turns out to have latency headroom on the real board, **two
segmenters at 640** is the better stop: it recovers AR to 0.844 (0.007 off
the shipped configuration, inside the noise band) and restores the pick
margin to 0.840, at 1.50 GB and 2.4× the sweep time. Going the other way —
below 640 — is not worth measuring on this evidence: 640 already costs
nothing in accuracy, and the time it would save sits in the 2 % of the
budget the segmenters own.

Both recommendations assume pick mode. A full sweep at 13–15 s per scene on
a *desktop* P-core is not a robot-cycle budget on an A57 in any
configuration; pick mode is the deployment path (`analysis/runtime.md`), and
the profile above only changes how much slack it has.

## What was not measured

- **The board itself.** Every number here comes from an x86 desktop
  (i5-14600K, Python 3.12, torch 2.13, Open3D 0.19) with four cores pinned
  and the GPU disabled. A Cortex-A57 is several times slower per core than a
  pinned P-core and the Nano's software stack is different (Python 3.8,
  torch 2.4.1, Open3D 0.18), so read the latency column as a *lower bound* on
  board time. **Both are now superseded for this profile**: the board itself
  measures 2.6-2.7 s per pick and 624 MB peak
  (`results/bench/board_nano640.json`), against the 5.63 s and 1.31 GB this
  table estimates from x86 and the 20.9 s and 740 MB the emulation estimated.
  The rows below stay as the cross-configuration comparison they were built
  for — the *relative* costs of imgsz and of the second segmenter — but the
  absolute latency and memory columns should be read as x86 figures, not as
  board ones. The memory column is an x86 estimate on x86 library versions
  (torch 2.13, open3d 0.19, numpy 2.5); the emulated aarch64 run in
  `results/bench/emulated_nano640.json` carries the board's own versions
  (torch 2.4.1, open3d 0.18, numpy 1.24) and peaks at 0.74 GB against
  1.09 GB for the same profile on x86, so that is the closer estimate of the
  two — still not a board measurement. The qemu-emulated arm64 run brackets it from the
  other side and is not a timing measurement either. The accuracy rows
  should carry across — same weights, same code — but only within the same
  ±0.015 band these single-segmenter rows carry, since a different Open3D
  draws RANSAC differently; the
  latency and memory rows carry across not at all and have to be
  re-measured on the board.
- **A genuinely smaller backbone.** "One segmenter" here means the
  YOLO11l-seg fold models with the synthetic-trained second model removed,
  not a smaller network. The two knobs priced above are the two that need no
  retraining; a YOLO11m/n-seg backbone needs the four fold models retrained
  before it can be evaluated honestly, and this file does not cover it.
  Whatever it saves comes out of the 2 % of the time budget and the 0.08 GB
  the segmenters own — the ~1 GB Python/Open3D floor and the registration
  cost do not follow it down.
- **Quantisation, TensorRT and half precision.** Untouched. On this
  workload they act on the segmenters, i.e. on 2 % of the time and 0.08 GB
  of the memory.
- **Machine quiet during the timing runs.** The desktop was shared: the
  1-minute load average was 4.20–6.59 at the start of the ten timed rows
  (recorded per row in `results/nano_runtime.json`). That is quiet enough
  for the 2.4× and 17 % effects above to be real, and too noisy to read
  anything into the sub-second differences between the three single-segmenter
  input sides.
