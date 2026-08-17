# Ablation of the pose pipeline

Leave-scenes-out CV on the 20 train scenes in the submitted configuration
(`scripts/eval_seg_folds.py --runs seg_runs_l --extra-weights
weights/part-seg-synthetic.pt --conf 0.25`), one stage switched off per row
via `--ablate`. Scored with the released `score.py`: recall at each MSSD
threshold, precision at 10 mm, AR = mean recall, top-1 = fraction of scenes
whose top-scored pose lands within 5 mm. Every row is a file in `results/`.

| Configuration | `results/` file | 2 mm | 4 mm | 6 mm | 8 mm | 10 mm | prec@10 | AR | top-1 | n preds |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full pipeline** (`--ablate none`, submitted) | `train_ensemble_run1.json` | 0.521 | 0.889 | 0.940 | 0.949 | 0.957 | 0.767 | **0.851** | 1.000 | 183 |
| no RGB hole cue | `ablation_no_hole_cue.json` | 0.521 | 0.897 | 0.932 | 0.932 | 0.940 | 0.748 | 0.844 | 1.000 | 185 |
| no own-mask check † | `ablation_no_own_mask.json` | 0.504 | 0.889 | 0.932 | 0.932 | 0.940 | 0.719 | 0.839 | 1.000 | 191 |
| no flip rivals | `ablation_no_flips_v2.json` | 0.530 | 0.889 | 0.940 | 0.949 | 0.957 | 0.747 | 0.853 | 1.000 | 186 |
| no rotation-grid fallback | `ablation_no_grid_v2.json` | 0.513 | 0.838 | 0.872 | 0.880 | 0.880 | 0.824 | 0.797 | 1.000 | 157 |
| no polish | `ablation_no_polish_v2.json` | 0.496 | 0.897 | 0.940 | 0.940 | 0.957 | 0.783 | 0.846 | 1.000 | 181 |
| no part-colour gate | `ablation_no_gate_v2.json` | 0.521 | 0.897 | 0.940 | 0.949 | 0.957 | 0.599 | 0.853 | 1.000 | 224 |
| single segmenter (YOLO11l, conf 0.4) | `train_yolo11l_single.json` | 0.496 | 0.855 | 0.906 | 0.906 | 0.915 | 0.863 | 0.815 | 1.000 | 152 |
| GT masks, one proposal per instance | `train_gt_masks.json` | 0.453 | 0.872 | 0.957 | 0.966 | 0.991 | 0.983 | 0.848 | 1.000 | 133 |
| second draw of the submitted configuration | `train_ensemble_run2.json` | 0.521 | 0.889 | 0.940 | 0.949 | 0.957 | 0.778 | 0.851 | 1.000 | 183 |
| no hole cue, two more draws | `ablation_no_hole_cue_run2.json`, `ablation_no_own_mask_run2/3.json` | 0.521 | 0.880 | 0.923 | 0.923 | 0.932 | 0.732 | 0.836–0.839 | 1.000 | 185 |

The pool holds 117 required instances (visible fraction ≥ 0.80), so one
instance is 0.0085 of recall at a threshold and 0.0017 of AR.

*Noise floor.* Open3D's RANSAC is stochastic (its OpenMP threads share one
random engine, so a seed does not make it bit-reproducible). Repeated draws
of the same configuration move AR by about ±0.005 and a single threshold's
recall by one or two instances: three draws without the hole cue and the
own-mask check give 0.839 / 0.836 / 0.836, four earlier draws spanned
0.829–0.844, and the two draws of the submitted configuration both give
0.851 (`train_ensemble_run1.json`, `train_ensemble_run2.json`). Deltas
inside that band are noise; only the larger ones below are read as
effects. Top-1 is 1.000 in every row.

† **The own-mask row predates the RGB hole cue**, so it has two stages off,
not one: compare it against `ablation_no_hole_cue.json` (AR 0.844), not
against the submitted row. Every other `no_*` row was measured against the
shipped configuration. (The `no_hole_cue` row is exact either way: with the
cue off, candidate ranking is the raw depth-confidence list it always was,
so a pre-cue draw is a valid no-cue row — which is why the two earlier
draws serve as its second and third measurements.) An earlier version of
this table compared four rows measured before the cue against a baseline
that had it, which inflated each of their deltas by whatever the cue was
worth; those four have been re-measured.

## What each stage buys

**RGB hole cue** (`hole_cue`, `HOLE_CUE_WEIGHT`, `src/verify.py:hole_conflict`). The depth verdict only judges pixels where the posed model has surface, so a predicted through-hole is a blind spot — and that is where a half-turn about the stem hides. Under a correct pose a see-through hole shows what lies *behind* the part; solid part-coloured surface at or in front of the hole's own rim plane is material the pose claims is empty, which is the same free-space argument the verifier already makes, applied to the one region it cannot see. The sign is what carries the signal: an earlier variant that punished part colour merely *near* the rim depth (|observed − rim| ≤ 5 mm) is worse than nothing, because in a dense bin a real hole frames a neighbour lying 3–5 mm below it (measured +3.1 to +4.7 mm behind the rim on 000019 #2, 000020 #1, 000059 #8) and the correct pose is punished for showing exactly what a hole should show; the wrong flips' conflicts sit on the other side of that plane (−2.3 to −26 mm). Turning the cue off costs recall at 10 mm 0.957 → 0.940 and precision 0.767 → 0.748 (two instances, both stem flips: 000041 #0 and 000047 #3), AR 0.851 → 0.844. The gain is not a redundant one — with the cue the only instances still missed at 10 mm are the five duplicate labels, i.e. the submission sits on the ceiling any one-pose-per-part entry can reach. Note it is a *ranking* change, not only a verifier patch: it is applied before the confidence shortlist, so on 000041 #0 — where the correct pose already wins on depth alone (0.744 vs 0.682) — it recovers the instance by demoting the flip candidate far enough that the correct RANSAC candidate survives `CONFIDENCE_TIE`, without any verdict having been wrong. Cost 2.71 ms per rendered candidate against `verify_pose`'s 0.74 ms, rendered only within `HOLE_CUE_REACH` of the lead: +2.5 % wall clock.

**Own-mask explanation check** (`OWN_MASK_MIN_FRACTION`,
`src/detect_seg.py`; new default). A verified pose must also explain the
mask that proposed it: at least 150 of the mask's points — or 30 % of them
for small masks — within 3 mm of the posed model. It mirrors the geometric
detector's progress invariant (`MIN_OWN_CONSUMED`): a pose that verifies
somewhere else does not deserve its proposal. The motivating case is a test
mask (scene 000053) cut off by the right image edge whose registration
drifted out of frame and settled the flat CAD plate on the tray floor:
free-space verification cannot object (nothing sits in front of the floor),
the pose scored 0.214, and it explained 0 of its own 4573 points. In CV the
check removes 6 of 191 predictions, all of them false positives (true
positives at 10 mm are 110 in both rows): precision at 10 mm 0.719 → 0.748,
AR unchanged within noise. It is a precision-only filter by construction —
it can never add a pose.

**Flip rivals** (`flips`, `src/register.py`). The part is nearly
180°-symmetric about its own axes, so ICP converges just as happily onto a
flipped pose; every converged pose spawns three π-rotated rivals, each
re-refined, and the depth verdict — not ICP fitness — picks between them
(report: *Verification beats fitness*). On the learned-mask path the
ablation shows no measurable loss without them (0.853 without against
0.851 with, inside the noise band), which
is consistent with the fallback covering the same failure: a flipped winner
verifies below 0.5, that triggers the rotation-grid sweep, and the grid
finds the right orientation instead. On this path the rivals are a
runtime trade rather than an accuracy component — and not a favourable
one: they cost three extra full refinements on *every* attempt, whereas
the grid runs only for the few proposals that need it. Timed on five test
scenes (registration only, two repeats each, shared machine so relative
numbers only): 197 s with flips, 132 s without, same predictions per scene
within one. Read together with the next row: flips and grid are redundant
with each other, not with nothing. The geometric detector's dependence on
them was not ablated, and the shipped default keeps them on.

**Rotation-grid fallback** (`grid`). When feature matching leaves nothing
verifying at ≥ 0.5, a Fibonacci-sphere grid of orientations (translation
anchored at the mask's closest-to-camera point) is coarse-ICP'd, ranked by
the depth verdict and the best few fully refined. This is the largest single
effect in the table: without it recall at 10 mm drops 0.957 → 0.880 (nine
instances) and AR 0.851 → 0.797. The grid is what makes the instances
feature matching misses registrable at all (report: it fixed 14 of the 15
hard instances on the geometric path). Precision rises without it (0.824)
only because the hard proposals then die instead of being solved.

**Polish** (`polish`, `src/edge_refine.py`). Integer-millimetre depth
erases ~2 mm of in-plane information; the polish alternates a deadzoned
Gauss-Newton against the CAD mesh with hole-centre alignment, whose
sub-pixel image features pin the in-plane shift. Its effect is confined to
the strictest threshold, as designed: 2 mm recall 0.496 → 0.521 (three
instances), with 4 mm and above unchanged or a shade better without it.
That is at the edge of the noise band, and
consistent with the report's finding that the depth quantisation floor —
not the refinement — bounds 2 mm recall near 0.5.

**Part-colour gate** (`colour_gate`, `MIN_PART_COLOUR_FRACTION`). The
synthetic-only segmenter is trained with randomised part colours and fires
on plain light background; a flat CAD plate sunk flush into the tray floor
passes free-space verification, and it also explains its (background)
mask's points, so the own-mask check does not catch it. Without the gate the
pipeline emits 224 predictions instead of 183 and precision at 10 mm falls
0.767 → 0.599; recall and top-1 are unchanged. Same conclusion as the report
(0.56 → 0.73 before the own-mask check existed): the gate is pure precision.

**Second segmenter** (`train_yolo11l_single.json` → the full pipeline). One
segmenter at conf 0.4 reaches AR 0.815; pooling the synthetic-only model's
proposals at conf 0.25 lifts it to 0.836–0.844. The gain is recall at every
threshold (10 mm 0.915 → 0.940): the two models miss different instances,
and several independent registrations per instance give RANSAC more draws.
The price is precision (0.863 → 0.748), paid by low-score proposals that
never outrank solid ones.

**GT masks** (`train_gt_masks.json`). One perfect proposal per labelled
instance, registration stack unchanged: the ceiling the *masks* impose,
AR 0.832 with the best 10 mm recall (0.974) and precision (0.974) of any
row. The learned-mask rows exceed it in AR because they register each
instance several times (better 2–4 mm recall) while trailing it at 10 mm
by four instances the predicted masks do not deliver.
