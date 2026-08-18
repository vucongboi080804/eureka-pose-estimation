# Failure analysis of the submitted configuration (train split)

Per-instance post-mortem of `results/train_ensemble_run1.json` (two-segmenter
ensemble with the RGB hole cue, leave-scenes-out CV; AR 0.851, recall
0.957 at 10 mm). Matching,
visibility and ignore handling are `score.py`'s own (imported), so every count
below agrees with the official scorer. Regenerate the tables and the crops
with

    .venv/bin/python scripts/analyze_failures.py --root . \
        --submission results/train_ensemble_run1.json --out analysis

Conventions: *tilt* is the angle between the part's plate normal
(`ModelCloud.plate_axis`, model Y) and the optical axis, 0° = lying flat;
*holes* counts the two large through-holes (the ones the polish uses) whose
rim projects inside the GT mask; model axes are X = bar, Y = plate normal,
Z = stem. "Also missed in" compares with the other result files
(`train_ensemble_run2`, `train_yolo11l_single`, `train_synthetic_only`,
`train_gt_masks`), all matched the same way.

## Headline

117 required instances (visible ≥ 0.8), **112 matched at 10 mm — every one
the metric can reach**. The 5 that remain are all duplicate annotations,
so the pipeline's own error at 10 mm is zero; earlier configurations also
missed the stem-axis flips the RGB hole cue now settles (rows below are
kept because they explain what the cue does).

| group | count | instances | consistent across runs? | data limit or fixable |
| ----- | ----- | --------- | ----------------------- | --------------------- |
| duplicate GT labels: a second `poses.json` entry within 4 mm MSSD of another, with its own near-identical mask | 5 | 000022 #4 (twin #5), 000022 #8 (twin #2), 000030 #3 and #6 (twins of #8), 000041 #5 (twin #7) | yes: every NMS'd run misses exactly these | data: one pose can claim one instance; only duplicate predictions could match them |
| stem-axis flips: the output pose is a half-turn (177–180°) about model Z, 6–9 mm off | 2 (+1 in the earlier draw) | 000041 #0, 000047 #3 (and 000014 #1 in the earlier draw) | yes: missed in run2, YOLO-single and synthetic-only; 000047 #3 also with GT masks | 1 verifier-blind (000047 #3, RGB cue fixes it: 73 → 6 mm), 2 search-reachability (000041 #0 and 000014 #1: from the GT mask the depth verdict already prefers the truth, 0.744 vs 0.682 — the pipeline's RANSAC candidates never reach that basin) |

So the attainable recall for any pipeline that submits one pose per part is
112/117 = 0.957 at every threshold, and the ensemble now reaches 112/112 of
it at 10 mm. No required instance was left without a proposal (the fold
segmenter puts a mask of IoU ≥ 0.79 on every missed instance, the synthetic
one ≥ 0.68 on all but one, all passing the size and colour gates), none was
mislocalised by 10–15 mm, and none of the near-misses is a segmentation
problem: **the whole remaining 10 mm gap is labels (5)**; before the hole cue it was
labels (5) plus two stem-axis flips.

## The flips

All three cases seen across draws (two in this draw; 000014 #1 was missed in
the earlier draw and found in this one) are the same geometric ambiguity: the T is almost symmetric under a
half-turn about its stem; the half-turn swaps the bar's two ends, i.e. puts
the boss where the end hole is and vice versa (and flips the plate over).
Depth-map verification (`src/verify.py`, GT pose vs the output pose, same
scene depth) says which side each case falls on:

| instance | GT pose: support / violation / confidence | output pose (flip) | reading |
| -------- | ----------------------------------------- | ------------------ | ------- |
| 000014 #1 (boss down, tilt 18°) | 0.89 / 0.06 / **0.76** | 0.79 / 0.20 / 0.39, score 0.37 | verifier is right, the search never reached the GT basin: the flipped output puts the boss over the far hole (measured 720 mm vs model 668 mm there) and still won. Reproducible with predicted masks (4 of 4 learned-mask runs), found from the GT mask. Search failure: fixable (retry the flip rivals with the mask centroid as translation anchor when confidence < 0.5; the rotation grid did not recover it either). |
| 000041 #0 (boss up, tilt 8°, visib 0.82) | 0.66 / 0.15 / 0.36 | 0.82 / 0.06 / **0.69**, score 0.66 | the boss end is under a leaning neighbour (measured 674–711 mm across the boss footprint), so the only depth evidence left is the visible end hole, and the sensor mostly fills it (708–719 mm inside the 12.6 mm hole against a plate top of 709 mm), within the verifier's margin. Depth cannot decide; the RGB can — the flip predicts no hole where a dark hole is visible. |
| 000047 #3 (boss up, tilt 7°) | 0.66 / 0.24 / 0.19 | 0.84 / 0.06 / **0.72**, score 0.67 | the RGB shows the 11 mm boss standing (side wall visible, `failures/000047_inst03_mssd74mm.png`), the depth does not: a flat 708–710 mm over the boss top where the plate reads 712 mm and the model expects 699 mm. Sensor artefact (the boss is flattened to ~3 mm); the verifier then prefers the boss-down flip, and the GT-mask run makes the same choice. Fixable by an RGB cue: the flip predicts a through-hole on solid part-coloured pixels (the boss top). |

Only 000047 #3 is a verdict failure in the strict sense: re-running the
three π-rivals from the *ground-truth* mask, the depth verifier prefers the
correct pose for 000041 #0 (0.744 vs 0.682) and only loses on 000047 #3,
where the sensor flattens the boss. The other two are search-reachability
failures — the basin exists, RANSAC on the predicted mask does not land in
it. Read the row below as "what the verifier sees when the search reaches
both rivals". One of the three is therefore a data/sensor limit **for a
depth-only verifier**,
not for the pipeline: an RGB hole-consistency term when ranking flip rivals
(observed dark holes inside the proposal mask that a rival leaves
unexplained, predicted holes landing on part-coloured pixels: `edge_refine`
already extracts both) settles it: on an isolated re-run the cue turns
000047 #3 from 73.3 mm into 6.2 mm and leaves 000041 #0 at its
search-limited 4.2 mm. Ranking the three π-rivals of every one of the 117
ground-truth poses, the correct one wins 116/117 without the cue and
117/117 with it at any weight between 0.2 and 0.8 — an upper bound, since
the pipeline ranks RANSAC candidates rather than GT-refined ones. The submission `score` already
flags one of them (0.37) as doubtful; the other two flips are confident
(0.66, 0.67) because the depth genuinely supports them.

## Duplicate labels

Four train scenes carry duplicate annotations: 000022 (#2 = #8 at 0.6 mm,
#4 = #5 at 3.6 mm), 000030 (#3 = #6 = #8 within 0.6 mm), 000041 (#5 = #7 at
1.1 mm), 000058 (#8 = #9 at 3.7 mm, both optional at 0.79 visible). Their
masks overlap at IoU 0.91–0.95, so both copies count as required. The
`train_gt_masks` row (0.991 at 10 mm) matches them only because it registers
every GT mask separately and never de-duplicates; the ensemble's NMS
(< 9 mm and < 30°) merges the twin proposals and forfeits 5 of 117 at every
threshold (−0.043 AR). Submitting a half-score clone of every pose recovers
4 of the 5 (the 000030 triple needs a third copy): scored with `score.py`,
AR 0.836 → 0.860 on the earlier draw, 10 mm recall 0.966, top-1 unchanged,
precision 0.73 → 0.36. Not done — it games a labelling defect rather than estimating poses —
but it is the single largest lever left on this metric and worth a
conscious decision if the test labels share the defect.

## Accuracy vs holes and tilt (matched instances)

The report's claim that hole-less parts sit at the depth floor holds:

| bin | n | median mm | p75 mm | < 2 mm | < 4 mm |
| --- | - | --------- | ------ | ------ | ------ |
| both large holes visible | 79 | 1.76 | 2.14 | 0.67 | 1.00 |
| one visible | 8 | 2.96 | 3.44 | 0.38 | 0.88 |
| none visible | 22 | 2.82 | 3.74 | 0.23 | 0.77 |
| tilt 0–15° | 56 | 1.89 | 2.59 | 0.59 | 0.95 |
| tilt 15–45° | 32 | 1.88 | 2.33 | 0.62 | 1.00 |
| tilt > 45° | 21 | 2.51 | 3.26 | 0.38 | 0.86 |

With both holes the polish pins the pose to a 1.7 mm median and every one
lands under 4 mm; without holes the median is 2.8 mm and a fifth exceed
4 mm. Tilt matters only through the holes: 15–45° is as good as flat, and
the > 45° bin is the standing/edge-on parts, 18 of whose 21 show no hole.
All 5 instances matched at 10 mm but not at 4 mm have ≤ 1 hole visible; 3 of them stand at 82–90° with 4–10 mm error from a 7–9° tilt error
about the bar or plate normal (the depth sees only a thin edge). The CAD
also has two small (r 1.4 mm) stem holes the polish ignores; counting all
four and splitting at ≥ 3 visible moves one instance between bins and no
statistic by more than 0.05, because on this data every instance with < 3
holes visible is one with no large hole visible.

## False positives

185 predictions; 73 unmatched at 10 mm = 23 on ignore regions + 16 on
optional (< 0.8 visible) instances — both free — + 34 counted false
positives (precision 0.767). None of the 34 lies within 15 mm of a GT pose
(no double detections), 8 are half-turn flips of a real part (the losing
rival kept because NMS keeps different orientations), 26 are other wrong
registrations of real proposals straddling neighbours in piles; none sits
on background. Their median `score` is 0.41 against 0.86 for the true
positives, which is why thresholding `score` buys precision cheaply
(precision 0.86 at 10 mm for `score` ≥ 0.4, `score_calibration.md`).

## Crops

`failures/<scene>_inst<idx>_mssd<nearest>mm.png`: RGB crop, GT silhouette
in green, nearest unclaimed prediction in red (each ≤ 150 KB).

| file | what it shows |
| ---- | ------------- |
| 000014_inst01_mssd75mm.png | flip; boss down; the red pose's boss sits over the visible end hole |
| 000041_inst00_mssd73mm.png | flip; boss end hidden under the leaning neighbour |
| 000047_inst03_mssd74mm.png | flip; boss visibly standing in RGB, invisible in depth |
| 000022_inst04_mssd3mm.png, 000022_inst08_mssd3mm.png, 000041_inst05_mssd1mm.png | duplicate labels: green and red coincide (the red pose is the twin's match, 1.3–3.1 mm) |
| 000030_inst03_mssd10mm.png, 000030_inst06_mssd10mm.png | duplicate labels of an edge-on part whose single pose is itself 9.7 mm off |

## Where the remaining error is

Segmentation: 0 of 5 misses (every missed instance had a ≥ 0.79 IoU
proposal from the fold segmenter, and both segmenters proposed all but
one); the segmenters' cost shows only in precision (34 FPs, all wrong
registrations of real proposals). Registration/verification: 0 misses at
10 mm in this draw — the two stem-axis flips of the earlier draws (one the
depth channel cannot arbitrate, one the search never reached) are settled
by the RGB hole cue — leaving the 4–10 mm tail (8 instances) and the
2 mm-recall plateau, both of which trace to the sensor: 1 mm quantisation,
hole floors smeared, an 11 mm boss flattened to 3 mm. Labels: 5 misses are
duplicate annotations no single-pose submission can match. Net: the whole
0.043 recall gap at 10 mm is unreachable by design; the remaining accuracy
gap (2 mm recall 0.52) is the depth sensor, and holes are the only lever
the RGB gives.

<!-- tables:start -->
Tables generated by `scripts/analyze_failures.py` for `results/train_ensemble_run1.json`.

### Required instances (117)

| scene | idx | visib | tilt deg | holes | matched@10 | MSSD mm | score | also missed in |
|---|---|---|---|---|---|---|---|---|
| 000007 | 0 | 0.894 | 8 | 2 | yes | 3.4 | 0.92 | - |
| 000007 | 1 | 0.892 | 12 | 2 | yes | 2.5 | 0.93 | - |
| 000007 | 2 | 0.907 | 6 | 2 | yes | 1.0 | 0.90 | - |
| 000007 | 3 | 0.944 | 6 | 2 | yes | 1.2 | 0.88 | - |
| 000007 | 4 | 0.878 | 86 | 0 | yes | 2.3 | 0.42 | - |
| 000008 | 0 | 0.980 | 11 | 2 | yes | 1.5 | 0.93 | - |
| 000008 | 1 | 0.929 | 9 | 2 | yes | 3.3 | 0.94 | - |
| 000008 | 2 | 0.975 | 5 | 2 | yes | 1.6 | 0.92 | - |
| 000008 | 3 | 0.948 | 19 | 2 | yes | 2.8 | 0.96 | - |
| 000009 | 0 | 0.881 | 11 | 1 | yes | 1.9 | 0.80 | - |
| 000009 | 1 | 0.950 | 37 | 2 | yes | 1.7 | 0.69 | - |
| 000009 | 2 | 0.974 | 5 | 2 | yes | 1.2 | 0.88 | - |
| 000009 | 3 | 0.901 | 21 | 2 | yes | 1.7 | 0.79 | - |
| 000009 | 5 | 0.942 | 36 | 2 | yes | 1.6 | 0.87 | - |
| 000009 | 6 | 1.020 | 81 | 0 | yes | 1.3 | 0.64 | - |
| 000010 | 0 | 0.950 | 11 | 2 | yes | 1.0 | 0.80 | - |
| 000010 | 1 | 0.958 | 21 | 2 | yes | 1.9 | 0.93 | - |
| 000010 | 2 | 0.897 | 17 | 2 | yes | 1.3 | 0.89 | - |
| 000010 | 3 | 0.929 | 6 | 2 | yes | 2.1 | 0.86 | - |
| 000010 | 4 | 0.974 | 11 | 2 | yes | 2.0 | 0.86 | - |
| 000010 | 5 | 0.938 | 78 | 0 | yes | 2.5 | 0.69 | - |
| 000011 | 0 | 0.983 | 82 | 0 | yes | 1.9 | 0.51 | - |
| 000011 | 1 | 0.922 | 13 | 2 | yes | 1.6 | 0.88 | - |
| 000011 | 2 | 0.972 | 11 | 2 | yes | 1.1 | 0.88 | - |
| 000011 | 3 | 0.921 | 7 | 2 | yes | 1.6 | 0.90 | - |
| 000011 | 4 | 0.898 | 16 | 2 | yes | 3.5 | 0.89 | - |
| 000014 | 0 | 0.965 | 14 | 2 | yes | 2.0 | 0.93 | - |
| 000014 | 1 | 0.962 | 18 | 2 | yes | 3.5 | 0.91 | yolo11l_single, synthetic_only |
| 000014 | 2 | 0.974 | 13 | 2 | yes | 2.0 | 0.88 | - |
| 000014 | 3 | 0.972 | 6 | 2 | yes | 1.7 | 0.70 | - |
| 000019 | 0 | 0.959 | 9 | 2 | yes | 1.5 | 0.91 | - |
| 000019 | 1 | 0.883 | 7 | 2 | yes | 1.1 | 0.87 | - |
| 000019 | 2 | 0.933 | 5 | 1 | yes | 3.2 | 0.94 | - |
| 000019 | 3 | 0.854 | 7 | 1 | yes | 4.6 | 0.85 | - |
| 000019 | 4 | 0.856 | 32 | 2 | yes | 1.6 | 0.84 | - |
| 000020 | 0 | 0.956 | 6 | 2 | yes | 0.8 | 0.90 | - |
| 000020 | 1 | 0.941 | 14 | 2 | yes | 3.1 | 0.92 | - |
| 000020 | 2 | 0.955 | 27 | 2 | yes | 0.9 | 0.77 | - |
| 000020 | 4 | 0.943 | 90 | 0 | yes | 2.8 | 0.41 | - |
| 000021 | 0 | 0.949 | 14 | 2 | yes | 0.9 | 0.92 | - |
| 000021 | 1 | 0.866 | 84 | 0 | yes | 3.5 | 0.38 | - |
| 000021 | 2 | 0.949 | 15 | 2 | yes | 2.0 | 0.85 | - |
| 000021 | 3 | 0.840 | 13 | 2 | yes | 3.1 | 0.81 | - |
| 000021 | 4 | 0.908 | 9 | 2 | yes | 2.6 | 0.89 | - |
| 000021 | 5 | 0.884 | 88 | 0 | yes | 1.7 | 0.74 | - |
| 000021 | 7 | 0.924 | 16 | 2 | yes | 3.2 | 0.84 | - |
| 000022 | 0 | 0.945 | 8 | 2 | yes | 1.8 | 0.90 | - |
| 000022 | 1 | 0.904 | 7 | 2 | yes | 1.4 | 0.93 | - |
| 000022 | 2 | 0.885 | 13 | 2 | yes | 2.6 | 0.89 | - |
| 000022 | 3 | 0.828 | 90 | 0 | yes | 4.9 | 0.42 | gt_masks |
| 000022 | 4 | 0.822 | 85 | 0 | **no** | - | - | ensemble_run2, yolo11l_single |
| 000022 | 5 | 0.978 | 90 | 0 | yes | 1.7 | 0.64 | synthetic_only |
| 000022 | 6 | 0.855 | 81 | 0 | yes | 2.4 | 0.47 | - |
| 000022 | 7 | 0.857 | 87 | 0 | yes | 2.5 | 0.65 | - |
| 000022 | 8 | 0.887 | 13 | 2 | **no** | - | - | ensemble_run2, yolo11l_single, synthetic_only |
| 000023 | 0 | 0.952 | 6 | 0 | yes | 3.8 | 0.89 | - |
| 000023 | 1 | 0.928 | 15 | 2 | yes | 1.0 | 0.82 | - |
| 000023 | 2 | 0.954 | 17 | 2 | yes | 1.1 | 0.91 | - |
| 000023 | 3 | 0.914 | 25 | 2 | yes | 1.2 | 0.83 | - |
| 000023 | 4 | 0.966 | 9 | 2 | yes | 1.3 | 0.89 | - |
| 000026 | 0 | 0.922 | 17 | 2 | yes | 2.2 | 0.92 | - |
| 000026 | 1 | 0.871 | 19 | 2 | yes | 1.5 | 0.78 | - |
| 000026 | 2 | 0.967 | 18 | 2 | yes | 1.6 | 0.87 | - |
| 000026 | 3 | 0.954 | 16 | 2 | yes | 1.1 | 0.92 | - |
| 000026 | 4 | 0.938 | 8 | 2 | yes | 1.2 | 0.89 | - |
| 000030 | 0 | 0.930 | 17 | 2 | yes | 1.8 | 0.88 | - |
| 000030 | 1 | 0.915 | 87 | 0 | yes | 1.0 | 0.73 | - |
| 000030 | 2 | 0.907 | 80 | 0 | yes | 3.3 | 0.71 | - |
| 000030 | 3 | 0.988 | 88 | 0 | **no** | - | - | ensemble_run2, yolo11l_single, synthetic_only |
| 000030 | 4 | 0.868 | 87 | 0 | yes | 2.5 | 0.70 | - |
| 000030 | 5 | 0.874 | 82 | 0 | yes | 4.3 | 0.41 | synthetic_only |
| 000030 | 6 | 0.985 | 88 | 0 | **no** | - | - | ensemble_run2, yolo11l_single, synthetic_only |
| 000030 | 7 | 0.871 | 90 | 0 | yes | 2.8 | 0.41 | synthetic_only |
| 000030 | 8 | 0.994 | 89 | 0 | yes | 9.7 | 0.61 | synthetic_only |
| 000033 | 0 | 0.962 | 10 | 2 | yes | 1.4 | 0.90 | - |
| 000033 | 1 | 0.922 | 73 | 2 | yes | 1.8 | 0.82 | - |
| 000033 | 2 | 0.981 | 18 | 2 | yes | 2.2 | 0.80 | - |
| 000033 | 3 | 0.951 | 9 | 2 | yes | 1.8 | 0.90 | - |
| 000033 | 4 | 0.924 | 11 | 2 | yes | 1.9 | 0.90 | - |
| 000033 | 5 | 0.838 | 87 | 0 | yes | 3.5 | 0.64 | - |
| 000033 | 6 | 0.919 | 29 | 2 | yes | 3.4 | 0.74 | - |
| 000033 | 7 | 0.871 | 6 | 1 | yes | 3.3 | 0.83 | - |
| 000033 | 8 | 0.927 | 7 | 0 | yes | 4.6 | 0.89 | - |
| 000040 | 0 | 0.949 | 7 | 2 | yes | 1.2 | 0.81 | - |
| 000040 | 1 | 0.955 | 12 | 2 | yes | 0.8 | 0.97 | - |
| 000041 | 0 | 0.819 | 8 | 2 | yes | 4.4 | 0.72 | yolo11l_single, synthetic_only |
| 000041 | 1 | 0.949 | 7 | 2 | yes | 1.3 | 0.92 | - |
| 000041 | 3 | 0.924 | 20 | 2 | yes | 2.1 | 0.81 | - |
| 000041 | 4 | 0.944 | 46 | 2 | yes | 1.5 | 0.86 | - |
| 000041 | 5 | 0.963 | 12 | 2 | **no** | - | - | ensemble_run2, yolo11l_single, synthetic_only |
| 000041 | 7 | 0.966 | 12 | 2 | yes | 0.9 | 0.82 | - |
| 000047 | 0 | 1.037 | 7 | 2 | yes | 2.8 | 0.79 | - |
| 000047 | 1 | 0.986 | 4 | 2 | yes | 3.6 | 0.77 | - |
| 000047 | 2 | 0.990 | 9 | 2 | yes | 2.0 | 0.69 | - |
| 000047 | 3 | 0.929 | 7 | 1 | yes | 6.9 | 0.34 | yolo11l_single, synthetic_only |
| 000047 | 4 | 0.998 | 4 | 2 | yes | 3.3 | 0.71 | - |
| 000047 | 5 | 0.954 | 12 | 2 | yes | 2.5 | 0.85 | - |
| 000047 | 6 | 1.015 | 10 | 2 | yes | 1.9 | 0.79 | - |
| 000047 | 7 | 0.909 | 7 | 2 | yes | 1.9 | 0.55 | - |
| 000054 | 0 | 0.902 | 6 | 2 | yes | 2.4 | 0.91 | - |
| 000054 | 1 | 0.936 | 35 | 2 | yes | 3.0 | 0.87 | - |
| 000054 | 2 | 0.843 | 6 | 0 | yes | 4.3 | 0.81 | - |
| 000058 | 0 | 0.942 | 13 | 2 | yes | 1.8 | 0.90 | - |
| 000058 | 1 | 0.904 | 56 | 2 | yes | 1.6 | 0.84 | - |
| 000058 | 2 | 0.868 | 20 | 2 | yes | 1.1 | 0.87 | - |
| 000058 | 3 | 0.808 | 28 | 1 | yes | 3.7 | 0.77 | yolo11l_single |
| 000058 | 4 | 0.835 | 27 | 0 | yes | 3.7 | 0.84 | yolo11l_single |
| 000058 | 6 | 0.868 | 20 | 2 | yes | 1.9 | 0.91 | - |
| 000058 | 7 | 0.911 | 40 | 1 | yes | 2.7 | 0.72 | - |
| 000059 | 0 | 0.928 | 16 | 2 | yes | 1.9 | 0.86 | - |
| 000059 | 1 | 0.843 | 5 | 2 | yes | 2.0 | 0.67 | - |
| 000059 | 2 | 0.920 | 16 | 2 | yes | 1.9 | 0.88 | - |
| 000059 | 3 | 0.931 | 17 | 2 | yes | 2.7 | 0.89 | - |
| 000059 | 4 | 0.863 | 19 | 1 | yes | 1.9 | 0.70 | - |
| 000059 | 5 | 0.901 | 26 | 1 | yes | 1.8 | 0.89 | - |
| 000059 | 6 | 0.923 | 44 | 2 | yes | 1.0 | 0.85 | - |
| 000059 | 8 | 1.007 | 11 | 2 | yes | 0.7 | 0.95 | - |

### Missed at 10 mm (5)

| scene | idx | visib | nearest pred: MSSD (score) | rot to GT | dt mm | pred < 15 mm | fold seg IoU (conf, gate) | synthetic IoU (conf, gate) | crop | diagnosis |
|---|---|---|---|---|---|---|---|---|---|---|
| 000022 | 4 | 0.822 | 2.9 (0.64) | 5 deg about X (bar) | 0.9 | yes | 0.89 (0.97, pass) | 0.35 (0.88, pass) | 000022_inst04_mssd3mm.png | duplicate GT label of #5 (twin matched); one pose cannot claim both |
| 000022 | 8 | 0.887 | 3.1 (0.89) | 3 deg about X (bar) | 1.7 | yes | 0.79 (0.99, pass) | 0.78 (0.96, pass) | 000022_inst08_mssd3mm.png | duplicate GT label of #2 (twin matched); one pose cannot claim both |
| 000030 | 3 | 0.988 | 9.8 (0.61) | 9 deg about Y (plate normal) | 4.3 | yes | 0.82 (0.95, pass) | 0.70 (0.88, pass) | 000030_inst03_mssd10mm.png | duplicate GT label of #6 (twin also missed); one pose cannot claim both |
| 000030 | 6 | 0.985 | 9.8 (0.61) | 9 deg about Y (plate normal) | 4.2 | yes | 0.82 (0.95, pass) | 0.68 (0.88, pass) | 000030_inst06_mssd10mm.png | duplicate GT label of #3 (twin also missed); one pose cannot claim both |
| 000041 | 5 | 0.963 | 1.3 (0.82) | 1 deg about X (bar) | 0.9 | yes | 0.81 (0.96, pass) | 0.84 (0.94, pass) | 000041_inst05_mssd1mm.png | duplicate GT label of #7 (twin matched); one pose cannot claim both |

### Matched at 10 mm but not at 4 mm (8)

| scene | idx | visib | tilt deg | holes | MSSD mm | rot err | dt mm | score |
|---|---|---|---|---|---|---|---|
| 000019 | 3 | 0.854 | 7 | 1 | 4.6 | 3.8 deg about Y (plate normal) | 2.8 | 0.85 |
| 000022 | 3 | 0.828 | 90 | 0 | 4.9 | 6.8 deg about X (bar) | 1.9 | 0.42 |
| 000030 | 5 | 0.874 | 82 | 0 | 4.3 | 8.4 deg about X (bar) | 1.4 | 0.41 |
| 000030 | 8 | 0.994 | 89 | 0 | 9.7 | 8.7 deg about Y (plate normal) | 4.1 | 0.61 |
| 000033 | 8 | 0.927 | 7 | 0 | 4.6 | 5.5 deg about X (bar) | 2.6 | 0.89 |
| 000041 | 0 | 0.819 | 8 | 2 | 4.4 | 3.2 deg about Y (plate normal) | 2.6 | 0.72 |
| 000047 | 3 | 0.929 | 7 | 1 | 6.9 | 10.0 deg about Y (plate normal) | 2.7 | 0.34 |
| 000054 | 2 | 0.843 | 6 | 0 | 4.3 | 3.7 deg about Y (plate normal) | 1.9 | 0.81 |

### Matched MSSD vs holes and tilt

| bin | n | median mm | p75 mm | < 2 mm | < 4 mm |
|---|---|---|---|---|---|
| holes visible = 2 | 81 | 1.76 | 2.22 | 0.65 | 0.99 |
| holes visible = 1 | 9 | 3.21 | 3.73 | 0.33 | 0.78 |
| holes visible = 0 | 22 | 2.82 | 3.73 | 0.23 | 0.77 |
| tilt 0-15 deg | 58 | 1.91 | 2.78 | 0.57 | 0.91 |
| tilt 15-45 deg | 33 | 1.88 | 2.67 | 0.61 | 1.00 |
| tilt > 45 deg | 21 | 2.48 | 3.27 | 0.38 | 0.86 |

### Tally

- misses: 5 duplicate labels, 0 flips, 0 segmenter misses, 0 mislocalised, 0 other
- matched at 10 mm but not at 4 mm: 8 of 117
- predictions: 185; unmatched at 10 mm: 73 = 23 on ignore regions + 16 on optional instances (< 0.8 visible) + 34 false positives, of which 0 < 15 mm from a GT pose (second pose of a found part), 8 flips of a GT pose (> 150 deg), 26 other wrong poses of real parts
<!-- tables:end -->
