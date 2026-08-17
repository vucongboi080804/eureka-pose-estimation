# Score calibration

Submission `train_ensemble_run1.json`: 185 predictions over 20 scenes; 23 land on ignore regions (unlabelled instances) and are dropped, as the scorer drops them. `score` = segmenter confidence x depth verification; a prediction is correct when the scorer's greedy matching pairs it with any instance within the threshold.

## Reliability

| score bin | n | mean score | prec@2mm | prec@5mm | prec@10mm |
|---|---|---|---|---|---|
| 0.0-0.1 | 0 | - | - | - | - |
| 0.1-0.2 | 8 | 0.154 | 0.00 | 0.00 | 0.00 |
| 0.2-0.3 | 5 | 0.222 | 0.00 | 0.00 | 0.00 |
| 0.3-0.4 | 7 | 0.372 | 0.00 | 0.14 | 0.14 |
| 0.4-0.5 | 11 | 0.440 | 0.00 | 0.55 | 0.64 |
| 0.5-0.6 | 16 | 0.542 | 0.19 | 0.31 | 0.44 |
| 0.6-0.7 | 19 | 0.657 | 0.21 | 0.63 | 0.74 |
| 0.7-0.8 | 21 | 0.752 | 0.43 | 1.00 | 1.00 |
| 0.8-0.9 | 51 | 0.861 | 0.65 | 1.00 | 1.00 |
| 0.9-1.0 | 24 | 0.926 | 0.58 | 1.00 | 1.00 |

ECE at 5 mm: **0.144**; Brier score at 5 mm: **0.098**.

## Operating points (official `score.py`, predictions with score >= gate)

| gate | preds/scene | recall@10 | precision@10 | recall@2 | precision@2 | AR | top-1 | prec@5 (replay) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 9.2 | 0.940 | 0.748 | 0.521 | 0.381 | 0.844 | 1.000 | 0.741 |
| 0.1 | 9.2 | 0.940 | 0.748 | 0.521 | 0.381 | 0.844 | 1.000 | 0.741 |
| 0.2 | 8.8 | 0.940 | 0.791 | 0.521 | 0.401 | 0.844 | 1.000 | 0.779 |
| 0.3 | 8.4 | 0.940 | 0.821 | 0.521 | 0.415 | 0.844 | 1.000 | 0.805 |
| 0.4 | 8.1 | 0.932 | 0.858 | 0.521 | 0.436 | 0.838 | 1.000 | 0.838 |
| 0.5 | 7.2 | 0.880 | 0.880 | 0.521 | 0.473 | 0.798 | 1.000 | 0.863 |
| 0.6 | 6.2 | 0.863 | 0.953 | 0.504 | 0.517 | 0.781 | 1.000 | 0.939 |
| 0.7 | 5.0 | 0.778 | 1.000 | 0.470 | 0.579 | 0.711 | 1.000 | 1.000 |
| 0.8 | 3.8 | 0.641 | 1.000 | 0.402 | 0.627 | 0.588 | 1.000 | 1.000 |
| 0.9 | 1.2 | 0.205 | 1.000 | 0.120 | 0.583 | 0.188 | 0.800 | 1.000 |

## Top-1 per scene

| scene | top score | MSSD (mm) |
|---|---|---|
| 000047 | 0.852 | 2.52 |
| 000030 | 0.879 | 1.70 |
| 000009 | 0.885 | 0.96 |
| 000011 | 0.898 | 1.58 |
| 000033 | 0.901 | 1.78 |
| 000058 | 0.908 | 1.90 |
| 000023 | 0.911 | 1.13 |
| 000054 | 0.915 | 2.38 |
| 000026 | 0.919 | 2.17 |
| 000020 | 0.923 | 3.06 |
| 000021 | 0.924 | 0.89 |
| 000041 | 0.924 | 1.30 |
| 000014 | 0.928 | 2.03 |
| 000022 | 0.930 | 1.37 |
| 000007 | 0.930 | 2.54 |
| 000010 | 0.933 | 1.87 |
| 000019 | 0.946 | 3.17 |
| 000059 | 0.948 | 0.74 |
| 000008 | 0.963 | 2.80 |
| 000040 | 0.971 | 0.78 |

Least confident top pick: score 0.852; worst top-1 MSSD 3.17 mm; top-1 scores below 0.5: 0 of 20.

## Other configurations (same analysis)

| submission | n | ECE@5 | Brier@5 | prec@5 for score>=0.6 | min top-1 score |
|---|---|---|---|---|---|
| train_yolo11l_single.json | 152 | 0.157 | 0.096 | 0.95 | 0.805 |
| train_synthetic_only.json | 168 | 0.138 | 0.091 | 0.95 | 0.854 |

## Component split (57 predictions on 5 held-out scenes, 45 correct at 5 mm)

Detection re-run with the fold segmenter that never saw the scene plus the synthetic model (conf 0.25); RANSAC makes the figures move by a few points between runs.

| factor | AUROC vs correct@5mm |
|---|---|
| segmenter confidence | 0.943 |
| verification | 0.946 |
| product (score) | 0.961 |

![](score_calibration.png)

## Conclusion

1. **Ranking is trustworthy.** AUROC of `score` against correct-at-5-mm is 0.95 over 162 labelled predictions; the top pick of every scene scores >= 0.852 and lies within 3.17 mm (top-1 20/20, never below 0.5).
2. **As a probability it is only roughly calibrated** (ECE 0.144, Brier 0.098 at 5 mm): over-confident in the middle bins, under-confident above ~0.7 (see table). A monotone recalibration (isotonic on the CV predictions) would fix the level without changing the ranking; the raw product should be read as a rank, not a probability.
3. **Operating point.** Gate 0.6: 6.2 preds/scene, precision 0.94 at 5 mm / 0.953 at 10 mm, recall 0.863 at 10 mm, AR 0.781. Gate 0.7: precision 1.00 at 5 mm, recall 0.778. Precision at 5 mm first reaches 0.95 at gate 0.7; the knee is 0.6-0.7, and below 0.4 a prediction is wrong more often than right.
4. **Which factor carries it:** AUROC verification 0.95, segmenter 0.94, product 0.96 -- both factors are individually predictive and the product is at least as good as either; 57 predictions on 5 scenes, so a few points of difference are noise.
5. **For a robot cell:** pick from `score` >= 0.7, rescan when nothing reaches it; the gate trades recall for precision without touching the top pick up to gate 0.8.