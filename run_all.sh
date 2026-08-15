#!/usr/bin/env bash
# One command for evaluation day: point at any release folder (holding
# model/ and a split of scenes) and get a submission plus overlays.
#
#   ./run_all.sh <release_path> [split] [out_dir]
#
# Examples:
#   ./run_all.sh .                          # this release, test split
#   ./run_all.sh /path/to/private_release   # the interview's dataset
#   ./run_all.sh /path/to/release val out_val
#
# Uses the shipped real-trained segmenter with automatic fallbacks (the
# synthetic-only model and the geometric detector cover domain shift).
# Requires the .venv from requirements.txt.
set -euo pipefail
cd "$(dirname "$0")"

RELEASE="${1:?usage: ./run_all.sh <release_path> [split] [out_dir]}"
SPLIT="${2:-test}"
OUT="${3:-out_$(basename "$RELEASE")_$SPLIT}"
PY=.venv/bin/python

mkdir -p "$OUT"
echo "== Detecting and estimating poses ($RELEASE / $SPLIT) =="
$PY scripts/run_pipeline.py --root "$RELEASE" --split "$SPLIT" \
    --out "$OUT/submission.json" --labels-out "$OUT/pred_labels" \
    --workers 6 --seg-model weights/part-seg.pt

echo "== Rendering overlays =="
MPLBACKEND=Agg $PY visualize.py --root "$RELEASE" --split "$SPLIT" \
    --labels "$OUT/pred_labels" --save "$OUT/overlays/"

if [ -f "$RELEASE/$SPLIT/$(ls "$RELEASE/$SPLIT" | head -1)/poses.json" ]; then
    echo "== Ground truth present: scoring =="
    $PY score.py --release "$RELEASE" --split "$SPLIT" \
        --submission "$OUT/submission.json"
fi

echo "Done: $OUT/submission.json + $OUT/overlays/"
