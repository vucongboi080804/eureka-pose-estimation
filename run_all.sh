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
# Pools proposals from the two shipped segmenters (real-trained and
# synthetic-only); the training-free geometric detector joins in
# automatically when too few detections verify (domain shift).
# Requires the .venv from requirements.txt. GPU optional (CPU is ~2x slower).
set -euo pipefail
cd "$(dirname "$0")"

RELEASE="${1:?usage: ./run_all.sh <release_path> [split] [out_dir]}"
SPLIT="${2:-test}"
TAG="$(basename "$(realpath "$RELEASE")")"
OUT="${3:-out_${TAG}_${SPLIT}}"
WORKERS="${WORKERS:-6}"     # override: WORKERS=2 ./run_all.sh ...
PY=.venv/bin/python

mkdir -p "$OUT"
echo "== Detecting and estimating poses ($RELEASE / $SPLIT) =="
$PY scripts/run_pipeline.py --root "$RELEASE" --split "$SPLIT" \
    --out "$OUT/submission.json" --labels-out "$OUT/pred_labels" \
    --workers "$WORKERS" --seg-model weights/part-seg.pt --extra-seg-model weights/part-seg-synthetic.pt

echo "== Rendering overlays =="
MPLBACKEND=Agg $PY visualize.py --root "$RELEASE" --split "$SPLIT" \
    --labels "$OUT/pred_labels" --save "$OUT/overlays/"

if [ -f "$RELEASE/$SPLIT/$(ls "$RELEASE/$SPLIT" | head -1)/poses.json" ]; then
    echo "== Ground truth present: scoring =="
    $PY score.py --release "$RELEASE" --split "$SPLIT" \
        --submission "$OUT/submission.json"
fi

echo "Done: $OUT/submission.json + $OUT/overlays/"
