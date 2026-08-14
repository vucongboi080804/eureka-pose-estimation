"""Union two submissions, deduplicating with the detector's own NMS rule.

The learned-mask path and the geometric path drop different instances;
their union keeps whatever either found. Rows are ranked by score, and a
pair counts as one detection only when both position and orientation
nearly coincide.

    .venv/bin/python scripts/merge_submissions.py a.json b.json --out merged.json
"""

import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect import NMS_ANGLE_DEG, NMS_DIST


def merge_nms(rows: list) -> list:
    kept = []
    for row in sorted(rows, key=lambda r: -r["score"]):
        R, t = np.array(row["R"]), np.array(row["t"])
        dup = False
        for k in kept:
            if np.linalg.norm(t - np.array(k["t"])) > NMS_DIST:
                continue
            cos = (np.trace(R.T @ np.array(k["R"])) - 1.0) / 2.0
            if np.degrees(np.arccos(np.clip(cos, -1, 1))) < NMS_ANGLE_DEG:
                dup = True
                break
        if not dup:
            kept.append(row)
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("submissions", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    loaded = [json.load(open(path)) for path in args.submissions]
    scenes = sorted(set().union(*loaded))
    merged = {}
    for sid in scenes:
        rows = [r for sub in loaded for r in sub.get(sid, [])]
        merged[sid] = merge_nms(rows)
    with open(args.out, "w") as fh:
        json.dump(merged, fh)
    total = sum(len(v) for v in merged.values())
    print("wrote %s: %d scenes, %d predictions" % (args.out, len(merged), total))


if __name__ == "__main__":
    main()
