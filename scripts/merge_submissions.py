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


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nms import is_duplicate


def merge_nms(rows: list) -> list:
    """NMS over submission rows, by the detector's own duplicate rule."""
    kept = []
    for row in sorted(rows, key=lambda r: -r["score"]):
        if not any(is_duplicate(row["R"], row["t"], k["R"], k["t"]) for k in kept):
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
