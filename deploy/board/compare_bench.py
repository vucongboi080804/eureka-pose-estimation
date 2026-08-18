"""Diff two bench.py records: does the emulation give the same answer, and
how much slower is it?

Two questions travel together and must not be confused. *Same answer* is a
correctness claim about the port -- an aarch64 build of Open3D, a CPU torch
wheel and a different BLAS must land the same pose the desktop lands, within
2 mm and 2 deg -- and it is the one this script gates on, exiting non-zero
when it fails. *Same speed* is a claim about the host, and qemu-user emulates
the instruction set, not the microarchitecture: it has no Cortex-A57 caches
and no LPDDR4 bandwidth, so its ratio bounds nothing and is reported without
a verdict.

    compare_bench.py out_bench/emulated.json out_bench/board.json [--md]

Scene sets are intersected and every field is optional, so a record written
by an older bench.py still compares on whatever it does carry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

#: A pose that moves less than this between two machines is the same pose:
#: the depth channel is quantised to 1 mm and the pipeline's own in-plane
#: floor is ~2 mm, so a tighter tolerance would be measuring noise.
DEFAULT_MAX_MM = 2.0
DEFAULT_MAX_DEG = 2.0

#: Poses further apart than this are different instances rather than the
#: same instance estimated differently, so pairing them would be meaningless.
PAIR_GATE_MM = 25.0


def _load(path: str) -> dict:
    with open(path) as fh:
        record = json.load(fh)
    if "scenes" not in record:
        raise SystemExit("%s is not a bench.py record (no 'scenes')" % path)
    return record


def _label(record: dict, fallback: str) -> str:
    """Short name for a column, taken from what the record says it ran on."""
    host = record.get("host") or {}
    kind = host.get("platform_kind")
    if not kind:
        return fallback
    if kind == "jetson":
        return "board"
    if kind == "qemu-user":
        return "qemu"
    return host.get("machine") or fallback


#: Config keys that change what the pipeline is asked to do. A pick-mode run
#: returns one pose and a full sweep returns nine, so comparing the two would
#: report a disagreement that is only a difference in the question asked.
COMPARED_CONFIG = ("split", "pick", "seg_model", "extra_seg_model",
                   "profile", "imgsz")


def config_mismatch(left: dict, right: dict) -> list:
    """Config keys the two records disagree on, as human-readable strings."""
    left_config = left.get("config") or {}
    right_config = right.get("config") or {}
    return ["%s: %r vs %r" % (key, left_config.get(key), right_config.get(key))
            for key in COMPARED_CONFIG
            if key in left_config and key in right_config
            and left_config[key] != right_config[key]]


def _first_repeat(scene: dict):
    """The first successful pass, or None. Poses are compared from one pass:
    averaging rotations across stochastic repeats would invent a pose."""
    for repeat in scene.get("repeats") or []:
        if "poses" in repeat:
            return repeat
    return None


def _wall_min(scene: dict):
    if "wall_s_min" in scene:
        return scene["wall_s_min"]
    times = [r["wall_s"] for r in scene.get("repeats") or [] if "wall_s" in r]
    return min(times) if times else None


def _pose_arrays(pose: dict):
    return (np.asarray(pose["R"], dtype=float).reshape(3, 3),
            np.asarray(pose["t"], dtype=float).reshape(3))


def _rotation_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first.T @ second)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def pair_poses(left: list, right: list) -> tuple:
    """Greedily pair two pose lists by proximity, best-scoring pose first.

    The comparison is deliberately not symmetry-aware. The part is nearly
    180 deg symmetric, so a symmetry-tolerant metric would call a run that
    picked the flip identical to one that picked the truth -- and choosing
    between those two is exactly what the verification stack does, so a
    machine that chooses differently is a finding, not a rounding error.
    """
    order = sorted(range(len(left)), key=lambda i: -left[i].get("score", 0.0))
    available = list(range(len(right)))
    pairs, unmatched_left = [], []
    for i in order:
        rot_l, t_l = _pose_arrays(left[i])
        best, best_mm = None, None
        for j in available:
            _, t_r = _pose_arrays(right[j])
            millimetres = float(np.linalg.norm(t_l - t_r)) * 1000.0
            if best_mm is None or millimetres < best_mm:
                best, best_mm = j, millimetres
        if best is None or best_mm > PAIR_GATE_MM:
            unmatched_left.append(i)
            continue
        available.remove(best)
        rot_r, _ = _pose_arrays(right[best])
        pairs.append({"mm": best_mm, "deg": _rotation_deg(rot_l, rot_r),
                      "score_delta": abs(left[i].get("score", 0.0)
                                         - right[best].get("score", 0.0))})
    return pairs, unmatched_left, available


def _stage_totals(record: dict, scene_ids: list) -> dict:
    """Best-case seconds per stage, summed over the compared scenes."""
    totals = {}
    for scene_id in scene_ids:
        stages = record["scenes"][scene_id].get("stages_s_min") or {}
        for name, seconds in stages.items():
            totals[name] = totals.get(name, 0.0) + seconds
    return totals


def _spread(scene: dict) -> tuple:
    """Worst disagreement between repeats of one scene on one machine.

    This is the noise floor the cross-machine tolerance has to clear: if a
    machine does not reproduce itself to 2 mm, 2 mm cannot be asked of two
    machines.
    """
    repeats = [r for r in scene.get("repeats") or [] if "poses" in r]
    worst_mm = worst_deg = 0.0
    for later in repeats[1:]:
        pairs, extra, missing = pair_poses(repeats[0]["poses"], later["poses"])
        if extra or missing:
            return None, None
        for pair in pairs:
            worst_mm = max(worst_mm, pair["mm"])
            worst_deg = max(worst_deg, pair["deg"])
    return (worst_mm, worst_deg) if len(repeats) > 1 else (None, None)


def compare(left: dict, right: dict, max_mm: float, max_deg: float) -> dict:
    """Per-scene agreement and speed over the scenes both records contain."""
    shared = sorted(set(left["scenes"]) & set(right["scenes"]))
    rows = []
    for scene_id in shared:
        left_scene = left["scenes"][scene_id]
        right_scene = right["scenes"][scene_id]
        left_run = _first_repeat(left_scene)
        right_run = _first_repeat(right_scene)
        row = {
            "scene": scene_id,
            "left_s": _wall_min(left_scene),
            "right_s": _wall_min(right_scene),
            "left_poses": left_run["n_poses"] if left_run else None,
            "right_poses": right_run["n_poses"] if right_run else None,
            "max_mm": None, "max_deg": None, "score_delta": None,
            "unmatched": None, "agrees": False,
            "why": "",
        }
        if row["left_s"] and row["right_s"]:
            row["ratio"] = row["left_s"] / row["right_s"]
        else:
            row["ratio"] = None
        if left_run is None or right_run is None:
            row["why"] = "no successful repeat on %s" % (
                "both" if left_run is None and right_run is None
                else ("left" if left_run is None else "right"))
            rows.append(row)
            continue
        pairs, extra, missing = pair_poses(left_run["poses"],
                                           right_run["poses"])
        row["unmatched"] = len(extra) + len(missing)
        row["max_mm"] = max((p["mm"] for p in pairs), default=0.0)
        row["max_deg"] = max((p["deg"] for p in pairs), default=0.0)
        row["score_delta"] = max((p["score_delta"] for p in pairs),
                                 default=0.0)
        row["agrees"] = (row["unmatched"] == 0
                         and row["max_mm"] <= max_mm
                         and row["max_deg"] <= max_deg)
        if not row["agrees"]:
            reasons = []
            if row["unmatched"]:
                reasons.append("%d pose(s) present on one side only"
                               % row["unmatched"])
            if row["max_mm"] > max_mm:
                reasons.append("%.2f mm > %.1f mm" % (row["max_mm"], max_mm))
            if row["max_deg"] > max_deg:
                reasons.append("%.2f deg > %.1f deg"
                               % (row["max_deg"], max_deg))
            row["why"] = "; ".join(reasons)
        rows.append(row)
    return {
        "shared": shared,
        "left_only": sorted(set(left["scenes"]) - set(right["scenes"])),
        "right_only": sorted(set(right["scenes"]) - set(left["scenes"])),
        "rows": rows,
    }


def _cell(value, fmt: str = "%.1f") -> str:
    return "-" if value is None else fmt % value


def render(result: dict, left: dict, right: dict, left_name: str,
           right_name: str, max_mm: float, max_deg: float,
           markdown: bool) -> str:
    rows = result["rows"]
    header = ["scene", "%s s" % left_name, "%s s" % right_name, "ratio",
              "poses", "max mm", "max deg", "d score", "output"]
    body = []
    for row in rows:
        body.append([
            row["scene"], _cell(row["left_s"]), _cell(row["right_s"]),
            _cell(row["ratio"], "%.1fx"),
            "%s/%s" % (_cell(row["left_poses"], "%d"),
                       _cell(row["right_poses"], "%d")),
            _cell(row["max_mm"], "%.2f"), _cell(row["max_deg"], "%.2f"),
            _cell(row["score_delta"], "%.3f"),
            "ok" if row["agrees"] else "DIFFERS",
        ])
    ratios = [r["ratio"] for r in rows if r["ratio"]]
    if ratios:
        ratios.sort()
        median = ratios[len(ratios) // 2]
        body.append(["overall",
                     _cell(sum(r["left_s"] for r in rows if r["left_s"])),
                     _cell(sum(r["right_s"] for r in rows if r["right_s"])),
                     "%.1fx" % median, "", "", "", "",
                     "%d/%d ok" % (sum(1 for r in rows if r["agrees"]),
                                   len(rows))])

    lines = []
    for name, record in ((left_name, left), (right_name, right)):
        host = record.get("host") or {}
        config = record.get("config") or {}
        cgroup = host.get("cgroup") or {}
        lines.append("%-6s %s | %s | %s cores, %s MB | %s threads, %s | "
                     "commit %s | %s"
                     % (name, host.get("platform_kind", "?"),
                        host.get("cpu_model", "?"),
                        cgroup.get("cpu_quota") or host.get("cpu_affinity")
                        or host.get("cpu_count"),
                        cgroup.get("memory_limit_mb")
                        or host.get("total_ram_mb"),
                        config.get("threads", "?"),
                        "cuda" if host.get("torch_cuda") else "cpu",
                        record.get("git_commit") or "unknown",
                        record.get("timestamp_utc", "?")))
        if record.get("note"):
            lines.append("       note: %s" % record["note"])
    mismatch = config_mismatch(left, right)
    if mismatch:
        lines.append("")
        lines.append("WARNING: the two runs were not asked the same question "
                     "(%s) -- the pose comparison below is meaningless until "
                     "they match." % "; ".join(mismatch))
    lines.append("")
    lines.append(_table(header, body, markdown))
    lines.append("")

    for key, name in (("left_only", left_name), ("right_only", right_name)):
        if result[key]:
            lines.append("only in %s, not compared: %s"
                         % (name, " ".join(result[key])))
    if not rows:
        lines.append("VERDICT: no shared scenes -- nothing to compare.")
        return "\n".join(lines)

    disagree = [r for r in rows if not r["agrees"]]
    worst_mm = max((r["max_mm"] for r in rows if r["max_mm"] is not None),
                   default=0.0)
    worst_deg = max((r["max_deg"] for r in rows if r["max_deg"] is not None),
                    default=0.0)
    plural = "" if len(rows) == 1 else "s"
    if disagree:
        lines.append("OUTPUT : DISAGREES on %d of %d scene%s (tolerance "
                     "%.1f mm / %.1f deg)"
                     % (len(disagree), len(rows), plural, max_mm, max_deg))
        for row in disagree:
            lines.append("         %s: %s" % (row["scene"], row["why"]))
    else:
        lines.append("OUTPUT : AGREES on all %d scene%s -- worst %.2f mm / "
                     "%.2f deg against a tolerance of %.1f mm / %.1f deg"
                     % (len(rows), plural, worst_mm, worst_deg,
                        max_mm, max_deg))

    floors = []
    for record, name in ((left, left_name), (right, right_name)):
        for scene_id in result["shared"]:
            millimetres, degrees = _spread(record["scenes"][scene_id])
            if millimetres is not None:
                floors.append((name, scene_id, millimetres, degrees))
    if floors:
        worst = max(floors, key=lambda f: f[2])
        lines.append("NOISE  : same machine, repeated -- worst %.2f mm / "
                     "%.2f deg (%s, %s). RANSAC is stochastic; a "
                     "cross-machine delta below this is not a port defect."
                     % (worst[2], worst[3], worst[0], worst[1]))
    else:
        lines.append("NOISE  : one repeat per scene, so the same-machine "
                     "spread is unmeasured. Re-run with --repeat 2 to know "
                     "how much of the delta above is just RANSAC.")

    left_stages = _stage_totals(left, result["shared"])
    right_stages = _stage_totals(right, result["shared"])
    per_stage = ["%s %.0fx" % (name, left_stages[name] / right_stages[name])
                 for name in ("io", "setup", "segmenter", "register")
                 if right_stages.get(name)]
    if per_stage:
        # The reason this line exists: the first emulated run of this
        # harness was 512x on the segmenters and 15x on registration. One
        # overall ratio hides that, and scaling a board estimate by it would
        # be wrong by more than an order of magnitude on both halves.
        lines.append("STAGES : %s. A slowdown that differs this much per "
                     "stage does not transfer as one number -- scale each "
                     "stage, or measure." % ", ".join(per_stage))

    if ratios:
        speed = ("SPEED  : %s takes %.1fx %s (median over %d scene%s)."
                 % (left_name, median, right_name, len(ratios),
                    "" if len(ratios) == 1 else "s"))
        kinds = {(record.get("host") or {}).get("platform_kind")
                 for record in (left, right)}
        if "qemu-user" in kinds:
            # Said only when it applies: between two real machines the ratio
            # is a measurement, and blanketing it with this caveat would
            # teach the reader to ignore the line that matters on the board.
            speed += (" Not a verdict: qemu-user emulates the instruction "
                      "set, not the microarchitecture -- no A57 caches, no "
                      "LPDDR4 bandwidth -- so treat this as an upper bound "
                      "and re-measure on the board.")
        lines.append(speed)
    return "\n".join(lines)


def _table(header: list, body: list, markdown: bool) -> str:
    widths = [max(len(str(row[i])) for row in [header] + body)
              for i in range(len(header))]
    def line(cells, pad="  "):
        return pad.join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
        return "\n".join(out)
    return "\n".join([line(header),
                      "  ".join("-" * w for w in widths)]
                     + [line(row) for row in body])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("emulated", help="bench.json from the slower or "
                                         "emulated host")
    parser.add_argument("board", help="bench.json from the reference host, "
                                      "normally the Jetson")
    parser.add_argument("--md", action="store_true",
                        help="Markdown table, for pasting into a report")
    parser.add_argument("--max-mm", type=float, default=DEFAULT_MAX_MM)
    parser.add_argument("--max-deg", type=float, default=DEFAULT_MAX_DEG)
    args = parser.parse_args(argv)

    left, right = _load(args.emulated), _load(args.board)
    left_name = _label(left, "left")
    right_name = _label(right, "right")
    if left_name == right_name:
        left_name, right_name = left_name + "-A", right_name + "-B"
    result = compare(left, right, args.max_mm, args.max_deg)
    print(render(result, left, right, left_name, right_name,
                 args.max_mm, args.max_deg, args.md))
    if not result["rows"] or config_mismatch(left, right):
        return 2
    return 1 if any(not row["agrees"] for row in result["rows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
