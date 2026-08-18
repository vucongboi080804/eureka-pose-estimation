"""Draw the figures embedded in README.md and report.md.

Every accuracy number is re-scored here from the result files with the
released ``score.py`` (same thresholds, same visibility rule), so a figure
can never disagree with the scorer. Runtime and memory come from
``results/nano_runtime.json`` and ``results/bench/*_nano640.json``; the
desktop stage profile lives only in ``analysis/runtime.md`` and is copied
below as a constant. Scoring and image helpers live in
``scripts/figure_helpers.py``. Writes seven PNGs::

    .venv/bin/python scripts/make_figures.py --root . --out docs/figures

    hero_overlays.png        eight test-scene overlays (README hero)
    recall_vs_threshold.png  recall vs MSSD threshold, four configurations
    failure_breakdown.png    matched required instances vs the label ceiling
    ablation_bars.png        one stage off: change in AR and precision
    runtime_breakdown.png    stage share of desktop scene time
    edge_tradeoff.png        segmenter configurations: AR vs pick latency
    board_vs_desktop.png     pick-mode stages on x86 / Jetson Nano / qemu
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

from figure_helpers import (THRESHOLDS_MM, Scorer, column, contact_sheet,  # noqa: E402
                            rgb_panel, save_fig, save_png8)

#: Okabe-Ito palette: distinguishable under the common colour-vision deficiencies.
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#8a8a8a", "light": "#d9d9d9"}
#: 117 required instances; 5 are duplicate labels of another instance, so a
#: one-pose-per-part submission can match at most 112 (analysis/failure_analysis.md).
REQUIRED, LABEL_CEILING = 117, 112
#: Run-to-run AR spread of repeated RANSAC draws (analysis/ablation.md,
#: analysis/edge_model.md): two segmenters +-0.005, one segmenter +-0.015.
NOISE_AR_DUAL, NOISE_AR_SINGLE = 0.005, 0.015

#: Stage share of desktop scene time, single worker with GPU, scenes 000001
#: 000003 000015 000043 000052, 59.7 s of scene time; wall-clock wrappers, so
#: only analysis/runtime.md holds these. "other" is the remainder to 100.
RUNTIME_SHARE = [("ICP refinement", 40.1), ("FPFH RANSAC", 33.4), ("rotation grid", 10.9),
                 ("judge / verify", 8.6), ("polish", 2.7), ("segmenters", 2.1)]

#: Table rows of recall_vs_threshold: label, results file, colour, line width.
TABLE_ROWS = [("ensemble (shipped)", "train_ensemble_run1", C["blue"], 2.6),
              ("GT masks", "train_gt_masks", C["green"], 1.6),
              ("single YOLO11l", "train_yolo11l_single", C["orange"], 1.6),
              ("geometric, no training", "train_geometric", C["red"], 1.6)]
#: Ablation rows: label, stage-off file, its baseline file. The own-mask row
#: predates the hole cue, so it compares against the no-hole-cue draw.
ABLATION_ROWS = [("no RGB hole cue", "ablation_no_hole_cue", "train_ensemble_run1"),
                 ("no own-mask check †", "ablation_no_own_mask", "ablation_no_hole_cue"),
                 ("no flip rivals", "ablation_no_flips_v2", "train_ensemble_run1"),
                 ("no rotation grid", "ablation_no_grid_v2", "train_ensemble_run1"),
                 ("no polish", "ablation_no_polish_v2", "train_ensemble_run1"),
                 ("no colour gate", "ablation_no_gate_v2", "train_ensemble_run1")]
#: Segmenter configurations of edge_tradeoff: label, results/nano_runtime.json
#: pick row, CV result files (draws averaged), weight files, single/dual.
EDGE_ROWS = [("L @960 + synthetic\n(shipped)", "dual_960_pick",
              ("train_ensemble_run1", "train_ensemble_run2"), ("part-seg", "part-seg-synthetic")),
             ("L @640 + synthetic", "dual_640_pick", ("nano_dual_640",), ("part-seg", "part-seg-synthetic")),
             ("L @960", "single_960_pick", ("nano_single_960",), ("part-seg",)),
             ("L @768", "single_768_pick", ("nano_single_768",), ("part-seg",)),
             ("L @640", "single_640_pick", ("nano_single_640", "nano_single_640_run2"), ("part-seg",)),
             ("nano @640 (board)", None, ("nano_yolo11n_640", "nano_yolo11n_640_run2",
                                         "nano_yolo11n_640_run3"), ("part-seg-nano",))]
#: The nano configuration is not in results/nano_runtime.json; its pick latency
#: (mean over the same five scenes, same 4-core CPU pinning) is the
#: "Full per-scene pipeline latency" table of analysis/edge_model.md.
NANO_PICK_MEAN_S = 0.73
#: Overlays of the README hero: light, dark and white trays, sparse and dense piles.
HERO_SCENES = ("000001", "000024", "000039", "000031", "000043", "000015", "000052", "000053")
HERO_TILE_W, HERO_GUTTER = 440, 6

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def hero_overlays(root, out):
    """2x4 contact sheet of the RGB panel of eight overlays_test images."""
    tiles = [rgb_panel(os.path.join(root, "overlays_test", sid + ".png"), HERO_TILE_W, sid)
             for sid in HERO_SCENES]
    save_png8(contact_sheet(tiles, 4, HERO_GUTTER), os.path.join(out, "hero_overlays.png"))


def recall_vs_threshold(scorer, out):
    """Recall vs MSSD threshold for the four table configurations."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, name, colour, lw in TABLE_ROWS:
        agg = scorer(name)
        ax.plot(THRESHOLDS_MM, column(agg, "recall"), "o-", color=colour, lw=lw, ms=5,
                label="%s, AR %.3f" % (label, agg["ar"]))
    ens = scorer("train_ensemble_run1")
    ax.plot(THRESHOLDS_MM, column(ens, "precision"), "o--", color=C["blue"], lw=1.4, ms=4,
            label="ensemble precision")
    ax.set(xlabel="MSSD threshold [mm]", ylabel="recall (precision dashed)", xticks=THRESHOLDS_MM,
           ylim=(0, 1.02), title="Recall on the 20 train scenes, leave-scenes-out CV")
    ax.legend(loc="lower right", fontsize=9.5)
    save_fig(fig, out, "recall_vs_threshold.png")


def failure_breakdown(scorer, out):
    """Required instances matched by the ensemble at each threshold vs the ceiling."""
    agg = scorer("train_ensemble_run1")
    tp = np.array(column(agg, "tp"))
    assert agg["required"] == REQUIRED, agg["required"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(THRESHOLDS_MM))
    ax.bar(x, tp, 0.62, color=C["blue"], label="matched")
    ax.bar(x, np.clip(LABEL_CEILING - tp, 0, None), 0.62, bottom=tp, color=C["light"],
           label="missed, reachable")
    ax.bar(x, [REQUIRED - LABEL_CEILING] * len(x), 0.62, bottom=LABEL_CEILING, color="white",
           edgecolor=C["grey"], hatch="////", lw=0.8, label="duplicate labels (5): unreachable")
    for xi, v in zip(x, tp):
        ax.text(xi, v - 3, str(v), ha="center", va="top", color="white", fontsize=10.5)
    ax.axhline(LABEL_CEILING, color=C["grey"], lw=1, ls="--", xmax=0.8)
    ax.text(x[-1] + 0.42, LABEL_CEILING, "112 ceiling", va="center", fontsize=9, color="#444")
    ax.text(x[-1] + 0.42, REQUIRED, "117 required", va="center", fontsize=9, color="#444")
    ax.set(xticks=x, xticklabels=["%g mm" % t for t in THRESHOLDS_MM], ylim=(0, 124),
           xlim=(-0.5, x[-1] + 1.55), ylabel="required instances",
           title="Ensemble: matched instances per MSSD threshold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    save_fig(fig, out, "failure_breakdown.png")


def ablation_bars(scorer, out):
    """One stage off: change in AR and in precision at 10 mm against its baseline."""
    labels, d_ar, d_prec = [], [], []
    for label, name, base in ABLATION_ROWS:
        a, b = scorer(name), scorer(base)
        labels.append(label)
        # Differences of the 3-dp values, so the bars agree with the tables in
        # analysis/ablation.md and report.md (exact differences move by 0.001).
        d_ar.append(round(a["ar"], 3) - round(b["ar"], 3))
        d_prec.append(round(column(a, "precision")[-1], 3) - round(column(b, "precision")[-1], 3))
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    panels = ((axes[0], d_ar, C["blue"], "Δ AR", 0.078, 0.02),
              (axes[1], d_prec, C["orange"], "Δ precision at 10 mm", 0.245, 0.1))
    for ax, vals, colour, title, lim, step in panels:
        if ax is axes[0]:
            ax.axvspan(-NOISE_AR_DUAL, NOISE_AR_DUAL, color=C["light"], alpha=0.7, lw=0, label="noise ±0.005")
            ax.legend(loc="lower left", fontsize=9)
        ax.barh(y, vals, 0.6, color=colour)
        ax.axvline(0, color="#444", lw=0.8)
        for yi, v in zip(y, vals):
            ax.text(v + np.sign(v) * lim * 0.03, yi, "%+.3f" % v, va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        ticks = np.round(np.arange(-step * (lim // step), lim, step), 3)
        ax.set(title=title, xlim=(-lim, lim), xticks=ticks, xticklabels=["%g" % t for t in ticks])
        ax.grid(axis="y", visible=False)
    axes[0].set(yticks=y, yticklabels=labels)
    fig.suptitle("One stage switched off: change against the full pipeline", fontsize=11, y=1.0)
    fig.text(0.13, -0.03, "† this row predates the hole cue; measured against the no-hole-cue draw (AR 0.844)",
             fontsize=8.5, color="#444")
    save_fig(fig, out, "ablation_bars.png")


def runtime_breakdown(out):
    """Stage share of desktop scene time (analysis/runtime.md)."""
    rows = RUNTIME_SHARE + [("other", round(100 - sum(s for _, s in RUNTIME_SHARE), 1))]
    colours = {"segmenters": C["blue"], "other": C["grey"]}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [s for _, s in rows], 0.62, color=[colours.get(n, C["green"]) for n, _ in rows])
    for yi, (_, s) in zip(y, rows):
        ax.text(s + 0.8, yi, "%.1f %%" % s, va="center", fontsize=10)
    ax.set(yticks=y, yticklabels=[n for n, _ in rows], xlim=(0, 50), xlabel="share of scene time [%]",
           title="Where a scene's 12 s go (desktop, one worker)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=C["green"], label="registration, CPU (96 %)"),
                       Patch(color=C["blue"], label="neural segmenters, GPU")], loc="lower right", fontsize=9.5)
    save_fig(fig, out, "runtime_breakdown.png")


def edge_tradeoff(root, scorer, out):
    """Segmenter configurations: AR vs pick-mode latency, marker area = weight size."""
    with open(os.path.join(root, "results", "nano_runtime.json")) as fh:
        latency = {r["name"]: r for r in json.load(fh)["rows"]}
    mb = lambda w: os.path.getsize(os.path.join(root, "weights", w + ".pt")) / 1e6  # noqa: E731
    area = lambda size_mb: 60 + 4.2 * size_mb  # noqa: E731  marker area in pt^2
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    # Label anchor per configuration (data units), its alignment and whether a
    # thin leader line points at the marker (used where the marker is hemmed in
    # by neighbouring error bars).
    places = {"L @960": ((1.77, 0.829), "left"), "L @768": ((0.985, 0.8185), "left", True),
              "L @640": ((1.21, 0.8376), "left"), "L @640 + synthetic": ((2.06, 0.8395), "left"),
              "L @960 + synthetic\n(shipped)": ((2.07, 0.8625), "center"),
              "nano @640 (board)": ((0.695, 0.8258), "right", True)}
    for label, row, draws, weights in EDGE_ROWS:
        x = latency[row]["mean_s"] if row else NANO_PICK_MEAN_S
        ar, size = scorer.ar(*draws), sum(mb(w) for w in weights)
        single = len(weights) == 1
        ax.errorbar(x, ar, yerr=NOISE_AR_SINGLE if single else NOISE_AR_DUAL, fmt="none",
                    ecolor=C["light"], elinewidth=1.2, capsize=3, zorder=1)
        ax.scatter(x, ar, s=area(size), color=C["orange"] if single else C["blue"], alpha=0.9,
                   edgecolor="black" if "(" in label else "white", lw=1.2, zorder=3)
        (lx, ly), ha, *leader = places[label]
        text = label + ("" if len(draws) == 1 else "\nmean of %d draws" % len(draws))
        if leader and leader[0]:
            ax.annotate(text, xy=(x, ar), xytext=(lx, ly), fontsize=8.5, ha=ha, va="center",
                        zorder=4, arrowprops=dict(arrowstyle="-", color="#777", lw=0.8, shrinkB=6))
        else:
            ax.text(lx, ly, text, fontsize=8.5, ha=ha, va="center", zorder=4)
    for s_mb, xl in ((6.0, 1.62), (55.9, 1.9), (101.1, 2.25)):
        ax.scatter(xl, 0.806, s=area(s_mb), color="none", edgecolor="#555", lw=0.8)
        ax.text(xl, 0.7985, "%.0f MB" % s_mb, ha="center", fontsize=8.5)
    ax.text(1.93, 0.8125, "weight size on disk", ha="center", fontsize=8.5, color="#444")
    ax.plot([], [], "o", color=C["orange"], label="one segmenter (±0.015 AR)")
    ax.plot([], [], "o", color=C["blue"], label="two segmenters (±0.005 AR)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set(xlabel="pick-mode latency, mean per scene [s]  (x86, CPU only, 4 cores)",
           ylabel="AR (leave-scenes-out CV)", xlim=(0.2, 2.6), ylim=(0.794, 0.87),
           title="Segmenter configurations: accuracy against pick latency")
    fig.text(0.125, -0.02, "L: YOLO11l-seg, %.1f MB    nano: YOLO11n-seg, %.1f MB    @N: input side in px\n"
             "synthetic: second segmenter (synthetic-only training), %.1f MB"
             % (mb("part-seg"), mb("part-seg-nano"), mb("part-seg-synthetic")),
             fontsize=8.5, color="#444", va="top")
    save_fig(fig, out, "edge_tradeoff.png")


def board_vs_desktop(root, out):
    """Pick-mode stage times on x86, the Jetson Nano and qemu emulation."""
    platforms = [("x86 desktop", "native"), ("Jetson Nano 4 GB", "board"), ("qemu emulation", "emulated")]
    stages = [("io", C["grey"]), ("setup", C["purple"]), ("segmenter", C["blue"]), ("register", C["green"])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.3})
    width, ticks = 0.19, []
    for gi, (label, name) in enumerate(platforms):
        with open(os.path.join(root, "results", "bench", "%s_nano640.json" % name)) as fh:
            rec = json.load(fh)
        scenes = rec["scenes"].values()
        ticks.append("%s\n(%d scene%s)" % (label, len(scenes), "s" if len(scenes) > 1 else ""))
        for si, (stage, colour) in enumerate(stages):
            v = np.mean([s["stages_s_min"][stage] for s in scenes])
            ax.bar(gi + (si - 1.5) * width, v, width * 0.92, color=colour, label=stage if gi == 0 else None)
            ax.text(gi + (si - 1.5) * width, v * 1.15, "%.2g" % v, ha="center", fontsize=8, color="#333")
        wall = np.mean([s["wall_s_min"] for s in scenes])
        ax.text(gi, 60, "wall %s s\npeak RSS %d MB" % ("%.2f" % wall if wall < 10 else "%.0f" % wall,
                                                       round(rec["peak_rss_mb"])),
                ha="center", va="bottom", fontsize=9)
        ax2.bar(gi, rec["model_load_s"], 0.6, color=C["light"], edgecolor=C["grey"])
        ax2.text(gi, rec["model_load_s"] + 1, "%.1f" % rec["model_load_s"], ha="center", fontsize=9)
    ax.set(yscale="log", ylim=(0.008, 400), xticks=range(len(platforms)), xticklabels=ticks,
           yticks=[0.01, 0.1, 1, 10, 100], yticklabels=["0.01", "0.1", "1", "10", "100"],
           ylabel="seconds per scene, log scale", title="Pick mode per stage (min of repeats, mean over scenes)")
    ax.minorticks_off()
    ax.legend(loc="upper left", ncol=4, fontsize=9, columnspacing=1.2, handlelength=1.2)
    ax.grid(axis="x", visible=False)
    ax2.set(xticks=range(len(platforms)), xticklabels=[p[0].split(" ")[0] for p in platforms],
            ylabel="seconds", title="Model load, one-off", ylim=(0, 48))
    ax2.grid(axis="x", visible=False)
    save_fig(fig, out, "board_vs_desktop.png")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=ROOT_DIR)
    p.add_argument("--out", default=os.path.join(ROOT_DIR, "docs", "figures"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scorer = Scorer(args.root)
    ens = scorer("train_ensemble_run1")
    print("ensemble AR %.3f top-1 %.3f recall %s precision %s TP %s" % (
        ens["ar"], ens["top1"], ["%.3f" % r for r in column(ens, "recall")],
        ["%.3f" % r for r in column(ens, "precision")], column(ens, "tp")))
    assert round(ens["ar"], 3) == 0.851, "headline drifted: AR %.4f" % ens["ar"]
    for label, name, _, _ in TABLE_ROWS[1:]:
        print("%-24s AR %.3f" % (label, scorer(name)["ar"]))
    hero_overlays(args.root, args.out)
    recall_vs_threshold(scorer, args.out)
    failure_breakdown(scorer, args.out)
    ablation_bars(scorer, args.out)
    runtime_breakdown(args.out)
    edge_tradeoff(args.root, scorer, args.out)
    board_vs_desktop(args.root, args.out)


if __name__ == "__main__":
    main()
