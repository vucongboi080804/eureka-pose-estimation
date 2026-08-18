"""Helpers behind scripts/make_figures.py: re-scoring with score.py, figure
saving, and the overlay contact sheet.

Kept apart from the figure definitions so that file reads as a list of
figures; nothing here knows what the figures look like.
"""

import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import score  # noqa: E402  (the official scorer, imported for its matching)

THRESHOLDS_MM = list(score.DEFAULT_THRESHOLDS_MM)


class Scorer:
    """Scores result files with score.py's own functions; train split loaded once."""

    def __init__(self, root):
        import trimesh
        ply = os.path.join(root, "model", "3d_model.ply")
        m = trimesh.load(ply, force="mesh")
        self.mesh = (np.asarray(m.vertices, float), np.asarray(m.faces))
        self.hull = score.load_hull_vertices(ply)
        self.scenes = score.load_scenes(root, "train", os.path.join(root, "train"), self.mesh)
        self.root, self.cache = root, {}

    def __call__(self, name):
        """Aggregate of results/<name>.json: recall/precision per threshold, AR, TP."""
        if name not in self.cache:
            with open(os.path.join(self.root, "results", name + ".json")) as fh:
                sub = json.load(fh)
            per_scene = {sid: score.score_scene(data, sub.get(sid, []), self.hull, self.mesh,
                                                THRESHOLDS_MM, score.DEFAULT_VISIB_MIN)
                         for sid, data in self.scenes.items()}
            self.cache[name] = score.aggregate(per_scene, THRESHOLDS_MM, score.DEFAULT_VISIB_MIN)
        return self.cache[name]

    def ar(self, *names):
        """AR averaged over repeated draws of one configuration."""
        return float(np.mean([self(n)["ar"] for n in names]))


def column(agg, key):
    """One per-threshold column (``recall``, ``precision``, ``tp`` ...) of an aggregate."""
    return [row[key] for row in agg["per_threshold"]]


def save_fig(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote %s (%d KB)" % (path, os.path.getsize(path) // 1024))


def rgb_panel(overlay_path, tile_w, tag):
    """The RGB panel of a visualize.py overlay, resized to ``tile_w`` and tagged.

    The panel is the left axes of the figure: its black frame is the rows and
    columns of the left half that are mostly dark, and the crop is inside it.
    """
    im = cv2.imread(overlay_path)
    dark = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, : im.shape[1] // 2] < 80
    rows = np.where(dark.sum(1) > 0.6 * dark.shape[1])[0]
    cols = np.where(dark.sum(0) > 0.6 * dark.shape[0])[0]
    panel = im[rows[0] + 1:rows[-1], cols[0] + 1:cols[1]]
    h = int(round(panel.shape[0] * tile_w / panel.shape[1]))
    tile = cv2.resize(panel, (tile_w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (78, 28), (40, 40, 40), -1)
    cv2.putText(tile, tag, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def contact_sheet(tiles, per_row, gutter):
    """Tiles of equal width laid out ``per_row`` per row with white gutters."""
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    gap_v = np.full((h, gutter, 3), 255, np.uint8)
    rows = [np.hstack(sum(([t, gap_v] for t in tiles[i:i + per_row]), [])[:-1])
            for i in range(0, len(tiles), per_row)]
    gap_h = np.full((gutter, rows[0].shape[1], 3), 255, np.uint8)
    return np.vstack(sum(([r, gap_h] for r in rows), [])[:-1])


def save_png8(bgr, path):
    """Write a BGR image as an 8-bit palette PNG (few-coloured overlays stay small)."""
    from PIL import Image
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    ).save(path, optimize=True)
    print("wrote %s (%d KB, %dx%d)" % (path, os.path.getsize(path) // 1024, bgr.shape[1], bgr.shape[0]))
