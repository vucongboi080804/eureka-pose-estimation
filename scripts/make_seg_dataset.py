"""Convert train scenes into a YOLO-segmentation dataset.

Every labelled mask AND every ignore mask becomes a "part" instance: for
segmentation they are all equally real parts (ignore only means no pose
label), and the scorer discards detections on ignore regions anyway.

Fold support: pass --val-scenes to hold scenes out as the validation set;
the same flag builds the folds for an honest leave-scenes-out evaluation
of the segmenter on the training split.

    .venv/bin/python scripts/make_seg_dataset.py --root . \
        --out seg_data/fold0 --val-scenes 000007 000019 000030 000047 000059
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scene_io import list_scenes


def polygons_of(mask: np.ndarray) -> list:
    """Outer contours of a binary mask as flat [x1 y1 x2 y2 ...] lists."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        c = c.reshape(-1, 2).astype(np.float64)
        out.append(c)
    return out


def write_scene(root, scene_id, img_dir, lbl_dir):
    src_img = os.path.join(root, "train", scene_id, "rgb.png")
    shutil.copy(src_img, os.path.join(img_dir, scene_id + ".png"))
    h, w = cv2.imread(src_img).shape[:2]

    lines = []
    mask_paths = (sorted(glob.glob(os.path.join(root, "train", scene_id,
                                                "masks", "*.png")))
                  + sorted(glob.glob(os.path.join(root, "train", scene_id,
                                                  "ignore", "*.png"))))
    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0
        for poly in polygons_of(mask):
            norm = (poly / [w, h]).reshape(-1)
            lines.append("0 " + " ".join("%.5f" % v for v in norm))
    with open(os.path.join(lbl_dir, scene_id + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--val-scenes", nargs="*", default=[])
    args = p.parse_args()

    scenes = list_scenes(args.root, "train")
    val = set(args.val_scenes)
    for split, ids in (("train", [s for s in scenes if s not in val]),
                       ("val", sorted(val) or [scenes[0]])):
        img_dir = os.path.join(args.out, "images", split)
        lbl_dir = os.path.join(args.out, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n = sum(write_scene(args.root, s, img_dir, lbl_dir) for s in ids)
        print("%s: %d scenes, %d instance polygons" % (split, len(ids), n))

    with open(os.path.join(args.out, "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: images/train\nval: images/val\n"
                 "names:\n  0: part\n" % os.path.abspath(args.out))
    print("wrote", os.path.join(args.out, "data.yaml"))


if __name__ == "__main__":
    main()
