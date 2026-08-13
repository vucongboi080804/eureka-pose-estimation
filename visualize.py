"""Draw the ground truth over each image: the masks, and the pose as axes.

Run this first. If the axes sit on the parts and point consistently, you have
read the conventions correctly and everything you build can rely on them.

    python visualize.py --root . --split train
    python visualize.py --root . --split train --save overlays/
    python visualize.py --root . --split train --scenes 000007 000020

Needs only numpy, opencv-python and matplotlib. The pose is drawn by
``cv2.drawFrameAxes`` straight from ``R``, ``t`` and ``K`` -- no mesh is
required to draw a pose, so the CAD is never read here.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

#: Axis length in model units (metres).
AXIS_LENGTH = 0.015

#: Axis line thickness. 3 rather than 2: `cv2.drawFrameAxes` draws flat lines
#: with no outline, and a red axis over an orange part needs the weight.
AXIS_THICKNESS = 3

MASK_COLOUR = (0, 255, 255)    # BGR: yellow
IGNORE_COLOUR = (0, 0, 255)    # BGR: red
FILL_ALPHA = 0.25


def draw_pose(canvas, R, t, K, length=AXIS_LENGTH, thickness=AXIS_THICKNESS):
    """Draw the object frame at the pose: X red, Y green, Z blue.

    ``cv2.drawFrameAxes`` takes exactly what `poses.json` provides -- its
    ``rvec``/``tvec`` pair *is* the object-to-camera transform, and ``length``
    is in the same units as ``tvec`` -- so the pose needs no mesh and no
    hand-rolled projection to draw. Distortion is zero for this dataset (see
    `camera.json`), hence the zero coefficients.

    Args:
        canvas: BGR image, modified in place.
        R: ``(3, 3)`` rotation, object -> camera.
        t: ``(3,)`` translation in model units.
        K: ``(3, 3)`` intrinsics.
        length: Axis length in model units.
        thickness: Line thickness in pixels.
    """
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    cv2.drawFrameAxes(canvas, np.asarray(K, dtype=np.float64), np.zeros(5),
                      rvec, np.asarray(t, dtype=np.float64), length, thickness)


def contours(mask):
    """Outer *and* inner contours of a binary mask, so holes are outlined too."""
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                             cv2.CHAIN_APPROX_SIMPLE)
    return found[0] if len(found) == 2 else found[1]


def label_at(canvas, mask, text):
    """Stamp ``text`` at a mask's centroid, outlined so it reads on any colour."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    at = (int(xs.mean()) - 6, int(ys.mean()) + 6)
    cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render(scene_dir, label_dir, axis_length=AXIS_LENGTH):
    """Composite one scene.

    Args:
        scene_dir: Folder holding ``rgb.png``, ``depth.png``, ``camera.json``.
        label_dir: Folder holding ``poses.json``, ``masks/``, ``ignore/``. Same
            as ``scene_dir`` for a split that ships its labels.
        axis_length: Pose arrow length in model units.

    Returns:
        ``(overlay_rgb, depth_metres, info)``.
    """
    bgr = cv2.imread(os.path.join(scene_dir, "rgb.png"))
    if bgr is None:
        raise FileNotFoundError(os.path.join(scene_dir, "rgb.png"))
    with open(os.path.join(scene_dir, "camera.json")) as fh:
        cam = json.load(fh)
    K = np.asarray(cam["K"], dtype=np.float64)
    depth = cv2.imread(os.path.join(scene_dir, "depth.png"),
                       cv2.IMREAD_UNCHANGED).astype(np.float64) * cam["depth_scale"]

    poses = []
    poses_path = os.path.join(label_dir, "poses.json")
    if os.path.exists(poses_path):
        with open(poses_path) as fh:
            poses = json.load(fh)

    canvas = bgr.copy()

    for i, pose in enumerate(poses):
        R = np.asarray(pose["R"], dtype=np.float64)
        t = np.asarray(pose["t"], dtype=np.float64)
        mask = cv2.imread(os.path.join(label_dir, pose["mask"]), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            m = mask > 0
            canvas[m] = ((1 - FILL_ALPHA) * canvas[m]
                         + FILL_ALPHA * np.array(MASK_COLOUR)).astype(np.uint8)
            cv2.drawContours(canvas, contours(m), -1, MASK_COLOUR, 1)

        draw_pose(canvas, R, t, K, axis_length)
        if mask is not None:
            label_at(canvas, mask > 0, str(i))

    n_ignore = 0
    for path in sorted(glob.glob(os.path.join(label_dir, "ignore", "*.png"))):
        ign = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ign is not None:
            cv2.drawContours(canvas, contours(ign > 0), -1, IGNORE_COLOUR, 2)
            n_ignore += 1

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), depth, {
        "n_poses": len(poses), "n_ignore": n_ignore,
        "size": (int(cam["width"]), int(cam["height"])),
    }


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Overlay ground-truth masks and poses on each scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default=".",
                   help="Dataset root (the folder holding train/ and test/)")
    p.add_argument("--split", default="train", help="Which split to walk")
    p.add_argument("--labels", default=None,
                   help="Folder of per-scene labels, if they do not ship beside "
                        "the images. Point this at a held-back split's ground truth.")
    p.add_argument("--scenes", nargs="*", default=None, help="Only these scene ids")
    p.add_argument("--limit", type=int, default=0, help="Stop after N scenes (0 = all)")
    p.add_argument("--save", default=None,
                   help="Write figures here instead of opening a window")
    p.add_argument("--axis-length", type=float, default=AXIS_LENGTH,
                   help="Pose arrow length, model units (metres)")
    p.add_argument("--no-depth", action="store_true", help="Show the image panel only")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    split_dir = os.path.join(args.root, args.split)
    if not os.path.isdir(split_dir):
        print("No %s under %s -- is --root the dataset root?" % (args.split, args.root))
        return 2
    scenes = sorted(s for s in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, s)))
    if args.scenes:
        scenes = [s for s in scenes if s in set(args.scenes)]
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes:
        print("No scenes to show")
        return 2
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    for scene in scenes:
        scene_dir = os.path.join(split_dir, scene)
        label_dir = os.path.join(args.labels, scene) if args.labels else scene_dir
        rgb, depth, info = render(scene_dir, label_dir, args.axis_length)

        print("%s/%s  %dx%d  %d labelled, %d ignore"
              % (args.split, scene, info["size"][0], info["size"][1],
                 info["n_poses"], info["n_ignore"]))

        ncols = 1 if args.no_depth else 2
        fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), squeeze=False)
        axes = axes[0]
        axes[0].imshow(rgb)
        axes[0].set_title(
            "%s/%s  -  yellow: mask, red: ignore, axes: X red / Y green / Z blue"
            % (args.split, scene), fontsize=9)
        if not args.no_depth:
            shown = axes[1].imshow(np.ma.masked_equal(depth, 0), cmap="viridis")
            axes[1].set_title("depth (m); blank = no measurement", fontsize=9)
            fig.colorbar(shown, ax=axes[1], fraction=0.035)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()

        if args.save:
            out = os.path.join(args.save, "%s.png" % scene)
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print("   -> %s" % out)
        else:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
