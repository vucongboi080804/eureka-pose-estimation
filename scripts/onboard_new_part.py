"""Onboard a NEW part with zero hand labels: render, train, done.

    .venv/bin/python scripts/onboard_new_part.py --cad path/to/new_part.ply \
        --workdir onboard_newpart [--frames 1500] [--epochs 150]

The pose stack needs nothing else: registration, verification, flips,
grid fallback and hole discovery all read the CAD at load time. Only the
segmenter is part-specific, and this script manufactures its training
data by domain-randomised rendering (`render_synthetic.py`) — part
colour, lighting, trays and backgrounds are all randomised, so the model
generalises to environments nobody photographed.

Afterwards:

    .venv/bin/python scripts/run_pipeline.py --root <release> --split test \
        --out submission.json --seg-model <workdir>/train/weights/best.pt
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cad", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--frames", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--val-fraction", type=float, default=0.05)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.dirname(sys.executable)
    data_dir = os.path.join(args.workdir, "data")

    run([os.path.join(venv, "blenderproc"), "run",
         os.path.join(here, "render_synthetic.py"), "--",
         "--cad", args.cad, "--out", data_dir,
         "--frames", str(args.frames)])

    # Carve a validation split off the rendered frames.
    img_dir = os.path.join(data_dir, "images", "train")
    lbl_dir = os.path.join(data_dir, "labels", "train")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(data_dir, sub, "val"), exist_ok=True)
    frames = sorted(os.listdir(img_dir))
    n_val = max(1, int(len(frames) * args.val_fraction))
    for name in frames[-n_val:]:
        stem = os.path.splitext(name)[0]
        shutil.move(os.path.join(img_dir, name),
                    os.path.join(data_dir, "images", "val", name))
        shutil.move(os.path.join(lbl_dir, stem + ".txt"),
                    os.path.join(data_dir, "labels", "val", stem + ".txt"))

    with open(os.path.join(data_dir, "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: images/train\nval: images/val\n"
                 "names:\n  0: part\n" % os.path.abspath(data_dir))

    run([os.path.join(venv, "yolo"), "segment", "train",
         "model=yolo11m-seg.pt",
         "data=%s" % os.path.join(data_dir, "data.yaml"),
         "project=%s" % os.path.abspath(args.workdir), "name=train",
         "exist_ok=True", "imgsz=960", "epochs=%d" % args.epochs,
         "patience=40", "batch=8", "amp=False", "optimizer=AdamW",
         "lr0=0.0002", "cos_lr=True", "fliplr=0.5", "flipud=0.5"])

    print("\nDone. Weights: %s"
          % os.path.join(args.workdir, "train", "weights", "best.pt"))


if __name__ == "__main__":
    main()
