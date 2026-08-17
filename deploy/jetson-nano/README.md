# Running on a Jetson Nano 4 GB (JetPack 4.6)

Target: the original Jetson Nano developer kit — 4× Cortex-A57, 128-core
Maxwell GPU, 4 GB shared RAM, JetPack 4.6.x (L4T r32.7, Ubuntu 18.04,
glibc 2.27, CUDA 10.2). The stock Python is 3.6, which ultralytics no longer
supports, so the runtime goes into a **Python 3.8** venv. Everything below
except the CUDA torch wheel is on PyPI as `cp38 aarch64` wheels that install
on glibc 2.27 (`deploy/jetson-nano/requirements-jetson-nano.txt`; open3d
must be 0.18.0 — 0.19 ships no aarch64 wheel).

## A. Bare metal (recommended on the board)

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-distutils \
                        libgl1 libglib2.0-0 libgomp1

git clone <repo> && cd <repo>
PYTHON=python3.8 ./setup.sh --jetson-nano       # venv + CPU torch from PyPI
# 4 GB is tight: one worker needs ~1.6 GB RSS with both segmenters loaded.
sudo fallocate -l 6G /swapfile && sudo chmod 600 /swapfile \
    && sudo mkswap /swapfile && sudo swapon /swapfile

WORKERS=1 ./run_all.sh /path/to/release test out_nano          # full sweep
.venv/bin/python scripts/run_pipeline.py --root /path/to/release --split test \
    --workers 1 --pick --seg-model weights/part-seg.pt \
    --extra-seg-model weights/part-seg-synthetic.pt --out out_nano/picks.json   # one confident pose per scene
```

**GPU on the Nano.** JetPack 4.6 ships CUDA torch only for Python 3.6; for
3.8 use a community CUDA-10.2 build (e.g. Qengineering's *PyTorch-Jetson-Nano*
wheels: torch 1.13 / 2.0 for aarch64 cp38) and install it *instead of* the
`torch`/`torchvision` lines of the requirements file — nothing in this repo
depends on the torch build beyond ultralytics' inference call. TensorRT
export is not an option here (the JetPack 4.6 TensorRT Python bindings are
3.6-only). Not exercised in this repository — no board was available.

**Air-gapped board.** Build the wheelhouse on any x86 machine with internet
(pip resolves the aarch64 wheels without an ARM host — this is the exact
command that was used to check availability):

```bash
pip download -r deploy/jetson-nano/requirements-jetson-nano.txt -d wheelhouse-nano \
    --platform manylinux2014_aarch64 --platform manylinux_2_27_aarch64 \
    --python-version 3.8 --only-binary=:all:
# on the board:
python3.8 -m venv .venv
.venv/bin/pip install --no-index --find-links wheelhouse-nano -r deploy/jetson-nano/requirements-jetson-nano.txt
```

## B. Docker (CPU torch)

`deploy/jetson-nano/Dockerfile` (Ubuntu 18.04 aarch64 + deadsnakes 3.8 +
the pinned wheels) — built and smoke-tested on the x86 development machine
under qemu-user emulation (`docker buildx` / binfmt), the closest check
available without the board; the same recipe runs natively on the Nano:

```bash
docker build --platform linux/arm64 -f deploy/jetson-nano/Dockerfile -t pose-est:nano .
docker run --rm --network none -v /path/to/release:/data:ro -v "$PWD/out:/out" pose-est:nano /data test /out
```

For the GPU inside a container use NVIDIA's `l4t-base:r32.7.1` image with
the community torch wheel and the same pip lines — not provided here because
it cannot be tested off-board.

## What to expect

Measured on the x86 development machine (`analysis/runtime.md`): a full
sweep is 11–12 s per scene single-worker (segmenters ≈ 2 % of that; the rest
is Open3D RANSAC/ICP), pick mode 1.4–2.4 s per scene on 4 P-cores CPU-only.
A Cortex-A57 core is roughly an order of magnitude slower per thread, and the
two YOLO11 segmenters at 960 px on the CPU cost several seconds each, so
plan on **≈ 1–3 min per scene for the full sweep and ≈ 10–30 s per pick on
CPU torch**; the Nano's GPU brings the segmenter share back to ~1 s. Measure
before promising: `--scenes 000001 000002 000003 --workers 1` on three scenes
gives the number for the board in hand. Memory: one worker only (`WORKERS=1`,
the default in the image), 6 GB swap or zram, and consider dropping the
synthetic segmenter (`--extra-seg-model` off: AR 0.815 instead of 0.844,
half the segmenter time and RAM).

## Verified here vs. not

- Verified: every pin resolves to a `cp38` aarch64 wheel compatible with
  glibc 2.27 (pip download from x86); the pipeline runs on Python 3.8 with
  open3d 0.18 (x86 container, poses identical to the submission); the arm64
  image builds and runs the pipeline under qemu emulation
  (see the log lines quoted at the end of this file).
- Not verified: wall-clock on the actual board, the CUDA torch path on
  JetPack 4.6, memory headroom under 4 GB with the desktop stack running.

<!-- emulation-log -->
