#!/usr/bin/env bash
# One-shot environment for a fresh clone:
#
#   ./setup.sh            # auto: CUDA torch if nvidia-smi is present, else CPU
#   ./setup.sh --cpu      # CPU-only torch (air-gapped PC, laptop, CI)
#   ./setup.sh --cuda     # PyPI torch with its CUDA runtime
#   ./setup.sh --jetson-nano   # Python 3.8 / aarch64 pins (deploy/jetson-nano/)
#
# Creates .venv next to this script from the exact pins in
# deploy/requirements-lock.txt (frozen on Linux x86_64, CPython 3.12) and
# checks the runtime imports. Then: ./run_all.sh <release_path> [split].
# Python 3.12 is expected; other versions fall back to the unpinned
# requirements.txt with a warning. Set PYTHON=/path/to/python3.x to choose.
set -euo pipefail
cd "$(dirname "$0")"

FLAVOUR="${1:-auto}"
PYTHON="${PYTHON:-python3}"
CPU_INDEX=(--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple)

if [ "$FLAVOUR" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        FLAVOUR=--cuda
    else
        FLAVOUR=--cpu
    fi
fi

if [ "$FLAVOUR" = "--jetson-nano" ]; then
    PYTHON="${PYTHON:-python3.8}"
    [ "$PYTHON" = python3 ] && PYTHON=python3.8
    REQ=(-r deploy/jetson-nano/requirements-jetson-nano.txt)
else
    VER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [ "$VER" = "3.12" ]; then
        REQ=(-r deploy/requirements-lock.txt)
    else
        echo "warning: Python $VER, not 3.12 -- installing unpinned requirements.txt" >&2
        REQ=(-r requirements.txt)
    fi
    [ "$FLAVOUR" = "--cpu" ] && REQ=("${CPU_INDEX[@]}" "${REQ[@]}")
fi

echo "== creating .venv with $PYTHON ($FLAVOUR)"
"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet "${REQ[@]}"
.venv/bin/yolo settings sync=False >/dev/null 2>&1 || true
.venv/bin/python -c "import open3d, cv2, ultralytics, trimesh, scipy, matplotlib, torch; \
print('runtime ok: torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"
echo "Done. Next: ./run_all.sh <release_path> [split]   (WORKERS=2 ./run_all.sh ... on a small CPU)"
