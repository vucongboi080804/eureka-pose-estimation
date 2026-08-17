#!/usr/bin/env bash
# One-shot environment for a fresh clone:
#
#   ./setup.sh                 # auto: CUDA torch when nvidia-smi answers, else CPU
#   ./setup.sh --cpu           # CPU-only torch (air-gapped PC, laptop, CI)
#   ./setup.sh --cuda          # PyPI torch with its CUDA runtime
#   ./setup.sh --jetson-nano   # Python 3.8 / aarch64 pins (deploy/jetson-nano/)
#
# Creates .venv next to this script from the exact pins in
# deploy/requirements-lock.txt (frozen on Linux x86_64, CPython 3.12) and
# checks that the runtime imports. Then: ./run_all.sh <release_path> [split].
#
# Python 3.12 is expected; another version falls back to the unpinned
# requirements.txt with a warning. Pick the interpreter with
# PYTHON=/path/to/python3.x ./setup.sh. The venv comes from `python -m venv`,
# or from uv / virtualenv when the interpreter ships without ensurepip
# (Debian and Ubuntu split that into a python3.x-venv package).
set -euo pipefail
cd "$(dirname "$0")"

FLAVOUR="${1:-auto}"
CPU_INDEX=(--index-url https://download.pytorch.org/whl/cpu
           --extra-index-url https://pypi.org/simple)

if [ "$FLAVOUR" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        FLAVOUR=--cuda
    else
        FLAVOUR=--cpu
    fi
fi

if [ "$FLAVOUR" = "--jetson-nano" ]; then
    PYTHON="${PYTHON:-python3.8}"
else
    PYTHON="${PYTHON:-python3}"
fi
PYTHON_BIN="$(command -v "$PYTHON" || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "error: $PYTHON not found; set PYTHON=/path/to/python3.x" >&2
    exit 1
fi
VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

if [ "$FLAVOUR" = "--jetson-nano" ]; then
    REQ=(-r deploy/jetson-nano/requirements-jetson-nano.txt)
elif [ "$VERSION" = "3.12" ]; then
    REQ=(-r deploy/requirements-lock.txt)
else
    echo "warning: Python $VERSION, not 3.12 -- installing the unpinned requirements.txt" >&2
    REQ=(-r requirements.txt)
fi
[ "$FLAVOUR" = "--cpu" ] && REQ=("${CPU_INDEX[@]}" "${REQ[@]}")

# `python -m venv` first; uv and virtualenv are the fallbacks for an
# interpreter without ensurepip. Each fallback is given the resolved
# interpreter path, so the venv is the Python this script just inspected.
make_venv() {
    rm -rf .venv
    if "$PYTHON_BIN" -m venv .venv >/dev/null 2>&1 && [ -x .venv/bin/pip ]; then
        return 0
    fi
    rm -rf .venv
    if command -v uv >/dev/null 2>&1 \
            && uv venv --python "$PYTHON_BIN" .venv >/dev/null 2>&1; then
        echo "   (python -m venv unavailable; used uv)"
        return 0
    fi
    rm -rf .venv
    if command -v virtualenv >/dev/null 2>&1 \
            && virtualenv -q -p "$PYTHON_BIN" .venv; then
        echo "   (python -m venv unavailable; used virtualenv)"
        return 0
    fi
    echo "error: cannot create a virtual environment with $PYTHON_BIN." >&2
    echo "       Install the venv module (e.g. sudo apt install python$VERSION-venv)" >&2
    echo "       or install uv (https://astral.sh/uv), then re-run." >&2
    return 1
}

# uv-created venvs carry no pip, so uv installs into them directly. Its
# default index strategy stops at the first index holding a package, which
# would hide the PyPI-only wheels behind the torch CPU index.
pip_install() {
    if [ -x .venv/bin/pip ]; then
        .venv/bin/pip install --quiet "$@"
    else
        uv pip install --quiet --python .venv/bin/python \
            --index-strategy unsafe-best-match "$@"
    fi
}

echo "== creating .venv with $PYTHON_BIN ($FLAVOUR, Python $VERSION)"
make_venv
if [ -x .venv/bin/pip ]; then
    .venv/bin/pip install --quiet --upgrade pip
fi
pip_install "${REQ[@]}"
.venv/bin/yolo settings sync=False >/dev/null 2>&1 || true
.venv/bin/python -c "import open3d, cv2, ultralytics, trimesh, scipy, matplotlib, torch; \
print('runtime ok: torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"
echo "Done. Next: ./run_all.sh <release_path> [split]   (WORKERS=2 ./run_all.sh ... on a small CPU)"
