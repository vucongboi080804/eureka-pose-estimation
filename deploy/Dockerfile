# CPU-only inference image for an air-gapped industrial PC.
#
#   docker build -f deploy/Dockerfile -t pose-est:cpu .
#   docker run --rm --network none -v /path/to/release:/data:ro -v $PWD/out:/out \
#       pose-est:cpu /data test /out
#
# The build context is the repository root (src/, scripts/, weights/,
# score.py, visualize.py, run_all.sh); keep datasets and .venv out of it with
# a .dockerignore, they are not part of the image. The release folder
# (model/ + a split of scenes) is mounted at run time, never baked in.
# Everything the runtime needs is installed at build; inference makes no
# network request (deploy/OFFLINE.md).

FROM python:3.12-slim-bookworm

# Shared libraries the opencv-python and open3d wheels link against
# (libGL/X11 are load-time deps only -- nothing is rendered on screen).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libsm6 libice6 \
        libudev1 libcap2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# A venv at .venv keeps run_all.sh (PY=.venv/bin/python) unchanged.
COPY deploy/requirements-lock.txt deploy/requirements-lock.txt
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r deploy/requirements-lock.txt

COPY src/ src/
COPY scripts/ scripts/
COPY weights/ weights/
COPY score.py visualize.py run_all.sh ./

# Writable caches under /tmp so the image also runs read-only or as any uid.
# YOLO_OFFLINE short-circuits ultralytics' DNS reachability probe and its
# usage events; sync=False disables them regardless of the online state.
ENV HOME=/tmp/home \
    YOLO_CONFIG_DIR=/tmp/yolo \
    MPLCONFIGDIR=/tmp/mpl \
    MPLBACKEND=Agg \
    YOLO_OFFLINE=1 \
    YOLO_AUTOINSTALL=0 \
    YOLO_VERBOSE=0 \
    OMP_NUM_THREADS=4 \
    WORKERS=2 \
    PATH=/app/.venv/bin:$PATH
RUN mkdir -p /tmp/home /tmp/yolo /tmp/mpl && chmod -R a+rwx /tmp/home /tmp/yolo /tmp/mpl \
    && yolo settings sync=False \
    && python -c "import open3d, cv2, ultralytics, trimesh, scipy, matplotlib; print('runtime imports ok')"

# Default: whole test split of the mounted release into /out.
# Override the command for other splits, output dirs or a direct
# scripts/run_pipeline.py invocation.
ENTRYPOINT ["./run_all.sh"]
CMD ["/data", "test", "/out"]
