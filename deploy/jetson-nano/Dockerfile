# Jetson Nano (JetPack 4.6 / Ubuntu 18.04, aarch64) inference image, CPU
# torch. Built and smoke-tested on an x86 host under qemu-user emulation
# (docker buildx / binfmt), which is the closest check available without the
# board; on the board itself the same recipe runs natively.
#
#   docker build --platform linux/arm64 -f deploy/jetson-nano/Dockerfile -t pose-est:nano .
#   docker run --rm --network none -v /path/to/release:/data:ro -v $PWD/out:/out \
#       pose-est:nano /data test /out
#
# For the Nano's GPU use the bare-metal recipe in deploy/jetson-nano/README.md
# (JetPack 4.6 CUDA torch for Python 3.8 is not on PyPI, so it cannot be
# pinned here).

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
# CPython 3.8 from deadsnakes (the stock 3.6 is too old for ultralytics) and
# the shared libraries the opencv / open3d wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3.8-distutils \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/jetson-nano/requirements-jetson-nano.txt deploy/jetson-nano/
RUN python3.8 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r deploy/jetson-nano/requirements-jetson-nano.txt

COPY src/ src/
COPY scripts/ scripts/
COPY weights/ weights/
COPY score.py visualize.py run_all.sh ./

ENV HOME=/tmp/home \
    YOLO_CONFIG_DIR=/tmp/yolo \
    MPLCONFIGDIR=/tmp/mpl \
    MPLBACKEND=Agg \
    YOLO_OFFLINE=1 \
    YOLO_AUTOINSTALL=0 \
    YOLO_VERBOSE=0 \
    OMP_NUM_THREADS=4 \
    WORKERS=1 \
    PATH=/app/.venv/bin:$PATH
RUN mkdir -p /tmp/home /tmp/yolo /tmp/mpl && chmod -R a+rwx /tmp/home /tmp/yolo /tmp/mpl \
    && yolo settings sync=False \
    && python -c "import open3d, cv2, ultralytics, trimesh, scipy, matplotlib; print('runtime imports ok')"

ENTRYPOINT ["./run_all.sh"]
CMD ["/data", "test", "/out"]
