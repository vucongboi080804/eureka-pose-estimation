# Jetson Nano (JetPack 4.6 / Ubuntu 18.04, aarch64) inference image, CPU
# torch. Built and smoke-tested on an x86 host under qemu-user emulation
# (docker buildx / binfmt), which is the closest check available without the
# board; on the board itself the same recipe runs natively. Build it on a
# workstation and `docker save` it over -- do not build on the Nano.
#
#   docker build --platform linux/arm64 -f deploy/board/Dockerfile -t pose-est:nano .
#
# Two ways to run it. The cell wants the service: a long-lived process that
# holds the CAD cloud and the segmenter in memory and answers /v1/estimate
# per frame. --network host puts its 127.0.0.1:8080 on the board's loopback,
# which is where the cell talks to it (deploy/board/README.md).
#
#   docker run --rm --network host -v /path/to/release/model:/app/model:ro \
#       --entrypoint python pose-est:nano \
#       -m deploy.pose.server --config /app/deploy/board/config.nano.json
#
# The default entrypoint is still the batch pipeline over a whole split,
# offline. The HEALTHCHECK below probes the service port, so a batch
# container needs --no-healthcheck or it is reported unhealthy while working
# perfectly.
#
#   docker run --rm --network none --no-healthcheck \
#       -v /path/to/release:/data:ro -v $PWD/out:/out pose-est:nano /data test /out
#
# For the Nano's GPU use the bare-metal recipe in deploy/board/README.md
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
COPY deploy/board/requirements-jetson-nano.txt deploy/board/
RUN python3.8 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r deploy/board/requirements-jetson-nano.txt

COPY src/ src/
COPY scripts/ scripts/
COPY weights/ weights/
# The whole cell (camera/, pose/, pick/, demo/) and the board profile in
# board/; `python -m deploy.pose.server` resolves against WORKDIR.
COPY deploy/ deploy/
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

# Must match "port" in deploy/board/config.nano.json.
EXPOSE 8080
# /healthz stays not-ready until the models are loaded and warmed, which is
# tens of seconds on four A57 cores -- hence the start period. urlopen raises
# on a non-2xx status, so a failed probe is a non-zero exit on its own.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=5).read()"]

ENTRYPOINT ["./run_all.sh"]
CMD ["/data", "test", "/out"]
