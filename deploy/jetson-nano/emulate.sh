#!/usr/bin/env bash
# Run deploy/jetson-nano/bench.py inside the aarch64 image under the board's
# limits, so the emulated record describes the same envelope the Nano has.
#
# The limits are not cosmetic. Without --cpus/--memory the container sees the
# host's cores and RAM, bench.py records that, and the resulting file claims
# an envelope the board never had. With them the record carries the cgroup
# budget instead, and compare_bench.py prints 4 cores / 4 GB on both sides.
#
#   deploy/jetson-nano/emulate.sh --check     # print the command and the caveat
#   deploy/jetson-nano/emulate.sh             # one scene, pick mode, ~4 min
#   SCENES="000001 000002" PICK=0 deploy/jetson-nano/emulate.sh
#
# What emulation cannot tell you: qemu-user emulates the ARMv8 instruction
# set, not a Cortex-A57. There is no A57 cache hierarchy, no LPDDR4 bandwidth
# and no thermal envelope behind these numbers, and qemu's own translation
# overhead dominates them (a pick measured 259 s emulated against 1.5 s
# native). Use this run to prove the aarch64 build produces the same poses;
# run bench.py on the board for the time.
#
# Environment overrides: IMAGE CPUS MEMORY MEMORY_SWAP DOCKER_USER RELEASE
#                        SPLIT SCENES PICK REPEAT PROFILE EXTRA_SEG IMGSZ
#                        OUT_DIR OUT_NAME NOTE
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

IMAGE="${IMAGE:-pose-est:nano}"
CPUS="${CPUS:-4}"                     # Nano: 4x Cortex-A57
MEMORY="${MEMORY:-4g}"                # Nano: 4 GB shared with the GPU
MEMORY_SWAP="${MEMORY_SWAP:-6g}"      # 4 GB RAM + the 2 GB swapfile the README adds
RELEASE="${RELEASE:-$ROOT}"           # release folder: <split>/ and model/
SPLIT="${SPLIT:-test}"
SCENES="${SCENES:-000001}"            # a full sweep of one scene is 30+ min here
PICK="${PICK:-1}"                     # 1: deployment latency mode
REPEAT="${REPEAT:-1}"
PROFILE="${PROFILE:-nano}"
# The board profile is what config.nano.json runs, so it is what the emulated
# baseline has to measure: one segmenter at 640. Anything else and the board
# run it is diffed against is answering a different question -- which
# compare_bench.py will say, loudly, rather than compare regardless.
# EXTRA_SEG=/app/weights/part-seg-synthetic.pt IMGSZ=960 measures the shipped
# desktop configuration instead.
EXTRA_SEG="${EXTRA_SEG:-}"
IMGSZ="${IMGSZ:-640}"
OUT_DIR="${OUT_DIR:-$ROOT/out_bench}"
OUT_NAME="${OUT_NAME:-emulated.json}"
# So the record lands owned by whoever ran the script, not by root. The image
# keeps its caches under a world-writable /tmp, so it needs no account here.
DOCKER_USER="${DOCKER_USER:-$(id -u):$(id -g)}"
NOTE="${NOTE:-qemu-user aarch64 on the x86 development machine, board limits applied}"

[ "$PICK" = "1" ] && PICK_FLAG="--pick" || PICK_FLAG=""

# bench.py is bind-mounted over whatever the image carries, so editing the
# benchmark does not cost a 2.2 GB rebuild. --no-healthcheck because the
# image's probe watches the service port: a bench container is not the
# service and would otherwise be reported unhealthy while working perfectly.
DOCKER_ARGS=(
    run --rm --platform linux/arm64 --network none --no-healthcheck
    --cpus "$CPUS" --memory "$MEMORY" --memory-swap "$MEMORY_SWAP"
    --user "$DOCKER_USER"
    -e "BENCH_GIT_COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    -v "$RELEASE:/data:ro"
    -v "$ROOT/deploy/jetson-nano:/app/deploy/jetson-nano:ro"
    -v "$OUT_DIR:/out"
    --entrypoint python "$IMAGE"
    deploy/jetson-nano/bench.py
    --root /data --split "$SPLIT" --scenes $SCENES
    --profile "$PROFILE" --repeat "$REPEAT" $PICK_FLAG
    --extra-seg-model "$EXTRA_SEG" --imgsz "$IMGSZ"
    --note "$NOTE" --out "/out/$OUT_NAME"
)

if [ "${1:-}" = "--check" ]; then
    echo "would run:"
    printf '  docker'; printf ' %q' "${DOCKER_ARGS[@]}"; printf '\n\n'
    echo "image     $IMAGE   (built by deploy/jetson-nano/Dockerfile; do not rebuild to change bench.py -- it is mounted)"
    echo "limits    $CPUS CPUs, $MEMORY RAM, $MEMORY_SWAP RAM+swap, no network"
    echo "profile   ${EXTRA_SEG:-one segmenter}, imgsz $IMGSZ  (config.nano.json's)"
    echo "release   $RELEASE -> /data (read-only)"
    echo "output    $OUT_DIR/$OUT_NAME"
    echo
    echo "caveat    qemu-user emulates the ARMv8 instruction set, not the"
    echo "          microarchitecture: no Cortex-A57 cache hierarchy, no LPDDR4"
    echo "          bandwidth, no thermal limit, plus qemu's own translation"
    echo "          overhead. Treat the ratio as an upper bound on the board's"
    echo "          time and re-measure on the board. What this run does prove"
    echo "          is that the aarch64 build returns the same poses."
    exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "no such image: $IMAGE" >&2
    echo "build it once (2.2 GB):" >&2
    echo "  docker build --platform linux/arm64 -f deploy/jetson-nano/Dockerfile -t $IMAGE $ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
docker "${DOCKER_ARGS[@]}"
echo
echo "$OUT_DIR/$OUT_NAME"
