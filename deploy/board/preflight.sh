#!/usr/bin/env bash
# The first thing to run against a Jetson before anything is copied to it:
# can this machine deploy to that board, and is that board what it claims to
# be? Most failed bring-ups are one of a handful of boring facts -- the board
# is on another subnet, sshd never came up, the SD card has 800 MB left, the
# stock Python is 3.6, the clock is two years behind -- and every one of them
# is cheaper to learn here than halfway through an install.
#
#   deploy/board/preflight.sh <user@host> [--port N] [--identity KEY]
#                                   [--mode venv|docker]
#
# Examples:
#   deploy/board/preflight.sh pose@192.168.102.39
#   deploy/board/preflight.sh pose@nano.local --port 2222 --identity ~/.ssh/pose-nano
#
# Exit codes, so a wrapper can branch on them:
#   0  every check passed        -- go ahead and deploy
#   1  at least one hard failure -- fix it first; the board is not ready
#   2  warnings only             -- deployable, with the caveats printed
#
# This script changes NOTHING. It reads the board over one SSH connection and
# writes no file on either side, so it is safe to run repeatedly, on a board
# that is already serving picks, and against a board someone else is using.
# Anything that installs, removes or restarts is a different script's job.
set -euo pipefail

# --- what this bring-up assumes -------------------------------------------
# The runbook's prefix. Only the FILESYSTEM this sits on matters here; the
# directory itself is usually absent on a fresh board, which is not an error.
INSTALL_PREFIX=/opt/pose-estimation
# The key this bring-up uses. Named rather than shared with the operator's
# personal key so it can be revoked from the board's authorized_keys alone.
DEFAULT_KEY="${HOME}/.ssh/pose-nano"
BOARD_CONFIG=config.nano.json          # the profile the board serves

# A board on a bench answers in milliseconds; 8 s is long enough to cross a
# slow switch and short enough that a dead board does not hold up the report.
SSH_CONNECT_TIMEOUT_S=8
TCP_TIMEOUT_S=5
PING_TIMEOUT_S=2
# bench.py records are matched up by their timestamps, and the Nano has no RTC
# battery: off the network its clock restarts at the last shutdown time. A
# minute of skew is more than NTP or a manual `date -s` ever leaves behind.
CLOCK_SKEW_MAX_S=60
# The runbook's swapfile (2 GB on top of 4 GB of RAM) is the envelope
# emulate.sh reproduces and the one the service's MemoryMax was sized against.
MIN_SWAP_MB=2048
# Measured, not guessed: `du -sm /app/.venv` in pose-est:nano -- the same
# aarch64 stack this board installs -- is 1262 MB; the wheels in
# requirements-jetson-nano.lock.txt total ~340 MB staged on the board while
# pip runs; the repo payload provision.sh copies is 7 MB, because the board
# gets part-seg-nano.pt alone and not the two desktop weights. That is ~1.6 GB
# to land, and an install that lands with nothing to spare fails later at the
# journal or the next release instead of here.
MIN_FREE_MB=2500
COMFORT_FREE_MB=4000
# open3d 0.18.0 ships a manylinux_2_27_aarch64 wheel and there is no older
# release with an aarch64 wheel at all, so glibc 2.27 (Ubuntu 18.04) is the
# floor for the whole stack, not a preference.
MIN_GLIBC=2.27

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
    # The header comment above is the help text; print it until it runs out.
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

# --- reporting ------------------------------------------------------------
PASSES=0; WARNS=0; FAILS=0
declare -a SUMMARY=()

line()  { printf '  %-4s  %-9s %s\n' "$1" "$2" "$3"; }
pass()  { line PASS "$1" "$2"; PASSES=$((PASSES+1)); }
warn()  { line WARN "$1" "$2"; WARNS=$((WARNS+1)); }
fail()  { line FAIL "$1" "$2"; FAILS=$((FAILS+1)); }
info()  { line INFO "$1" "$2"; }
# The fix belongs next to the failure, not in a document the operator has to
# go and find. Each argument is one copy-pasteable line.
fix()   { printf '        %s\n' "$@"; }
section() { printf '\n== %s ==\n' "$1"; }
remember() { SUMMARY+=("$1"); }

TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT
SSH_ERR="$TMPDIR_RUN/ssh.err"

# --- arguments ------------------------------------------------------------
TARGET=""; PORT=22; IDENTITY=""
#: Which runtime the board will use. "auto" prefers an existing python3.8 and
#: falls back to docker when the board has docker and no 3.8 -- which is the
#: normal state of a stock JetPack 4.6 image, where demanding a 3.8 the docker
#: path never needs would be a false failure.
MODE="auto"
RUNTIME=venv
while [ $# -gt 0 ]; do
    case "$1" in
        --port|-p)     PORT="${2:?--port needs a number}"; shift 2 ;;
        --identity|-i) IDENTITY="${2:?--identity needs a key path}"; shift 2 ;;
        --mode)        MODE="${2:?--mode needs venv or docker}"; shift 2 ;;
        -h|--help)     usage 0 ;;
        -*)            echo "unknown option: $1" >&2; usage 1 >&2 ;;
        *)             [ -z "$TARGET" ] || { echo "one target only: already have $TARGET" >&2; exit 1; }
                       TARGET="$1"; shift ;;
    esac
done
[ -n "$TARGET" ] || { echo "usage: $0 <user@host> [--port N] [--identity KEY]" >&2; exit 1; }
case "$PORT" in ''|*[!0-9]*) echo "--port must be a number, got: $PORT" >&2; exit 1 ;; esac
HOST="${TARGET##*@}"
[ "$HOST" != "$TARGET" ] || info target "no user in '$TARGET' -- ssh will use $(id -un)"

printf 'preflight for %s (port %s)\n' "$TARGET" "$PORT"

# --- 0. this machine ------------------------------------------------------
# The board install copies weights and the CAD model from this checkout. A
# missing one of those is a startup failure on the board, hours later.
section "this machine"
CONF="$REPO_ROOT/deploy/board/$BOARD_CONFIG"
if [ -r "$CONF" ]; then
    missing=""
    for key in seg_weights cad_path; do
        rel="$(sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$CONF")"
        [ -n "$rel" ] || continue
        [ -e "$REPO_ROOT/$rel" ] || missing="$missing $rel"
    done
    if [ -n "$missing" ]; then
        fail payload "$BOARD_CONFIG names files this checkout does not have:$missing"
        fix "the release folder carries model/; weights/ comes with the repo -- copy them in before deploying"
    else
        pass payload "$BOARD_CONFIG's weights and CAD model are present here"
    fi
else
    fail payload "no $CONF -- run this from the repository it ships in"
fi

if [ -z "$IDENTITY" ] && [ -f "$DEFAULT_KEY" ]; then
    IDENTITY="$DEFAULT_KEY"
fi
if [ -n "$IDENTITY" ]; then
    IDENTITY="${IDENTITY/#\~/$HOME}"
    if [ -r "$IDENTITY" ]; then
        info key "$IDENTITY"
    else
        fail key "$IDENTITY is not readable"
        fix "ssh-keygen -t ed25519 -f $IDENTITY -N '' -C pose-nano"
    fi
else
    info key "no $DEFAULT_KEY -- ssh will offer whatever ~/.ssh holds"
fi

SSH_OPTS=(-o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT_S"
          -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -p "$PORT")
if [ -n "$IDENTITY" ] && [ -r "$IDENTITY" ]; then
    SSH_OPTS+=(-i "$IDENTITY" -o IdentitiesOnly=yes)
fi

# --- 1. reachability ------------------------------------------------------
section "reaching $HOST"

# ICMP is informational on purpose: plenty of sane networks drop it, so a
# silent ping is not evidence of anything. It is here because when it DOES
# answer, a failing TCP connect below means sshd, not the network.
if command -v ping >/dev/null 2>&1; then
    if ping -c 1 -W "$PING_TIMEOUT_S" "$HOST" >/dev/null 2>&1; then
        info ping "answers (ICMP)"
    else
        info ping "no reply -- often just a firewall; the TCP check below decides"
    fi
else
    info ping "no ping on this machine; skipped"
fi

# bash's /dev/tcp rather than nc, which is three different programs with three
# different flag sets. The failure TEXT is the whole point: "no route",
# "refused" and "timed out" have different fixes.
tcp_err=""; tcp_rc=0
tcp_err="$(P_HOST="$HOST" P_PORT="$PORT" timeout "$TCP_TIMEOUT_S" \
           bash -c 'exec 3<>/dev/tcp/$P_HOST/$P_PORT' 2>&1)" || tcp_rc=$?
if [ "$tcp_rc" -eq 0 ]; then
    pass tcp "$HOST:$PORT accepts connections"
else
    case "$tcp_rc:$tcp_err" in
        124:*)
            fail tcp "no answer from $HOST:$PORT in ${TCP_TIMEOUT_S}s (silently dropped)"
            fix "the board is off, on another subnet, or behind a firewall that drops instead of refusing" \
                "check this machine's route:  ip route get $HOST" \
                "check the board's address on its own screen:  ip -4 addr" ;;
        *"Connection refused"*)
            fail tcp "$HOST is up but refuses port $PORT -- nothing is listening"
            fix "on the board:  sudo apt-get install -y openssh-server && sudo systemctl enable --now ssh" \
                "or the sshd there listens on another port: re-run with --port N" ;;
        *"No route to host"*|*"Network is unreachable"*)
            fail tcp "no route to $HOST -- this machine cannot reach that network"
            fix "ip route get $HOST      # what this machine would do with that address" \
                "put both on one subnet, or add the route/VPN that reaches the cell network" ;;
        *"Name or service not known"*|*"Temporary failure in name resolution"*|*"nodename nor servname"*)
            fail tcp "cannot resolve '$HOST'"
            fix "use the board's IP address, or add it to /etc/hosts" \
                "avahi resolves nano.local only on the same L2 segment" ;;
        *)
            fail tcp "$HOST:$PORT unreachable: ${tcp_err:-unknown error}" ;;
    esac
fi

# --- 2. ssh auth ----------------------------------------------------------
# Everything below needs a shell on the board; without one there is nothing
# further to say, so stop here rather than printing eight unknowns.
finish() {
    section summary
    printf '  target   %s:%s\n' "$TARGET" "$PORT"
    local s
    if [ "${#SUMMARY[@]}" -gt 0 ]; then
        for s in "${SUMMARY[@]}"; do printf '  %s\n' "$s"; done
    fi
    printf '  result   %d passed, %d warning(s), %d failure(s)\n' "$PASSES" "$WARNS" "$FAILS"
    if [ "$FAILS" -gt 0 ]; then
        printf '\n  Not ready. Fix the FAIL lines above and re-run; this script changed nothing.\n'
        exit 1
    fi
    # provision.sh runs this script as its own first step, so the two agree on
    # what "ready" means; when it is not in the checkout, name the runbook.
    local next="deploy/board/README.md step 2 -- build the wheelhouse, then copy the repo to $TARGET:$INSTALL_PREFIX"
    if [ -x "$REPO_ROOT/deploy/board/provision.sh" ]; then
        next="deploy/board/provision.sh $TARGET --port $PORT${IDENTITY:+ --identity $IDENTITY} --dry-run"
    fi
    if [ "$WARNS" -gt 0 ]; then
        printf '\n  Deployable, with the warnings above. Next:\n    %s\n' "$next"
        exit 2
    fi
    printf '\n  Ready. Next:\n    %s\n' "$next"
    exit 0
}
[ "$FAILS" -eq 0 ] || finish

if ssh "${SSH_OPTS[@]}" "$TARGET" true 2>"$SSH_ERR"; then
    pass ssh "key auth works, non-interactively"
else
    auth_err="$(tr '\n' ' ' <"$SSH_ERR")"
    case "$auth_err" in
        *"Permission denied"*)
            fail ssh "authentication refused for $TARGET"
            [ -f "${IDENTITY:-$DEFAULT_KEY}" ] || \
                fix "ssh-keygen -t ed25519 -f ${IDENTITY:-$DEFAULT_KEY} -N '' -C pose-nano"
            fix "ssh-copy-id -i ${IDENTITY:-$DEFAULT_KEY}.pub -p $PORT $TARGET" \
                "(that asks for the board's password once; after it, this script needs none)" ;;
        *"Host key verification failed"*|*"REMOTE HOST IDENTIFICATION HAS CHANGED"*)
            fail ssh "the board's host key is not the one this machine remembers"
            fix "a reflashed board gets a new key -- that is expected after a re-image:" \
                "ssh-keygen -R '[$HOST]:$PORT' && ssh-keygen -R '$HOST'" ;;
        *"Connection closed"*|*"kex_exchange_identification"*)
            fail ssh "sshd closed the connection before authenticating"
            fix "usually fail2ban/MaxStartups after earlier attempts, or an sshd that is still starting" \
                "wait a minute, then:  ssh -v -p $PORT $TARGET" ;;
        *)
            fail ssh "${auth_err:-ssh failed with no message}" ;;
    esac
    finish
fi

# --- the board, in one round trip ----------------------------------------
# One connection, one payload, KEY<TAB>VALUE back. Nine separate ssh calls
# would each pay a full handshake, and on a board that is the difference
# between a report and a wait. Nothing here writes; every command is guarded
# so a missing file leaves an empty value instead of aborting the probe.
read -r -d '' PROBE <<'PROBE_EOF' || true
say() { printf '%s\t%s\n' "$1" "${2-}"; }

# On Linux /proc/device-tree is a symlink to the sysfs copy; read whichever
# exists. Device-tree strings are NUL-terminated, hence tr.
model=""
for f in /proc/device-tree/model /sys/firmware/devicetree/base/model; do
    [ -r "$f" ] || continue
    model="$(tr -d '\000' <"$f" 2>/dev/null)"
    [ -n "$model" ] && break
done
say model "$model"
say l4t "$(head -n1 /etc/nv_tegra_release 2>/dev/null)"
say arch "$(uname -m)"
say kernel "$(uname -r)"
say os "$( . /etc/os-release 2>/dev/null; printf '%s' "${PRETTY_NAME:-unknown}" )"
say glibc "$(ldd --version 2>/dev/null | head -n1 | grep -o '[0-9]\+\.[0-9]\+$')"

say docker "$(command -v docker >/dev/null 2>&1 && docker --version 2>/dev/null | head -n1)"
say dockergrp "$(id -nG 2>/dev/null | tr ' ' '\n' | grep -qx docker && echo yes || echo no)"
say dockerimg "$(docker images pose-est --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -n1)"
say py38 "$(command -v python3.8 2>/dev/null)"
say py38ver "$(python3.8 -V 2>&1 | head -n1)"
say py3ver "$(python3 -V 2>&1 | head -n1)"
# Ubuntu splits venv out of the interpreter; without it `python3.8 -m venv`
# dies at the end of a long install instead of at the start.
if command -v python3.8 >/dev/null 2>&1 && python3.8 -c 'import ensurepip' 2>/dev/null; then
    say py38venv yes
else
    say py38venv no
fi
# The wheels link against these at import time; missing ones surface as
# ImportError: libGL.so.1 after everything else has already succeeded.
if command -v ldconfig >/dev/null 2>&1; then
    libs=""
    for so in libGL.so.1 libglib-2.0.so.0 libgomp.so.1; do
        ldconfig -p 2>/dev/null | grep -q "$so" || libs="$libs $so"
    done
    say libs "$libs"
else
    say libs unknown
fi

say mem "$(awk '/^MemTotal/{t=$2}/^MemAvailable/{a=$2}/^SwapTotal/{s=$2}
                END{printf "%d %d %d", t/1024, a/1024, s/1024}' /proc/meminfo 2>/dev/null)"
p="__PREFIX__"
while [ ! -d "$p" ] && [ "$p" != "/" ]; do p="$(dirname "$p")"; done
say diskprobe "$p"
say disk "$(df -Pm "$p" 2>/dev/null | awk 'NR==2{print $4, $6}')"
say prefix "$( [ -e "__PREFIX__" ] && echo present || echo absent )"

say target "$(systemctl get-default 2>/dev/null)"
say gui "$(pgrep -c -x 'Xorg|Xwayland|gnome-shell|lightdm|gdm3' 2>/dev/null || true)"
say nvpmodel "$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
t=""
for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    t="$t$(cat "$z/type" 2>/dev/null)=$(( $(cat "$z/temp" 2>/dev/null || echo 0) / 1000 ))C "
done
say thermal "$t"
say epoch "$(date +%s)"
say utc "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Informational only: this deployment is designed to install from a
# wheelhouse and run with no network at all.
inet=no
if command -v timeout >/dev/null 2>&1 && timeout 3 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then
    inet=yes
fi
say inet "$inet"
say dns "$(getent hosts pypi.org >/dev/null 2>&1 && echo yes || echo no)"
PROBE_EOF
PROBE="${PROBE//__PREFIX__/$INSTALL_PREFIX}"

# Bracket the probe with local clock reads so the skew is measured against
# the midpoint of the round trip rather than against one end of it.
t_before="$(date +%s)"
probe_out=""
if ! probe_out="$(printf '%s\n' "$PROBE" | ssh "${SSH_OPTS[@]}" "$TARGET" bash -s 2>"$SSH_ERR")"; then
    fail probe "the board accepted the connection but the probe did not run"
    fix "$(head -n2 "$SSH_ERR" 2>/dev/null)" \
        "the probe needs bash on the board; JetPack has it, a stripped image may not"
    finish
fi
t_after="$(date +%s)"

declare -A R=()
while IFS=$'\t' read -r k v; do
    if [ -n "$k" ]; then R["$k"]="$v"; fi
done <<<"$probe_out"

get() { printf '%s' "${R[$1]:-}"; }

# --- 3. identity ----------------------------------------------------------
section "the board"
model="$(get model)"; l4t="$(get l4t)"
# L4T release -> JetPack, for the versions this runbook has anything to say
# about. The mapping is NVIDIA's; anything else is reported unmapped rather
# than guessed at.
jetpack=""
case "$l4t" in
    *"R32"*"REVISION: 6.1"*) jetpack="JetPack 4.6" ;;
    *"R32"*"REVISION: 7.1"*) jetpack="JetPack 4.6.1" ;;
    *"R32"*"REVISION: 7.2"*) jetpack="JetPack 4.6.2" ;;
    *"R32"*"REVISION: 7.3"*) jetpack="JetPack 4.6.3" ;;
    *"R32"*"REVISION: 7.4"*) jetpack="JetPack 4.6.4" ;;
    *"R35"*)                 jetpack="JetPack 5.x" ;;
    *"R36"*)                 jetpack="JetPack 6.x" ;;
esac
l4t_short="$(printf '%s' "$l4t" | sed -n 's/.*\(R[0-9]\+\).*REVISION: \([0-9.]\+\).*/\1.\2/p')"

if [ -z "$model" ] && [ -z "$l4t" ]; then
    warn identity "no device-tree model and no /etc/nv_tegra_release -- this does not look like a Jetson"
    fix "the scripts only need aarch64 + Python 3.8, so this is a warning, not a stop" \
        "but nvpmodel, jetson_clocks and the memory budget in the runbook assume a Jetson"
    remember "board    unidentified (no device-tree model)"
else
    desc="${model:-unknown model}${l4t_short:+ -- L4T $l4t_short}${jetpack:+ ($jetpack)}"
    case "$model" in
        *"Orin"*)
            warn identity "$desc -- an Orin, not the Nano this runbook was measured on"
            fix "the install works here (aarch64, Python 3.8+), but every memory and latency" \
                "number in deploy/board/README.md belongs to the Nano 4 GB profile" ;;
        *"Nano"*"2GB"*|*"Nano 2GB"*)
            warn identity "$desc"
            fix "this runbook is written for the 4 GB Nano; 2 GB will not hold the 1.6 GB working set" ;;
        *"Nano"*)
            pass identity "$desc" ;;
        *)
            warn identity "$desc -- not a Jetson Nano"
            fix "the install works on Orin too, but every memory and latency number in" \
                "deploy/board/README.md was measured for the Nano 4 GB profile" ;;
    esac
    remember "board    $desc"
fi
[ -n "$l4t" ] || info l4t "no /etc/nv_tegra_release -- JetPack version unknown"

# --- 4. architecture and OS ----------------------------------------------
arch="$(get arch)"; os="$(get os)"; glibc="$(get glibc)"
if [ "$arch" = "aarch64" ]; then
    pass arch "aarch64, kernel $(get kernel)"
else
    fail arch "uname -m says '${arch:-unknown}', not aarch64"
    fix "every wheel in requirements-jetson-nano.lock.txt is an aarch64 build" \
        "this target is not a Jetson board -- check the address"
fi
remember "os       ${os:-unknown}, ${arch:-unknown}, kernel $(get kernel)"
if [ -n "$glibc" ]; then
    # sort -V is the only portable way to compare 2.27 against 2.9 correctly.
    if [ "$(printf '%s\n%s\n' "$MIN_GLIBC" "$glibc" | sort -V | head -n1)" = "$MIN_GLIBC" ]; then
        pass os "$os (glibc $glibc)"
    else
        fail os "$os has glibc $glibc; the stack needs $MIN_GLIBC or newer"
        fix "open3d 0.18.0's only aarch64 wheel is manylinux_2_27 and there is no older" \
            "release with an aarch64 wheel -- flash JetPack 4.6.x (Ubuntu 18.04) instead"
    fi
else
    info os "${os:-unknown} (glibc version not reported)"
fi

# --- 5. runtime: a Python 3.8 venv, or the container that carries one ------
# The board can run either way. Docker is the path with fewer moving parts --
# the image already holds the exact stack the lock file pins -- so a board
# with docker is ready even when it has no python3.8 at all, and demanding
# one there would be a false failure.
py38="$(get py38)"; py38ver="$(get py38ver)"; py3ver="$(get py3ver)"
docker_ver="$(get docker)"; docker_grp="$(get dockergrp)"; docker_img="$(get dockerimg)"

if [ "$MODE" = "docker" ] || { [ "$MODE" = "auto" ] && [ -z "$py38" ] && [ -n "$docker_ver" ]; }; then
    RUNTIME=docker
    if [ -n "$docker_ver" ]; then
        pass docker "$docker_ver"
        remember "runtime  docker -- $docker_ver"
    else
        fail docker "no docker on the board, and --mode docker was asked for"
        fix "sudo apt-get install -y docker.io && sudo usermod -aG docker \$USER" \
            "log out and back in for the group to take effect"
        remember "runtime  docker missing"
    fi
    if [ "$docker_grp" = yes ]; then
        pass dockergrp "the login user is in the docker group -- no sudo needed"
    else
        warn dockergrp "the login user is not in the docker group; every docker call will need sudo"
        fix "sudo usermod -aG docker \$USER   # then log out and back in"
    fi
    if [ -n "$docker_img" ]; then
        pass image "$docker_img is already loaded"
    else
        info image "pose-est:nano is not loaded yet -- carry it over with:"
        fix "docker save pose-est:nano | gzip -1 | ssh $TARGET 'docker load'"
    fi
    if [ -n "$py38" ]; then
        info python "$py38 is present too, so the venv path is also open"
    else
        info python "no python3.8 -- not needed in docker mode; the image carries CPython 3.8"
    fi
elif [ -n "$py38" ]; then
    RUNTIME=venv
    pass python "$py38 ($py38ver)"
    remember "python   $py38 ($py38ver)"
else
    RUNTIME=venv
    fail python "no python3.8 on the board (and no docker to fall back to)"
    fix "sudo apt-get install -y software-properties-common" \
        "sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update" \
        "sudo apt-get install -y python3.8 python3.8-venv python3.8-distutils libgl1 libglib2.0-0 libgomp1" \
        "(that step needs the board online once; after it the install is air-gapped)"
    remember "python   missing -- deadsnakes step not done"
fi
case "$py3ver" in
    *" 3.6"*) info python3 "$py3ver is the stock interpreter -- ultralytics 8.4 needs 3.8, so nothing installs into it" ;;
    "")       info python3 "no python3 in PATH" ;;
    *)        info python3 "$py3ver" ;;
esac
if [ -n "$py38" ] && [ "$RUNTIME" = venv ]; then
    if [ "$(get py38venv)" = "yes" ]; then
        pass venv "python3.8 -m venv will work (ensurepip present)"
    else
        fail venv "python3.8 has no ensurepip -- 'python3.8 -m venv .venv' will fail"
        fix "sudo apt-get install -y python3.8-venv python3.8-distutils"
    fi
fi
libs="$(get libs)"
if [ "$libs" = "unknown" ]; then
    info libs "no ldconfig here -- could not check libGL, libglib and libgomp"
elif [ -n "$libs" ]; then
    warn libs "not in the linker cache:$libs"
    fix "the opencv and open3d wheels link against these at import time:" \
        "sudo apt-get install -y libgl1 libglib2.0-0 libgomp1"
else
    pass libs "libGL, libglib and libgomp are installed"
fi

# --- 6. resources ---------------------------------------------------------
read -r mem_total mem_avail swap_mb <<<"$(get mem)" || true
mem_total="${mem_total:-0}"; mem_avail="${mem_avail:-0}"; swap_mb="${swap_mb:-0}"
info memory "${mem_total} MB total, ${mem_avail} MB available"
remember "memory   ${mem_total} MB total, ${mem_avail} MB available, ${swap_mb} MB swap"
if [ "$swap_mb" -ge "$MIN_SWAP_MB" ]; then
    pass swap "${swap_mb} MB"
else
    warn swap "${swap_mb} MB, under the ${MIN_SWAP_MB} MB the runbook assumes"
    fix "zram alone is enough for a headless board; with the desktop up, add file swap:" \
        "sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile" \
        "echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
fi

read -r disk_mb disk_mount <<<"$(get disk)" || true
disk_mb="${disk_mb:-0}"; disk_mount="${disk_mount:-unknown}"
remember "disk     ${disk_mb} MB free on ${disk_mount} (the install needs ~1.6 GB)"
if [ "$disk_mb" -lt "$MIN_FREE_MB" ]; then
    fail disk "${disk_mb} MB free on ${disk_mount}; the install needs ${MIN_FREE_MB} MB"
    fix "1262 MB venv + ~340 MB of wheels staged + 7 MB of code, weights and CAD model" \
        "reclaim it:  sudo apt-get clean && sudo journalctl --vacuum-size=100M" \
        "or move $INSTALL_PREFIX onto a larger card and re-run"
elif [ "$disk_mb" -lt "$COMFORT_FREE_MB" ]; then
    warn disk "${disk_mb} MB free on ${disk_mount} -- enough to install, little to spare"
    fix "the journal, a second release and pip's temporary files all land on this filesystem"
else
    pass disk "${disk_mb} MB free on ${disk_mount}"
fi
if [ "$(get prefix)" = "present" ]; then
    info prefix "$INSTALL_PREFIX already exists -- a deploy will be an upgrade, not a first install"
else
    info prefix "$INSTALL_PREFIX does not exist yet (expected on a fresh board)"
fi

gui="$(get gui)"; deftarget="$(get target)"
if [ "${gui:-0}" -gt 0 ] || [ "$deftarget" = "graphical.target" ]; then
    warn desktop "a desktop session is up (${gui:-0} process(es), default target ${deftarget:-unknown})"
    fix "it costs roughly 500 MB of the 4 GB the pipeline shares with the GPU:" \
        "sudo systemctl set-default multi-user.target && sudo reboot"
else
    pass desktop "headless (default target ${deftarget:-unknown})"
fi

# --- 7. power and thermals (informational) --------------------------------
nvp="$(get nvpmodel)"
if [ -n "$nvp" ]; then
    info power "$nvp"
else
    info power "nvpmodel not available here -- on the board, run 'sudo nvpmodel -m 0 && sudo jetson_clocks' before timing anything"
fi
therm="$(get thermal)"
[ -n "$therm" ] && info thermal "$therm" || info thermal "no readable thermal zones"

# --- 8. clock -------------------------------------------------------------
board_epoch="$(get epoch)"
if [ -n "$board_epoch" ]; then
    host_mid=$(( (t_before + t_after) / 2 ))
    skew=$(( board_epoch - host_mid ))
    if [ "$skew" -lt 0 ]; then skew=$(( -skew )); fi
    if [ "$skew" -le "$CLOCK_SKEW_MAX_S" ]; then
        pass clock "$(get utc), ${skew}s from this machine"
    else
        warn clock "$(get utc) -- ${skew}s away from this machine"
        fix "the Nano has no RTC battery, so off the network it restarts at the last shutdown time" \
            "sudo date -s '$(date -u '+%Y-%m-%d %H:%M:%S')' UTC   # or NTP, or: sudo apt-get install fake-hwclock" \
            "bench records are matched by timestamp, and the journal orders by it"
    fi
else
    warn clock "the board did not report its time"
fi

# --- 9. network posture (informational) -----------------------------------
if [ "$(get inet)" = "yes" ]; then
    info network "the board reaches the internet (DNS to pypi.org: $(get dns))"
    info network "not required: the install comes from a wheelhouse and the service never calls out"
else
    info network "no route out from the board -- which is the design; the wheelhouse install and"
    info network "the service both work air-gapped. Only the deadsnakes step above needs a network."
fi

finish
