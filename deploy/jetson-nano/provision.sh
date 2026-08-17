#!/usr/bin/env bash
# Take a reachable Jetson Nano from a bare JetPack 4.6 install to a pose
# service that has returned a pose. This is deploy/jetson-nano/README.md's
# bring-up, in its order, as one command:
#
#   deploy/jetson-nano/provision.sh nano@192.168.1.50 --dry-run   # run this first
#   deploy/jetson-nano/provision.sh nano@192.168.1.50
#   deploy/jetson-nano/provision.sh nano@192.168.1.50 --wheelhouse wheelhouse-nano
#
#   1 preflight   deploy/jetson-nano/preflight.sh -- the board's own checks
#                 (exit 0 pass, 2 warnings, anything else stops this script)
#   2 payload     the repo subset the board needs, and one commissioning frame
#   3 wheelhouse  the offline wheels, or the lock file over the network
#   4 venv        python3.8 venv + install, proved by importing on the board
#   5 config      config.nano.json with the paths rewritten, validated there
#   6 service     the pose user, the unit, daemon-reload, enable, restart
#   7 health      /healthz (loaded) then /readyz (loaded and warmed)
#   8 first pick  one estimate against the commissioning frame -- the accept test
#   9 summary     where it is, how to watch it, what to measure next
#
# Every step announces itself, is safe to re-run, and fails with the command
# that fixes it rather than working around the problem. What this script
# never does: delete anything outside --prefix, copy train/ test/ seg_data/
# or the 56 MB and 45 MB desktop weights (the board runs part-seg-nano.pt),
# or change the board's apt state -- a missing python3.8 is reported with the
# apt line from README section 1, not installed behind your back.
#
# --force continues past a failing preflight. --adopt, separately, installs
# into a prefix this script did not create. Neither deletes anything. The only thing
# this script ever deletes is <prefix>/.venv, and only when --rebuild-venv
# says so. Removing an installation is uninstall.sh's job.
#
# --prefix has to be an absolute path of at least two components and not a
# system directory: step 2 and step 6 both chown the whole prefix, once to the
# login user and once to the service user, and a chown -R over /usr costs the
# board its setuid sudo.
#
# It needs sudo on the board without a password, or a root login: each step
# is a script piped into a non-interactive shell, and sudo has no terminal to
# ask on. It checks that once, before anything is copied, and names both
# fixes rather than dying at the first sudo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# -- what the board gets -------------------------------------------------
#
# An allow-list, not an exclude-list: a new dataset directory appearing in
# the repo can never leak onto a customer board by being forgotten here.
PAYLOAD_DIRS=(src scripts deploy model)
PAYLOAD_FILES=(
    score.py visualize.py run_all.sh setup.sh requirements.txt
    weights/part-seg-nano.pt      # the only weight config.nano.json names
)

PREFIX="/opt/pose-estimation"     # README sections 2 and 5; the shipped unit file's path
SERVICE_USER="pose"               # README section 5
UNIT_NAME="pose-service.service"
CONFIG_REL="deploy/jetson-nano/config.nano.json"
LOCK_REL="deploy/jetson-nano/requirements-jetson-nano.lock.txt"
PREFLIGHT="$ROOT/deploy/jetson-nano/preflight.sh"
WHEELHOUSE_DIRNAME="wheelhouse-nano"   # the name README section 2 uses on the board
MARKER_NAME=".pose-estimation-install" # uninstall.sh refuses a prefix without it
MARKER_ID="pose-estimation-provision-v1"

# test/000001 is the frame the whole repo is calibrated against: pick mode
# returns one pose at score ~0.85 for it on every platform measured so far
# (results/bench/*.json), so a board that answers differently is telling you
# something. One frame, 0.9 MB -- the split it came from is never copied.
SCENE_SRC="$ROOT/test/000001"

# The service binds its socket before it imports the estimator, so the port
# opens within a second or two of the unit starting; 60 s is a stuck process,
# not a slow one (README section 6).
SOCKET_TIMEOUT=60
# Then it loads the CAD cloud and the segmenter and runs one warmup pick. A
# Jetson Nano 4 GB took 23.6 s for that load at MAXN with the files in the
# page cache (results/bench/board_nano640.json); the unit itself budgets
# TimeoutStartSec=300. 600 s is twice the unit's budget and 25x the measured
# load: the margin is for a first start reading torch off a cold SD card, not
# for the load itself, and it is still narrow enough that a genuinely stuck
# import is reported rather than waited on forever.
READY_TIMEOUT=600
# The client defaults to a 120 s watchdog. A pick on the board measured
# 2.6-2.7 s (same record); 300 s keeps a first post-warmup pick over a cold
# cache from reading as a hang.
PICK_TIMEOUT=300

WHEELHOUSE=""
IDENTITY=""
SSH_PORT=""
NO_SERVICE=0
UNIT_STATE=""
DRY_RUN=0
FORCE=0
ADOPT=0
REBUILD_VENV=0
LIST_ONLY=0
SCENE_GIVEN=0
TARGET=""

usage() {
    cat <<'USAGE'
usage: deploy/jetson-nano/provision.sh <user@host> [options]

  --prefix DIR       install prefix on the board (default /opt/pose-estimation)
  --wheelhouse DIR   local directory of aarch64 wheels, for an air-gapped board
  --identity KEY     ssh private key
  --port N           ssh port
  --scene DIR        commissioning frame to carry (default test/000001)
  --ready-timeout S  seconds to wait for /readyz (default 600)
  --no-service       install everything, leave systemd alone
  --dry-run          print every remote command, run none of them
  --force            continue past a failing preflight
  --adopt            install into an existing prefix that carries no install
                     marker (it will be overwritten and chowned)
  --rebuild-venv     delete <prefix>/.venv and install into a new one
  --list-payload     print the exact file list that would be copied, and stop
  -h, --help         this text
USAGE
}

die() {
    printf '\nerror: %s\n' "$1" >&2
    [ $# -gt 1 ] && printf '%s\n' "$2" >&2
    exit 1
}

# -- arguments -----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --wheelhouse) WHEELHOUSE="${2:?--wheelhouse needs a directory}"; shift 2 ;;
        --identity) IDENTITY="${2:?--identity needs a key file}"; shift 2 ;;
        --port) SSH_PORT="${2:?--port needs a number}"; shift 2 ;;
        --scene) SCENE_SRC="${2:?--scene needs a directory}"; SCENE_GIVEN=1; shift 2 ;;
        --ready-timeout) READY_TIMEOUT="${2:?--ready-timeout needs seconds}"; shift 2 ;;
        --no-service) NO_SERVICE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list-payload) LIST_ONLY=1; shift ;;
        --force) FORCE=1; shift ;;
        --adopt) ADOPT=1; shift ;;
        --rebuild-venv) REBUILD_VENV=1; shift ;;
        -*) usage >&2; die "unknown option: $1" ;;
        *)  [ -z "$TARGET" ] || { usage >&2; die "one target only, got '$TARGET' and '$1'"; }
            TARGET="$1"; shift ;;
    esac
done
[ -n "$TARGET" ] || [ "$LIST_ONLY" = 1 ] \
    || { usage >&2; die "no target: give <user@host>"; }

# Both end up inside a remote command line; a non-number there fails on the
# board, minutes in, as a traceback from a helper.
case "${SSH_PORT:-22}" in ''|*[!0-9]*) die "--port must be a number, got '$SSH_PORT'" ;; esac
case "$READY_TIMEOUT" in
    ''|*[!0-9]*) die "--ready-timeout must be whole seconds, got '$READY_TIMEOUT'" ;;
esac

# -- the prefix has to survive this before the board is contacted ---------
#
# Everything this script writes lives under the prefix, step 2 chowns the
# whole tree to the login user and step 6 chowns it to '$SERVICE_USER'. A
# prefix that names a system directory is therefore a recursive chown of that
# directory, and `chown -R` clears the setuid bit on /usr/bin/sudo -- an
# SSH-only board that loses sudo is a board that needs a re-flash. The same
# set uninstall.sh refuses, so a prefix accepted here is one that can still be
# removed afterwards.
DENY="/ /bin /boot /dev /etc /home /lib /media /mnt /opt /proc /root /run /sbin /srv /sys /tmp /usr /var
      /etc/systemd /usr/bin /usr/lib /usr/local /usr/share /var/lib /var/log /var/run"

case "$PREFIX" in
    /*) ;;
    *) die "--prefix must be absolute, got '$PREFIX'" ;;
esac
# Trailing slash first: '/' would otherwise pass the test above and become an
# empty string, which reaches the board as `install -d ''`.
PREFIX="${PREFIX%/}"
# "/opt" is one component, "/opt/pose-estimation" is two. An install prefix
# has to be a directory somebody made for it, not a place the distribution
# owns.
[ -n "$PREFIX" ] \
    && [ "$(printf '%s' "${PREFIX#/}" | awk -F/ '{ print NF }')" -ge 2 ] \
    || die "refusing '${PREFIX:-/}' as an install prefix: fewer than two path components" \
"Everything installed goes under the prefix and step 6 hands the whole tree to
the '$SERVICE_USER' user, so the prefix must be its own directory:
    --prefix /opt/pose-estimation"
for denied in $DENY; do
    [ "$PREFIX" = "$denied" ] && die "refusing '$PREFIX' as an install prefix: it is a system directory" \
"Whatever is there, it is not only ours, and step 6 would chown all of it to
the '$SERVICE_USER' user. Install into a directory of its own:
    --prefix /opt/pose-estimation"
done

if [ -n "$WHEELHOUSE" ]; then
    [ -d "$WHEELHOUSE" ] || die "no such wheelhouse: $WHEELHOUSE" \
        "Build it inside the aarch64 image (the lock file's header has the command)."
    WHEELHOUSE="$(cd "$WHEELHOUSE" && pwd)"
fi

# -- ssh -----------------------------------------------------------------
#
# One authenticated connection, reused by every step and by rsync. Without
# multiplexing a key with a passphrase, or a password login, is nine prompts.
SSH_CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pose-provision.XXXXXX")"
LAST_OUT="$SSH_CONTROL_DIR/last-output"
: > "$LAST_OUT"
SSH_OPTS=(-o ConnectTimeout=10
          -o "ControlMaster=auto"
          -o "ControlPath=$SSH_CONTROL_DIR/ssh-%r@%h:%p"
          -o "ControlPersist=120s")
[ -n "$IDENTITY" ] && SSH_OPTS+=(-i "$IDENTITY")
[ -n "$SSH_PORT" ] && SSH_OPTS+=(-p "$SSH_PORT")
# rsync needs the same options as one string, and it splits that string on
# whitespace with no quoting of any kind -- a backslash stays a backslash and
# becomes part of the filename it tries to exec. So an option containing a
# space (an --identity under a path with one, or a TMPDIR with one) cannot go
# through rsync at all; record that and use the tar transport instead, which
# passes the options as an argv array.
SSH_E="ssh"
SSH_E_USABLE=1
for opt in "${SSH_OPTS[@]}"; do
    case "$opt" in *[[:space:]]*) SSH_E_USABLE=0 ;; esac
    SSH_E="$SSH_E $opt"
done

# Set once the payload is on the board: from then on a failure leaves an
# install behind, and the operator has to be told rather than left to infer it
# from the last error line.
LANDED=0
# Set once step 9 has written the marker: after that the install is complete
# and the closing block below is the one that judges it, so the trap must not
# also call the board part-installed.
PROVISIONED=0

cleanup() {
    local rc=$?
    if [ "$DRY_RUN" != 1 ]; then
        ssh "${SSH_OPTS[@]}" -O exit "$TARGET" >/dev/null 2>&1 || true
    fi
    rm -rf "$SSH_CONTROL_DIR"
    if [ "$rc" != 0 ] && [ "$LANDED" = 1 ] && [ "$PROVISIONED" = 0 ]; then
        printf '\n== stopped in step %d/9. %s on %s is part-installed.\n' \
            "$STEP" "$PREFIX" "$TARGET" >&2
        printf '   The payload and %s are there; whether the service runs is\n' \
            "$MARKER_NAME" >&2
        printf '   whatever the error above says. Fix that and re-run this same\n' >&2
        printf '   command -- every step overwrites rather than duplicates. Or\n' >&2
        printf '   take it off the board:\n' >&2
        printf '     deploy/jetson-nano/uninstall.sh %s --prefix %s --yes\n' \
            "$TARGET" "$PREFIX" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT

qq() { printf '%q' "$1"; }

STEP=0
step() { STEP=$((STEP + 1)); printf '\n== %d/9  %s\n' "$STEP" "$1"; }
note() { printf '   %s\n' "$1"; }

print_remote() {
    printf '\n   --- %s: would run on %s ---\n' "$1" "$TARGET"
    printf '%s\n' "$2" | sed 's/^/   | /'
    printf '   --- sent as:'
    printf ' %q' ssh "${SSH_OPTS[@]}" "$TARGET" bash -s
    printf ' <<script ---\n'
}

# Every remote step is a whole script, rendered here and sent over one ssh
# call, so --dry-run prints exactly what would run instead of a description of
# it. Output is streamed as it arrives (a Nano is slow enough to want that)
# and teed, so a step can hand a value to the summary.
remote() {
    local label="$1" script="$2"
    : > "$LAST_OUT"
    if [ "$DRY_RUN" = 1 ]; then
        print_remote "$label" "$script"
        return 0
    fi
    printf '%s' "$script" | ssh "${SSH_OPTS[@]}" "$TARGET" bash -s | tee "$LAST_OUT"
}

# KEY=value lines a remote step printed for the summary to pick up.
from_last() { sed -n "s/^$1=//p" "$LAST_OUT" | tail -1; }

remote_preamble() {
    cat <<EOF
set -euo pipefail
PREFIX=$(qq "$PREFIX")
SERVICE_USER=$(qq "$SERVICE_USER")
UNIT=$(qq "$UNIT_NAME")
VENV="\$PREFIX/.venv"
PY="\$VENV/bin/python"
CONFIG="\$PREFIX/$CONFIG_REL"
# Empty when the login is already root; deliberately unquoted at the call
# sites so that an empty value disappears instead of becoming argv[0].
SUDO=$(qq "$SUDO")

# The guard the driver applied, re-applied on the board where the chown and
# the venv rm actually happen, so it does not depend on which script called.
guard_prefix() {
    case "\$PREFIX" in
        /*) ;;
        *) echo "error: prefix '\$PREFIX' is not absolute" >&2; exit 1 ;;
    esac
    if [ "\$(printf '%s' "\${PREFIX#/}" | awk -F/ '{ print NF }')" -lt 2 ]; then
        echo "error: refusing to touch '\$PREFIX': fewer than two path components" >&2
        exit 1
    fi
    for denied in $(printf '%s' "$DENY" | tr '\n' ' '); do
        if [ "\$PREFIX" = "\$denied" ]; then
            echo "error: refusing to touch '\$PREFIX': system directory" >&2
            exit 1
        fi
    done
}
EOF
}

# -- payload -------------------------------------------------------------
payload_list() {
    (
        cd "$ROOT"
        for d in "${PAYLOAD_DIRS[@]}"; do
            # __pycache__ is the host's x86 bytecode: wrong magic for the
            # board's 3.8 and pure noise on the wire.
            find "$d" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print
        done
        printf '%s\n' "${PAYLOAD_FILES[@]}"
    ) | LC_ALL=C sort
}

payload_bytes() {
    payload_list | (cd "$ROOT" && xargs -d '\n' -r stat -c %s) \
        | awk '{ s += $1 } END { print s + 0 }'
}

human_bytes() {
    awk -v b="$1" 'BEGIN {
        split("B KB MB GB", u, " "); i = 1
        while (b >= 1024 && i < 4) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s\n" : "%.1f %s\n"), b, u[i]
    }'
}

# The lock file header carries the only correct recipe (native resolution
# inside the aarch64 image). Print that rather than a copy that can drift.
wheelhouse_recipe() {
    sed -n '/^#   docker run/,/^#$/p' "$ROOT/$LOCK_REL" | sed -e 's/^# \{0,1\}//' -e '/^$/d'
}

if [ "$LIST_ONLY" = 1 ]; then
    payload_list
    printf '# %s files, %s\n' "$(payload_list | wc -l)" \
        "$(payload_bytes | { read -r n; human_bytes "$n"; })"
    exit 0
fi

# Rewritten whole rather than patched, so running it twice cannot leave two
# of anything. The date comes from this host: the Nano has no RTC battery and
# its clock restarts at the last shutdown (README section 0).
marker_script() {
    cat <<EOF
$(remote_preamble)
\$SUDO tee "\$PREFIX/$MARKER_NAME" >/dev/null <<'MARKER'
{
  "marker": "$MARKER_ID",
  "prefix": "$PREFIX",
  "unit": "$UNIT_NAME",
  "service_user": "$SERVICE_USER",
  "config_digest": "${1:-unknown, provisioning did not get that far}",
  "installed_by": "deploy/jetson-nano/provision.sh",
  "installed_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "installed_from": "$(hostname):$ROOT",
  "source_commit": "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}
MARKER
\$SUDO chmod 0644 "\$PREFIX/$MARKER_NAME"
echo "marker \$PREFIX/$MARKER_NAME"
EOF
}

printf '== provisioning %s\n' "$TARGET"
note "prefix     $PREFIX"
note "payload    $(payload_bytes | { read -r n; human_bytes "$n"; }) from $ROOT"
note "wheelhouse ${WHEELHOUSE:-none given (the board installs from the network)}"
if [ "$NO_SERVICE" = 1 ]; then
    note "service    skipped (--no-service)"
else
    note "service    $UNIT_NAME as $SERVICE_USER"
fi
[ "$DRY_RUN" = 1 ] && note "DRY RUN -- nothing below is executed"

for item in "${PAYLOAD_FILES[@]}" "${PAYLOAD_DIRS[@]}"; do
    [ -e "$ROOT/$item" ] || die "payload item missing: $ROOT/$item" \
        "Provision from a complete checkout; weights/part-seg-nano.pt is what config.nano.json names."
done

# -- 1. preflight --------------------------------------------------------
step "preflight"
SUDO="sudo"      # assumed until the check below; --dry-run never finds out
if [ ! -x "$PREFLIGHT" ]; then
    [ "$FORCE" = 1 ] || die "no executable $PREFLIGHT" \
"The board checks live there, not here -- this script does not duplicate them.
Fix: restore the file (git checkout deploy/jetson-nano/preflight.sh), or
     re-run with --force to provision without the gate."
    note "no $PREFLIGHT -- skipped (--force)"
elif [ "$DRY_RUN" = 1 ]; then
    note "would run: $PREFLIGHT $TARGET${IDENTITY:+ --identity $IDENTITY}${SSH_PORT:+ --port $SSH_PORT}"
else
    PREFLIGHT_ARGS=("$TARGET")
    [ -n "$IDENTITY" ] && PREFLIGHT_ARGS+=(--identity "$IDENTITY")
    [ -n "$SSH_PORT" ] && PREFLIGHT_ARGS+=(--port "$SSH_PORT")
    PREFLIGHT_RC=0
    "$PREFLIGHT" "${PREFLIGHT_ARGS[@]}" || PREFLIGHT_RC=$?
    # Its own contract: 0 every check passed, 2 warnings only (deployable
    # with the caveats it printed), anything else a hard failure.
    case "$PREFLIGHT_RC" in
        0) note "preflight passed" ;;
        2) note "preflight passed with warnings -- see above; continuing" ;;
        *) [ "$FORCE" = 1 ] && note "preflight failed -- continuing anyway (--force)" \
               || die "preflight failed on $TARGET (exit $PREFLIGHT_RC)" \
"Fix what it reported, or re-run with --force if you accept the risk.
Nothing has been copied to the board yet." ;;
    esac
fi

# sudo is needed from step 2 (creating the prefix) onwards, so find out now
# rather than half way through.
if [ "$DRY_RUN" != 1 ]; then
    SUDO_MODE="$(printf '%s' 'if [ "$(id -u)" = 0 ]; then echo root
elif sudo -n true 2>/dev/null; then echo sudo
else echo password; fi' | ssh "${SSH_OPTS[@]}" "$TARGET" bash -s)" \
        || die "cannot reach $TARGET over ssh" \
"Check the address, the key and that sshd is running:
    ssh${IDENTITY:+ -i $IDENTITY}${SSH_PORT:+ -p $SSH_PORT} $TARGET true"
    case "$SUDO_MODE" in
        root) SUDO="" ; note "remote login is root" ;;
        sudo) SUDO="sudo"; note "remote sudo works without a password" ;;
        *) die "sudo on $TARGET wants a password, and there is no terminal to ask on" \
"Each step is a script piped into a non-interactive shell. Pick one:
    - provision as root:  $(basename "$0") root@${TARGET#*@} ...
    - or, on the board:   echo \"\$USER ALL=(ALL) NOPASSWD:ALL\" | sudo tee /etc/sudoers.d/90-pose-provision
Nothing has been copied to the board yet." ;;
    esac
fi

# -- 2. payload ----------------------------------------------------------
step "payload"
PAYLOAD_N="$(payload_list | wc -l)"
PAYLOAD_B="$(payload_bytes)"
note "$PAYLOAD_N files, $(human_bytes "$PAYLOAD_B") -> $TARGET:$PREFIX"
note "not copied: train/ test/ seg_data/ seg_runs*/ results/ plans/ .git/ .venv/,"
note "            weights/part-seg.pt (56 MB) and weights/part-seg-synthetic.pt (45 MB)"

remote "prepare the prefix" "$(cat <<EOF
$(remote_preamble)
guard_prefix
if [ ! -d "\$PREFIX" ]; then
    \$SUDO install -d -m 0755 "\$PREFIX"
    echo "created \$PREFIX"
elif [ ! -e "\$PREFIX/$MARKER_NAME" ] && [ -n "\$(ls -A "\$PREFIX" 2>/dev/null)" ]; then
    # An existing directory with contents and no marker was put there by
    # something else -- a typo that happens to name a real directory, or a
    # release someone unpacked by hand. Installing into it means overwriting
    # its files and handing all of them to the '$SERVICE_USER' user, so ask
    # rather than assume. uninstall.sh makes the same demand before it deletes.
    if [ "$ADOPT" = 1 ]; then
        echo "adopting \$PREFIX: it has contents and no $MARKER_NAME (--adopt)"
    else
        echo "error: \$PREFIX already exists, is not empty, and carries no" >&2
        echo "       $MARKER_NAME -- it was not installed by this script." >&2
        echo "       Step 6 would chown all of it to '$SERVICE_USER'." >&2
        echo "It holds:" >&2
        ls -A "\$PREFIX" | head -n 10 | sed 's/^/    /' >&2 || true
        echo "Fix: install somewhere else (--prefix DIR), or re-run with" >&2
        echo "     --adopt to install into that directory anyway." >&2
        exit 1
    fi
fi
# A re-provision finds the prefix owned by the service user (step 6 hands it
# over). Take it back for the transfer so the copy never runs as root, and
# hand it back afterwards. Bounded to the prefix, like everything else here.
\$SUDO chown -R "\$(id -un):\$(id -gn)" "\$PREFIX"
mkdir -p "\$PREFIX/commissioning"
printf 'PREFIX_OWNER=%s\n' "\$(id -un)"
df -Pk "\$PREFIX" | awk 'NR==2 { printf "PREFIX_FREE_MB=%d\n", \$4/1024 }'
EOF
)"

if [ "$DRY_RUN" = 1 ]; then
    printf '\n   --- payload: would run locally ---\n'
    printf '   | # file list (%s files, %s), identical for both transports:\n' \
        "$PAYLOAD_N" "$(human_bytes "$PAYLOAD_B")"
    printf '   | deploy/jetson-nano/provision.sh --list-payload   # the same allow-list\n'
    printf '   | # if both ends have rsync:\n'
    printf '   | payload_list | rsync -a --files-from=- --info=stats1 -e %q %q %q\n' \
        "$SSH_E" "$ROOT/" "$TARGET:$PREFIX/"
    printf '   | # otherwise:\n'
    printf '   | payload_list | tar -czf - -C %q -T - | ssh ... %q\n' \
        "$ROOT" "tar -xzf - -C $PREFIX"
else
    # No --delete: rsync pruning a directory on the board is a deletion, and
    # deletions here are uninstall.sh's job, with its own flag. Re-provisioning
    # overwrites; it does not tidy.
    if [ "$SSH_E_USABLE" = 0 ]; then
        note "transport: tar over ssh (an ssh option contains a space, which"
        note "           rsync's -e cannot carry -- the copy is identical)"
        payload_list | tar -czf - -C "$ROOT" -T - \
            | ssh "${SSH_OPTS[@]}" "$TARGET" "tar -xzf - -C $(qq "$PREFIX")"
    elif command -v rsync >/dev/null 2>&1 \
            && printf 'command -v rsync >/dev/null 2>&1' | ssh "${SSH_OPTS[@]}" "$TARGET" bash -s
    then
        note "transport: rsync"
        payload_list | rsync -a --files-from=- --info=stats1 -e "$SSH_E" \
            "$ROOT/" "$TARGET:$PREFIX/"
    else
        # gzip: the compression happens on this machine and only the
        # decompression on the Nano, which is the cheap half.
        note "transport: tar over ssh (no rsync on one end)"
        payload_list | tar -czf - -C "$ROOT" -T - \
            | ssh "${SSH_OPTS[@]}" "$TARGET" "tar -xzf - -C $(qq "$PREFIX")"
    fi
fi

[ "$DRY_RUN" = 1 ] || LANDED=1

SCENE_ID=""
if [ -d "$SCENE_SRC" ]; then
    SCENE_ID="$(basename "$SCENE_SRC")"
    SCENE_B="$(du -sb "$SCENE_SRC" | cut -f1)"
    note "commissioning frame: $SCENE_SRC ($(human_bytes "$SCENE_B")) -> $PREFIX/commissioning/$SCENE_ID"
    if [ "$DRY_RUN" = 1 ]; then
        printf '   | tar -czf - -C %q %q | ssh ... %q\n' \
            "$(dirname "$SCENE_SRC")" "$SCENE_ID" "tar -xzf - -C $PREFIX/commissioning"
    else
        tar -czf - -C "$(dirname "$SCENE_SRC")" "$SCENE_ID" \
            | ssh "${SSH_OPTS[@]}" "$TARGET" "tar -xzf - -C $(qq "$PREFIX/commissioning")"
    fi
elif [ "$SCENE_GIVEN" = 1 ]; then
    die "no such commissioning frame: $SCENE_SRC" \
"--scene names the directory holding rgb.png, depth.png and camera.json for one
frame. The repo's default is test/000001."
else
    note "no scene at $SCENE_SRC -- step 8 has nothing to pick from (--scene DIR)"
fi

# Written now, not at the end: a run that dies half way must still leave
# uninstall.sh something it recognises, so the board can be cleaned up.
# The date comes from this host -- the Nano has no RTC battery and its clock
# restarts at the last shutdown (README section 0).
PREFIX_FREE_MB="$(from_last PREFIX_FREE_MB)"
# The venv alone is 1.3 GB installed on aarch64 (measured in the pose-est:nano
# image), before the wheelhouse and the payload. Under 2 GB free the install
# will fail somewhere in the middle of pip, which is the worst place for it.
if [ -n "$PREFIX_FREE_MB" ] && [ "$PREFIX_FREE_MB" -lt 2048 ]; then
    note "WARNING: only ${PREFIX_FREE_MB} MB free at $PREFIX; the venv needs ~1.3 GB"
fi

remote "install marker" "$(marker_script "")"

# -- 3. wheelhouse -------------------------------------------------------
step "wheelhouse"
if [ -n "$WHEELHOUSE" ]; then
    WHEEL_N="$(find "$WHEELHOUSE" -maxdepth 1 -name '*.whl' -o -maxdepth 1 -name '*.tar.gz' | wc -l)"
    [ "$WHEEL_N" -gt 0 ] || die "no wheels in $WHEELHOUSE" "$(wheelhouse_recipe)"
    WHEEL_B="$(du -sb "$WHEELHOUSE" | cut -f1)"
    note "$WHEEL_N wheels, $(human_bytes "$WHEEL_B") -> $PREFIX/$WHEELHOUSE_DIRNAME"
    if [ "$DRY_RUN" = 1 ]; then
        printf '   | tar -czf - -C %q . | ssh ... %q\n' \
            "$WHEELHOUSE" "mkdir -p $PREFIX/$WHEELHOUSE_DIRNAME && tar -xzf - -C $PREFIX/$WHEELHOUSE_DIRNAME"
    else
        tar -czf - -C "$WHEELHOUSE" . | ssh "${SSH_OPTS[@]}" "$TARGET" \
            "mkdir -p $(qq "$PREFIX/$WHEELHOUSE_DIRNAME") && tar -xzf - -C $(qq "$PREFIX/$WHEELHOUSE_DIRNAME")"
    fi
else
    note "none given -- the install below uses $PREFIX/$WHEELHOUSE_DIRNAME if a"
    note "previous run left one there, then the network, then it fails with the recipe"
fi

# -- 4. venv -------------------------------------------------------------
step "venv and install"
remote "venv and install" "$(cat <<EOF
$(remote_preamble)
LOCK="\$PREFIX/$LOCK_REL"
WHEELHOUSE="\$PREFIX/$WHEELHOUSE_DIRNAME"

command -v python3.8 >/dev/null 2>&1 || {
    echo "error: no python3.8 on this board." >&2
    echo "The stock 3.6 is too old for ultralytics (README section 1). Fix:" >&2
    echo "    sudo apt-get install -y software-properties-common" >&2
    echo "    sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update" >&2
    echo "    sudo apt-get install -y python3.8 python3.8-venv python3.8-distutils \\\\" >&2
    echo "                            libgl1 libglib2.0-0 libgomp1" >&2
    exit 1
}

# The one thing this script deletes, and only when asked in so many words.
# Bounded to the prefix: a broken venv is rebuilt, nothing else is touched.
if [ "$REBUILD_VENV" = 1 ] && [ -d "\$VENV" ]; then
    echo "removing \$VENV and rebuilding it (--rebuild-venv)"
    rm -rf "\$VENV"
fi
if [ ! -x "\$PY" ]; then
    python3.8 -m venv "\$VENV" || {
        echo "error: python3.8 -m venv failed." >&2
        echo "Fix: sudo apt-get install -y python3.8-venv" >&2
        exit 1
    }
    echo "created \$VENV"
else
    echo "venv already there: \$("\$PY" -V 2>&1)"
fi

# --no-cache-dir throughout: the pip cache is hundreds of MB of wheels the
# board will never reinstall, on an SD card.
# python3.8 -m venv on Ubuntu 18.04 bootstraps pip 9.0.1 (measured in the
# aarch64 image), which predates every wheel tag this stack needs: polars is
# manylinux_2_24_aarch64 (PEP 600, pip 20.3) and torchvision carries the bare
# linux_aarch64 tag. It does not fail with "your pip is old", it fails with
# "no matching distribution", which reads like a missing wheel. Check first.
pip_recent() {
    "\$PY" - <<'PIPCHECK'
import sys
try:
    from pip import __version__ as version
except ImportError:
    sys.exit(1)
sys.exit(0 if tuple(int(p) for p in version.split(".")[:2]) >= (20, 3) else 1)
PIPCHECK
}

if [ -d "\$WHEELHOUSE" ]; then
    if ! pip_recent; then
        if compgen -G "\$WHEELHOUSE/pip-*.whl" >/dev/null; then
            echo "upgrading pip from the wheelhouse first"
            "\$VENV/bin/pip" install --no-cache-dir --disable-pip-version-check \\
                --no-index --find-links "\$WHEELHOUSE" --upgrade pip
        else
            echo "error: this venv has pip \$("\$VENV/bin/pip" --version | cut -d' ' -f2)," >&2
            echo "       which is too old to read the wheel tags in the lock file" >&2
            echo "       (polars is manylinux_2_24_aarch64, torchvision is bare" >&2
            echo "       linux_aarch64), and the wheelhouse holds no pip wheel." >&2
            echo "Fix: add one to the wheelhouse on the machine with the image," >&2
            echo "     then re-run:" >&2
            echo "    docker run --rm --platform linux/arm64 -v \"\\\$PWD/wheelhouse-nano:/wh\" \\\\" >&2
            echo "        --entrypoint pip pose-est:nano download pip -d /wh" >&2
            exit 1
        fi
    fi
    echo "installing offline from \$WHEELHOUSE"
    "\$VENV/bin/pip" install --no-cache-dir --disable-pip-version-check \\
        --no-index --find-links "\$WHEELHOUSE" -r "\$LOCK"
elif "\$PY" - <<'PROBE' 2>/dev/null
import socket, sys
try:
    socket.create_connection(("pypi.org", 443), timeout=5).close()
except OSError:
    sys.exit(1)
PROBE
then
    echo "no wheelhouse, but pypi.org answers: installing from the lock file"
    "\$VENV/bin/pip" install --no-cache-dir --disable-pip-version-check --upgrade pip
    "\$VENV/bin/pip" install --no-cache-dir --disable-pip-version-check -r "\$LOCK"
else
    cat >&2 <<'NOWHEELS'
error: no wheelhouse at the prefix and pypi.org does not answer, so there is
       nothing to install from. Build the wheelhouse on the machine that has
       the aarch64 image -- inside that image, so the resolution is native:

$(wheelhouse_recipe)

       then re-run with --wheelhouse wheelhouse-nano.
NOWHEELS
    exit 1
fi

# An install that does not import is not an install. YOLO_OFFLINE keeps
# ultralytics from reaching for the network while we are only asking it what
# version it is.
YOLO_OFFLINE=1 YOLO_AUTOINSTALL=0 "\$PY" - <<'VERIFY'
import open3d, cv2, ultralytics, trimesh, scipy, matplotlib, torch
print("IMPORTS=ok")
print("VERSIONS=torch %s  open3d %s  cv2 %s  ultralytics %s  numpy %s"
      % (torch.__version__, open3d.__version__, cv2.__version__,
         ultralytics.__version__, __import__("numpy").__version__))
VERIFY
EOF
)"
[ "$DRY_RUN" = 1 ] || [ -n "$(from_last IMPORTS)" ] \
    || die "the board's venv did not import the runtime" \
"Re-run with --rebuild-venv to build it from scratch."

# -- 5. config -----------------------------------------------------------
step "config"
remote "install and validate the config" "$(cat <<EOF
$(remote_preamble)
# The service resolves relative paths against its working directory and
# hashes the resolved absolute paths into the digest that comes back with
# every pose. Rewriting them here means the digest is stable no matter who
# starts the service or from where. Re-running is a no-op: an absolute path
# stays as it is.
"\$PY" - "\$PREFIX" "\$CONFIG" <<'REWRITE'
import json, os, sys
prefix, path = sys.argv[1], sys.argv[2]
with open(path) as handle:
    config = json.load(handle)
for key in ("cad_path", "seg_weights", "extra_seg_weights"):
    value = config.get(key)
    if value and not os.path.isabs(value):
        config[key] = os.path.join(prefix, value)
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print("rewrote %s" % path)
REWRITE

cd "\$PREFIX"
"\$PY" - "\$CONFIG" <<'VALIDATE'
import sys
from deploy.pose_service.config import ConfigError, ServiceConfig
try:
    config = ServiceConfig.from_file(sys.argv[1]).validate()
except ConfigError as exc:
    print("error: %s" % exc, file=sys.stderr)
    sys.exit(1)
print(config.summary())
print("DIGEST=%s" % config.digest())
print("ENDPOINT=http://%s:%d" % (config.host, config.port))
VALIDATE
EOF
)"
DIGEST="$(from_last DIGEST)"
ENDPOINT="$(from_last ENDPOINT)"
[ "$DRY_RUN" = 1 ] && { DIGEST="<computed on the board>"; ENDPOINT="http://127.0.0.1:8080"; }
[ -n "$ENDPOINT" ] || die "the board did not validate $PREFIX/$CONFIG_REL" \
"Fix the setting it named and re-run; nothing was started."

# -- 6. service ----------------------------------------------------------
step "systemd unit"
if [ "$NO_SERVICE" = 1 ]; then
    note "skipped (--no-service). To run it by hand on the board:"
    note "    cd $PREFIX && .venv/bin/python -m deploy.pose_service.server --config $PREFIX/$CONFIG_REL"
else
    remote "install and start the unit" "$(cat <<EOF
$(remote_preamble)
if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
    echo "error: no running systemd here, so the unit cannot be installed." >&2
    echo "Fix: re-run with --no-service and start the service yourself:" >&2
    echo "    cd \$PREFIX && .venv/bin/python -m deploy.pose_service.server --config \$CONFIG" >&2
    exit 1
fi

if id -u "\$SERVICE_USER" >/dev/null 2>&1; then
    echo "service user \$SERVICE_USER exists"
else
    \$SUDO useradd --system --home-dir /var/lib/pose-service \\
                  --shell /usr/sbin/nologin "\$SERVICE_USER"
    echo "created service user \$SERVICE_USER"
fi
\$SUDO chown -R "\$SERVICE_USER:\$SERVICE_USER" "\$PREFIX"

# The shipped unit hard-codes /opt/pose-estimation in WorkingDirectory,
# ExecStart and Documentation; a different --prefix has to reach all three.
# With the default prefix this substitution changes nothing.
UNIT_TMP="\$(mktemp)"
trap 'rm -f "\$UNIT_TMP"' EXIT
sed "s|/opt/pose-estimation|\$PREFIX|g" \\
    "\$PREFIX/deploy/jetson-nano/\$UNIT" > "\$UNIT_TMP"
\$SUDO install -m 0644 "\$UNIT_TMP" "/etc/systemd/system/\$UNIT"
\$SUDO systemctl daemon-reload
\$SUDO systemctl enable "\$UNIT"
# restart, not start: a re-provision must run the payload it just copied.
if ! \$SUDO systemctl restart "\$UNIT"; then
    echo "error: \$UNIT did not start." >&2
    \$SUDO systemctl status "\$UNIT" --no-pager -l 2>&1 | sed 's/^/    /' >&2 || true
    echo "The board is installed but not serving. Look at:" >&2
    echo "    journalctl -u \$UNIT -n 50 --no-pager" >&2
    exit 1
fi
printf 'UNIT_STATE=%s %s\n' "\$(\$SUDO systemctl is-enabled "\$UNIT" || true)" \\
                            "\$(\$SUDO systemctl is-active "\$UNIT" || true)"
EOF
)"
    UNIT_STATE="$(from_last UNIT_STATE)"
fi

# -- 7. health -----------------------------------------------------------
step "health"
# With --no-service nothing was started, so this is only asking whether
# something happens to be serving already; do not spend a minute finding out.
[ "$NO_SERVICE" = 1 ] && SOCKET_TIMEOUT=5
HEALTH_RC=0
remote "poll /healthz then /readyz" "$(cat <<EOF
$(remote_preamble)
"\$PY" - $(qq "$ENDPOINT") $(qq "$SOCKET_TIMEOUT") $(qq "$READY_TIMEOUT") $(qq "$NO_SERVICE") <<'POLL' || exit \$?
import json
import sys
import time
import urllib.error
import urllib.request

base, socket_budget, ready_budget, optional = (
    sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1")


def probe(path):
    """(status, body). 503 is an answer -- its body says how far it got."""
    try:
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return response.getcode(), json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except ValueError:
            return exc.code, {}
    except OSError:
        return None, {}


# A monotonic clock, not the wall clock: the Nano has no RTC battery and its
# time can jump the moment it reaches an NTP server.
start = time.monotonic()
while probe("/healthz")[0] is None:
    if time.monotonic() - start > socket_budget:
        if optional:
            print("SKIP=nothing is listening on %s (--no-service)" % base)
            sys.exit(3)
        print("error: nothing answers on %s after %.0f s. The service binds "
              "its socket before it loads anything, so a silent port means "
              "the process is not running.\n"
              "    journalctl -u pose-service -n 50 --no-pager"
              % (base, socket_budget), file=sys.stderr)
        sys.exit(1)
    time.sleep(1)
print("socket answering after %.0f s" % (time.monotonic() - start))

# /healthz turns 200 when the CAD cloud and the segmenter are in memory;
# /readyz waits for the warmup inference on top of that.
last = {}
reported = 0.0
for route in ("/healthz", "/readyz"):
    while True:
        status, last = probe(route)
        waited = time.monotonic() - start
        if status == 200:
            print("%s 200 after %.0f s" % (route, waited))
            break
        if waited > ready_budget:
            print("error: %s still %s after %.0f s (status %r). Not ready is "
                  "not the same as broken -- check the journal before "
                  "restarting anything:\n"
                  "    journalctl -u pose-service -n 50 --no-pager"
                  % (route, last.get("status", "silent"), waited, status),
                  file=sys.stderr)
            sys.exit(1)
        # A line every 15 s: enough to see it moving on a board that takes
        # minutes, not so much that it buries the result.
        if waited - reported >= 15:
            reported = waited
            print("  %s: %s, %.0f s elapsed" % (route, last.get("status", "?"), waited))
        time.sleep(2)

memory = last.get("memory", {})
print("READY=%s status=%s load %.1fs warmup %.1fs rss %.1f MB (delta %.1f MB) digest %s"
      % (base, last.get("status"), memory.get("load_s", 0.0),
         memory.get("warmup_s", 0.0), memory.get("rss_mb", 0.0),
         memory.get("delta_mb", 0.0), last.get("config_digest")))
POLL
EOF
)" || HEALTH_RC=$?

HEALTH_SKIPPED=0
case "$HEALTH_RC" in
    0) ;;
    3) HEALTH_SKIPPED=1; note "no service listening and --no-service was given: steps 7 and 8 skipped" ;;
    *) die "the service on $TARGET never became ready" \
"It is installed and enabled. Read the journal before restarting it:
    ssh $TARGET journalctl -u $UNIT_NAME -n 80 --no-pager" ;;
esac
READY_LINE="$(from_last READY)"

# -- 8. first pick -------------------------------------------------------
step "first pick"
ACCEPTED=0
if [ "$HEALTH_SKIPPED" = 1 ]; then
    note "skipped: nothing is serving on the board"
elif [ -z "$SCENE_ID" ]; then
    note "skipped: no commissioning frame was carried (--scene DIR)"
else
    PICK_RC=0
    remote "estimate one frame" "$(cat <<EOF
$(remote_preamble)
cd "\$PREFIX"
set +e
"\$PY" -m deploy.pose_service.client estimate \\
    --scene "\$PREFIX/commissioning/$SCENE_ID" --timeout $PICK_TIMEOUT
rc=\$?
set -e
case "\$rc" in
    0) echo "PICK=ok" ;;
    2) echo "PICK=rescan" ;;
    *) echo "error: the estimate request failed (exit \$rc). The chain is not" >&2
       echo "       proven; do not hand this board to the cell." >&2
       echo "    journalctl -u \$UNIT -n 50 --no-pager" >&2
       exit 1 ;;
esac
EOF
)" || PICK_RC=$?
    [ "$PICK_RC" = 0 ] || die "the first pick failed on $TARGET" \
"The service answered health checks but could not estimate the frame.
    ssh $TARGET journalctl -u $UNIT_NAME -n 80 --no-pager"
    case "$(from_last PICK)" in
        ok) ACCEPTED=1 ;;
        rescan) ACCEPTED=1
            note "the service answered but the top pose did not clear the accept gate."
            note "The chain works; that frame did not pick. On every platform measured"
            note "so far test/000001 comes back at score ~0.85, so check the weights and"
            note "the config digest before trusting this board." ;;
        *) [ "$DRY_RUN" = 1 ] || die "no verdict from the client on $TARGET" \
"The estimate call returned 0 but printed neither PICK=ok nor PICK=rescan, so
the client on the board is not the one this script expects. Check that the
payload copied cleanly:
    ssh $TARGET '$PREFIX/.venv/bin/python -m deploy.pose_service.client --help'" ;;
    esac
fi

# -- 9. summary ----------------------------------------------------------
step "summary"
remote "record the digest in the marker" "$(marker_script "$DIGEST")"
PROVISIONED=1

cat <<SUMMARY

   prefix     $PREFIX  on $TARGET
   config     $PREFIX/$CONFIG_REL   digest $DIGEST
   endpoint   $ENDPOINT
SUMMARY
if [ "$NO_SERVICE" = 1 ]; then
    printf '   service    not installed (--no-service)\n'
else
    printf '   service    %s -- %s\n' "$UNIT_NAME" "${UNIT_STATE:-see systemctl status}"
fi
[ -n "$READY_LINE" ] && printf '   ready      %s\n' "$READY_LINE"
if [ "$DRY_RUN" = 1 ]; then
    printf '   accepted   step 8 decides that on a real run\n'
elif [ "$ACCEPTED" = 1 ]; then
    printf '   accepted   the board returned a pose for %s\n' "$SCENE_ID"
elif [ "$HEALTH_SKIPPED" = 1 ]; then
    printf '   accepted   not run: nothing was serving (--no-service)\n'
else
    printf '   ACCEPTED   NO -- this board has not returned a pose yet\n'
fi
# accept.sh's default is the full test split, which this script never copies.
# When all the board has is the commissioning frame, point it at that instead
# of printing a command that fails its own preflight.
SCENE_HINT=""
[ -n "$SCENE_ID" ] && SCENE_HINT=" \\
          --release $PREFIX --split commissioning --scenes $SCENE_ID"

cat <<SUMMARY

   logs       ssh $TARGET journalctl -u $UNIT_NAME -f
   remove     deploy/jetson-nano/uninstall.sh $TARGET --prefix $PREFIX --yes

   next, in this order:
   1. clock the board, or its timings mean nothing (README section 0):
      ssh $TARGET 'sudo nvpmodel -m 0 && sudo jetson_clocks'
   2. run the acceptance from here. It benches the board, diffs the record
      against results/bench/board_nano640.json -- a Jetson Nano 4 GB at MAXN,
      2.6-2.7 s a pick, 624 MB -- and writes one page plus the artefacts to
      send back. It does the copying back; nothing else has to.
      deploy/jetson-nano/accept.sh $TARGET --prefix $PREFIX$SCENE_HINT
      Thresholds and what to do about each line: deploy/jetson-nano/ACCEPTANCE.md
SUMMARY

if [ "$DRY_RUN" = 1 ]; then
    printf '\n== dry run complete -- nothing above was executed\n'
elif [ "$ACCEPTED" = 1 ]; then
    printf '\n== %s is serving poses\n' "$TARGET"
elif [ "$NO_SERVICE" = 1 ]; then
    printf '\n== %s is installed; the service was not started (--no-service)\n' "$TARGET"
else
    printf '\n== %s is installed but has not returned a pose -- not accepted\n' "$TARGET" >&2
    exit 1
fi
