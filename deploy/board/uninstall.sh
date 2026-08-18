#!/usr/bin/env bash
# Remove what deploy/board/provision.sh installed on a Jetson, and
# nothing else:
#
#   deploy/board/uninstall.sh nano@192.168.1.50               # show the plan
#   deploy/board/uninstall.sh nano@192.168.1.50 --yes         # do it
#   deploy/board/uninstall.sh nano@192.168.1.50 --dry-run     # print the commands
#
# In order: stop the unit, disable it, remove the unit file, daemon-reload,
# then remove the install prefix. Without --yes it prints exactly what it
# would delete and stops -- reading the board, changing nothing.
#
# Blast radius is the point of this file. It removes ONE directory, and only
# if all of the following hold:
#
#   - the prefix is absolute and has at least two path components, so /opt,
#     /home, /usr and / cannot be named even by accident;
#   - it is not one of the system directories in DENY below;
#   - it carries the marker file provision.sh writes, so a prefix that is
#     someone else's directory -- or a typo that happens to exist -- is
#     refused rather than emptied.
#
# The prefix check is repeated on the board, immediately before the rm, so
# the guarantee does not depend on this script being the thing that called
# it. What it deliberately leaves behind: the "pose" system user and
# /var/lib/pose-service (the unit's StateDirectory, outside the prefix). Both
# are printed at the end with the command that removes them, because a
# service account and its state are an operator's decision, not a script's.
set -euo pipefail

PREFIX="/opt/pose-estimation"
UNIT_NAME="pose-service.service"
SERVICE_USER="pose"
MARKER_NAME=".pose-estimation-install"
MARKER_ID="pose-estimation-provision-v1"
STATE_DIR="/var/lib/pose-service"      # systemd StateDirectory=, outside the prefix

# Directories that must never be the prefix, whatever the marker says. The
# general guard is the marker file -- no list can enumerate every directory
# that is not ours -- but the obvious ones are worth naming, because the
# expensive mistake is a typo that happens to name a real system directory.
DENY="/ /bin /boot /dev /etc /home /lib /media /mnt /opt /proc /root /run /sbin /srv /sys /tmp /usr /var
      /etc/systemd /usr/bin /usr/lib /usr/local /usr/share /var/lib /var/log /var/run"

IDENTITY=""
SSH_PORT=""
CONFIRMED=0
DRY_RUN=0
TARGET=""

usage() {
    cat <<'USAGE'
usage: deploy/board/uninstall.sh <user@host> [options]

  --prefix DIR     install prefix to remove (default /opt/pose-estimation)
  --identity KEY   ssh private key
  --port N         ssh port
  --yes            actually stop the service and delete the prefix
  --dry-run        print every remote command, run none of them
  -h, --help       this text

Without --yes nothing is stopped and nothing is deleted: it prints the plan.
USAGE
}

die() {
    printf '\nerror: %s\n' "$1" >&2
    [ $# -gt 1 ] && printf '%s\n' "$2" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --identity) IDENTITY="${2:?--identity needs a key file}"; shift 2 ;;
        --port) SSH_PORT="${2:?--port needs a number}"; shift 2 ;;
        --yes) CONFIRMED=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -*) usage >&2; die "unknown option: $1" ;;
        *)  [ -z "$TARGET" ] || { usage >&2; die "one target only, got '$TARGET' and '$1'"; }
            TARGET="$1"; shift ;;
    esac
done
[ -n "$TARGET" ] || { usage >&2; die "no target: give <user@host>"; }

# -- the prefix has to survive this before anything is contacted -----------
case "$PREFIX" in
    /) die "refusing to remove '/'" "That is the whole filesystem." ;;
    /*) ;;
    *) die "--prefix must be absolute, got '$PREFIX'" ;;
esac
PREFIX="${PREFIX%/}"

# "/opt" is one component, "/opt/pose-estimation" is two. Anything this
# script may delete has to be a directory somebody made for it, not a place
# the distribution owns.
COMPONENTS="$(printf '%s' "${PREFIX#/}" | awk -F/ '{ print NF }')"
[ "$COMPONENTS" -ge 2 ] || die "refusing to remove '$PREFIX': fewer than two path components" \
"An install prefix must be something like /opt/pose-estimation, so that a
typo cannot name a system directory. Use --prefix to say which one."
for denied in $DENY; do
    [ "$PREFIX" = "$denied" ] && die "refusing to remove '$PREFIX': it is a system directory" \
"Whatever is there, it is not only ours."
done

# -- ssh ------------------------------------------------------------------
SSH_CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pose-uninstall.XXXXXX")"
LAST_OUT="$SSH_CONTROL_DIR/last-output"
: > "$LAST_OUT"
SSH_OPTS=(-o ConnectTimeout=10
          -o "ControlMaster=auto"
          -o "ControlPath=$SSH_CONTROL_DIR/ssh-%r@%h:%p"
          -o "ControlPersist=60s")
[ -n "$IDENTITY" ] && SSH_OPTS+=(-i "$IDENTITY")
[ -n "$SSH_PORT" ] && SSH_OPTS+=(-p "$SSH_PORT")

cleanup() {
    if [ "$DRY_RUN" != 1 ]; then
        ssh "${SSH_OPTS[@]}" -O exit "$TARGET" >/dev/null 2>&1 || true
    fi
    rm -rf "$SSH_CONTROL_DIR"
}
trap cleanup EXIT

qq() { printf '%q' "$1"; }
note() { printf '   %s\n' "$1"; }

remote() {
    local label="$1" script="$2"
    : > "$LAST_OUT"
    if [ "$DRY_RUN" = 1 ]; then
        printf '\n   --- %s: would run on %s ---\n' "$label" "$TARGET"
        printf '%s\n' "$script" | sed 's/^/   | /'
        printf '   --- sent as:'
        printf ' %q' ssh "${SSH_OPTS[@]}" "$TARGET" bash -s
        printf ' <<script ---\n'
        return 0
    fi
    printf '%s' "$script" | ssh "${SSH_OPTS[@]}" "$TARGET" bash -s | tee "$LAST_OUT"
}

from_last() { sed -n "s/^$1=//p" "$LAST_OUT" | tail -1; }

# The same guard the driver applied, re-applied where the rm actually
# happens. Sourced into every remote script that touches the filesystem.
remote_preamble() {
    cat <<EOF
set -euo pipefail
PREFIX=$(qq "$PREFIX")
UNIT=$(qq "$UNIT_NAME")
MARKER="\$PREFIX/$MARKER_NAME"
# Empty when the login is already root; unquoted at the call sites so an
# empty value disappears instead of becoming argv[0].
SUDO=$(qq "$SUDO")

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
    if [ ! -f "\$MARKER" ] || ! grep -q '$MARKER_ID' "\$MARKER"; then
        echo "error: \$PREFIX carries no $MARKER_NAME from provision.sh." >&2
        echo "       This uninstaller only removes directories it installed;" >&2
        echo "       whatever is there was put there by something else." >&2
        exit 1
    fi
}
EOF
}

printf '== uninstalling from %s\n' "$TARGET"
note "prefix   $PREFIX"
note "unit     $UNIT_NAME"
[ "$DRY_RUN" = 1 ] && note "DRY RUN -- nothing below is executed"

# Stopping a unit and removing a directory under /opt both need root. Find
# out once, before printing a plan this login cannot carry out.
SUDO="sudo"
if [ "$DRY_RUN" != 1 ]; then
    SUDO_MODE="$(printf '%s' 'if [ "$(id -u)" = 0 ]; then echo root
elif sudo -n true 2>/dev/null; then echo sudo
else echo password; fi' | ssh "${SSH_OPTS[@]}" "$TARGET" bash -s)" \
        || die "cannot reach $TARGET over ssh" \
"Check the address and the key:
    ssh${IDENTITY:+ -i $IDENTITY}${SSH_PORT:+ -p $SSH_PORT} $TARGET true"
    case "$SUDO_MODE" in
        root) SUDO="" ;;
        sudo) SUDO="sudo" ;;
        *) die "sudo on $TARGET wants a password, and there is no terminal to ask on" \
"Pick one:
    - run it as root:   $(basename "$0") root@${TARGET#*@} --prefix $PREFIX
    - or, on the board: echo \"\$USER ALL=(ALL) NOPASSWD:ALL\" | sudo tee /etc/sudoers.d/90-pose-provision
Nothing has been changed." ;;
    esac
fi

# -- what is actually there ----------------------------------------------
printf '\n== 1/3  what is on the board\n'
remote "inspect" "$(cat <<EOF
$(remote_preamble)
if [ -d "\$PREFIX" ]; then
    guard_prefix
    echo "PREFIX_PRESENT=1"
    echo "marker:"
    sed 's/^/    /' "\$MARKER"
    printf 'PREFIX_SIZE=%s\n' "\$(du -sh "\$PREFIX" 2>/dev/null | cut -f1)"
    echo "top level of \$PREFIX:"
    ls -A "\$PREFIX" | sed 's/^/    /'
else
    # An uninstall that has already run leaves nothing; say so instead of
    # accusing the operator of pointing at someone else's directory.
    echo "PREFIX_PRESENT=0"
    echo "\$PREFIX is not there"
fi

if [ -f "/etc/systemd/system/\$UNIT" ]; then
    printf 'UNIT_FILE=/etc/systemd/system/%s\n' "\$UNIT"
    # A board has one unit of this name. Which prefix it serves decides
    # whether it is this uninstall's to remove.
    printf 'UNIT_PREFIX=%s\n' \\
        "\$(sed -n 's|^WorkingDirectory=||p' "/etc/systemd/system/\$UNIT" | tail -1)"
    if command -v systemctl >/dev/null 2>&1; then
        printf 'UNIT_STATE=%s %s\n' "\$(systemctl is-enabled "\$UNIT" 2>/dev/null || true)" \\
                                    "\$(systemctl is-active "\$UNIT" 2>/dev/null || true)"
    fi
else
    echo "UNIT_FILE=none installed"
fi
EOF
)"

UNIT_FILE="$(from_last UNIT_FILE)"
PREFIX_SIZE="$(from_last PREFIX_SIZE)"
PREFIX_PRESENT="$(from_last PREFIX_PRESENT)"
[ "$DRY_RUN" = 1 ] && { PREFIX_PRESENT=1; UNIT_FILE="/etc/systemd/system/$UNIT_NAME"; }
UNIT_PREFIX="$(from_last UNIT_PREFIX)"
HAVE_UNIT=1
[ "${UNIT_FILE:-none installed}" = "none installed" ] && HAVE_UNIT=0
# The unit on the board may belong to a different install. Stopping the cell's
# vision service while removing some other directory is not a tidy-up, it is
# an outage.
FOREIGN_UNIT=0
if [ "$HAVE_UNIT" = 1 ] && [ -n "$UNIT_PREFIX" ] && [ "$UNIT_PREFIX" != "$PREFIX" ]; then
    FOREIGN_UNIT=1
    HAVE_UNIT=0
fi

printf '\n== 2/3  what will be removed\n'
if [ "$PREFIX_PRESENT" != 1 ] && [ "$HAVE_UNIT" = 0 ]; then
    if [ "$FOREIGN_UNIT" = 1 ]; then
        note "nothing: no $PREFIX here, and $UNIT_NAME serves $UNIT_PREFIX."
        note "         Point --prefix at $UNIT_PREFIX to remove that one."
    else
        note "nothing: no $PREFIX and no unit file. This board is already clean."
    fi
    exit 0
fi
if [ "$FOREIGN_UNIT" = 1 ]; then
    note "left alone        $UNIT_NAME serves $UNIT_PREFIX, not $PREFIX"
elif [ "$HAVE_UNIT" = 0 ]; then
    note "no unit file on the board -- nothing to stop"
else
    note "stop and disable  $UNIT_NAME  (the cell loses its vision service)"
    note "delete            $UNIT_FILE"
fi
if [ "$PREFIX_PRESENT" = 1 ]; then
    note "delete            $PREFIX  ${PREFIX_SIZE:+($PREFIX_SIZE)}  -- the whole directory, recursively"
else
    note "no prefix         $PREFIX is not there"
fi
note "kept              $STATE_DIR and the '$SERVICE_USER' user (outside the prefix)"

if [ "$CONFIRMED" != 1 ]; then
    printf '\n== nothing was removed. Re-run with --yes to do it:\n'
    printf '   %s %s --prefix %s --yes\n' \
        "deploy/board/uninstall.sh" "$TARGET" "$PREFIX"
    exit 0
fi

# -- remove ---------------------------------------------------------------
printf '\n== 3/3  removing\n'
remote "stop, disable, delete" "$(cat <<EOF
$(remote_preamble)
if [ "$HAVE_UNIT" = 1 ] && [ -f "/etc/systemd/system/\$UNIT" ] \\
        && command -v systemctl >/dev/null 2>&1; then
    # Stop before disable so a unit that refuses to die is reported here,
    # while its files are still on the board to look at.
    \$SUDO systemctl stop "\$UNIT" || echo "warning: systemctl stop \$UNIT failed" >&2
    \$SUDO systemctl disable "\$UNIT" || true
    \$SUDO rm -f "/etc/systemd/system/\$UNIT"
    \$SUDO systemctl daemon-reload
    # Otherwise a unit that died leaves its failure state behind and the next
    # install inherits it.
    \$SUDO systemctl reset-failed "\$UNIT" 2>/dev/null || true
    echo "unit stopped, disabled and removed"
elif [ "$FOREIGN_UNIT" = 1 ]; then
    echo "\$UNIT serves $UNIT_PREFIX -- left installed and running"
else
    echo "no unit file installed -- nothing to stop"
fi

if [ -d "\$PREFIX" ]; then
    # Guarded twice on purpose: once in the driver before anything was
    # contacted, and again here, where the rm actually runs.
    guard_prefix
    \$SUDO rm -rf -- "\$PREFIX"
    if [ -e "\$PREFIX" ]; then
        echo "error: \$PREFIX is still there after rm -rf" >&2
        exit 1
    fi
    echo "removed \$PREFIX"
else
    echo "\$PREFIX was already gone"
fi
EOF
)"

# What happened, not what was planned. A dry run removed nothing, and a prefix
# that was already gone was not removed by this run either; reporting either as
# a removal teaches the operator to distrust the whole page.
if [ "$DRY_RUN" = 1 ]; then
    REMOVED="nothing -- dry run. Re-run without --dry-run to carry the plan out."
elif [ "$PREFIX_PRESENT" = 1 ] && [ "$HAVE_UNIT" = 1 ]; then
    REMOVED="$PREFIX and $UNIT_NAME on $TARGET"
elif [ "$PREFIX_PRESENT" = 1 ]; then
    REMOVED="$PREFIX on $TARGET (no unit file was installed)"
elif [ "$HAVE_UNIT" = 1 ]; then
    REMOVED="$UNIT_NAME on $TARGET ($PREFIX was already gone)"
else
    REMOVED="nothing on $TARGET -- it was already clean"
fi

cat <<SUMMARY

   removed    $REMOVED
   left       $STATE_DIR (the unit's StateDirectory) and the '$SERVICE_USER' user

   Both survive a reinstall, which is why they are left. To remove them too:
       ssh $TARGET 'sudo rm -rf $STATE_DIR && sudo userdel $SERVICE_USER'
SUMMARY
