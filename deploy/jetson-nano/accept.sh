#!/usr/bin/env bash
# Acceptance run for a provisioned Jetson Nano: collect the evidence that says
# the board is working, in a form that can be mailed back and diffed against
# the committed baselines in results/bench/.
#
#   deploy/jetson-nano/accept.sh <user@host> [options]
#
#   --prefix DIR         install prefix on the board   (default /opt/pose-estimation)
#   --release DIR        release folder on the board holding <split>/ and model/
#                        (default: the prefix)
#   --split NAME         split to benchmark            (default test)
#   --scenes "A B C"     scenes to benchmark           (default 000001 000002 000015)
#   --out DIR            where the artefacts land here (default out_board)
#   --journal-lines N    service journal tail          (default 400)
#   --clean              remove the board-side work directory when done
#
# What it collects into --out:
#
#   board.json     bench.py on the board, board profile, three repeats
#   compare-*.txt  compare_bench.py against the three committed baselines --
#                  the measured board first, then emulated and x86
#   service.json   /healthz, /readyz, /metrics, the unit, the config digest
#   env.txt        board fingerprint, thermals before and after the bench
#   journal.txt    the service journal tail -- an OOM or a restart the
#                  numbers alone would hide
#   bench.log      everything bench.py printed, including why it stopped
#   summary.txt    the one page, also printed here
#
# The thresholds and what to do when one fails are in
# deploy/jetson-nano/ACCEPTANCE.md. Read that before acting on a FAIL.
#
# Exit: 0 everything collected and every gate passed, 1 collected but a gate
# failed, 2 could not collect (unreachable board, missing files, bench died).
#
# Environment: SSH (the ssh binary, default `ssh`), SSH_OPTS (extra ssh
# arguments, word-split), PY (the local interpreter, default the repo venv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The board profile, spelled out. bench.py's own defaults are the desktop
# configuration (part-seg.pt, both segmenters, 960); a run without these three
# arguments benches a 27.62 M-parameter model while config.nano.json serves a
# 2.84 M one, and the difference would read as hardware when the record is
# diffed against the baselines. They are constants here, not flags, because
# the point of this script is that the board is asked the shipped question.
SEG_MODEL="weights/part-seg-nano.pt"
EXTRA_SEG_MODEL=""            # one segmenter: config.nano.json's extra_seg_weights is null
IMGSZ=640                     # config.nano.json's seg_imgsz
REPEAT=3                      # RANSAC is stochastic; min and median need repeats
# bench.py defaults to this today, but compare_bench.py refuses to compare
# records whose profile differs, so the run that has to match the baselines
# states it rather than inheriting it.
BENCH_PROFILE=nano            # 4 threads, CPU only -- config.nano.json's omp_threads

# The committed baselines this board record is diffed against. The board
# record is the one that matters -- same silicon, same JetPack, same profile --
# so a second board is judged against the first rather than against a guess.
BASELINE_BOARD="$ROOT/results/bench/board_nano640.json"
BASELINE_EMULATED="$ROOT/results/bench/emulated_nano640.json"
BASELINE_NATIVE="$ROOT/results/bench/native_nano640.json"

# A bin-picking cell's vision step has to fit inside the robot's own motion,
# which is 4-8 s per pick for this class of cell; the cell's watchdog upper
# bound is ~60 s (deploy/OFFLINE.md). Unlike every other estimate in this
# repository these bounds now have a Cortex-A57 measurement behind them:
# results/bench/board_nano640.json is a Jetson Nano 4 GB on JetPack 4.6 at
# MAXN, headless, and it picks in 2.6-2.7 s. So this run confirms a number
# rather than establishing one, and a board far off it is a board in a
# different state -- which is what BOARD_REFERENCE_S below is for.
BUDGET_LOW_S=4.0
BUDGET_HIGH_S=8.0
# The slowest scene median in results/bench/board_nano640.json (000001, 2.708 s;
# the other two are 2.69). Reported beside this run's number so a regression is
# visible even while both sit inside the budget.
BOARD_REFERENCE_S=2.7
# Beyond this multiple of the reference, say so. The board record's own
# repeat-to-repeat spread is ~2 % (2.646-2.708 s on one scene), and the usual
# causes of a slow board -- the 5 W power model instead of MAXN, unpinned
# clocks, a desktop session holding cores and RAM -- each cost more than half
# again. 1.5x is therefore well clear of measurement noise and below the
# smallest real fault.
CYCLE_REGRESSION_FACTOR=1.5

# pose-service.service caps the unit at 2600M. Read from the running unit when
# systemd answers; this is the fallback so the summary still has a denominator.
MEMORY_MAX_MB=2600
UNIT="pose-service"

# The cell's accept gate: score >= 0.7 carries ~0.99 precision at 5 mm on
# cross-validated data (analysis/score_calibration.md), and pick mode stops
# the sweep at the first pose over 0.8 (config.nano.json's pick_score).
ACCEPT_SCORE=0.7
PICK_SCORE=0.8

PREFIX="/opt/pose-estimation"
RELEASE=""
SPLIT="test"
SCENES="000001 000002 000015"
OUT="out_board"
JOURNAL_LINES=400
CLEAN=0

die() { printf 'accept: %s\n' "$1" >&2; exit "${2:-2}"; }
step() { printf '\n== %s ==\n' "$1"; }
q() { printf '%q' "$1"; }

usage() {
    # The header comment is the help text; print it until it runs out, so
    # editing the header cannot silently truncate --help.
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' \
        "${BASH_SOURCE[0]}"
}

TARGET=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --release) RELEASE="${2:?--release needs a directory}"; shift 2 ;;
        --split) SPLIT="${2:?--split needs a name}"; shift 2 ;;
        --scenes) SCENES="${2:?--scenes needs a list, e.g. \"000001 000002\"}"; shift 2 ;;
        --out) OUT="${2:?--out needs a directory}"; shift 2 ;;
        --journal-lines) JOURNAL_LINES="${2:?--journal-lines needs a count}"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        -*) die "unknown option: $1 (try --help)" ;;
        *)
            [ -z "$TARGET" ] || die "one target only, got '$TARGET' and '$1'"
            TARGET="$1"; shift ;;
    esac
done

[ -n "$TARGET" ] || { usage >&2; die "no target: give the board as user@host"; }
RELEASE="${RELEASE:-$PREFIX}"

case "$PREFIX" in
    /*) : ;;
    *) die "--prefix must be an absolute path on the board, got '$PREFIX'" ;;
esac
# Blast radius. The only thing this script may remove is its own work
# directory under the prefix, so a prefix that is a system root is refused
# before anything is written, not at --clean time.
case "$PREFIX" in
    /|/usr|/usr/*|/etc|/var|/bin|/sbin|/lib|/lib64|/boot|/root|/home|/opt|/srv|/tmp)
        die "refusing to work under '$PREFIX': too close to a system root. \
The install prefix must be its own directory, e.g. /opt/pose-estimation" ;;
esac

for scene in $SCENES; do
    case "$scene" in
        *[!0-9A-Za-z_-]*) die "scene id '$scene' is not [0-9A-Za-z_-]+" ;;
    esac
done
case "$JOURNAL_LINES" in
    ''|*[!0-9]*) die "--journal-lines must be a count, got '$JOURNAL_LINES'" ;;
esac

# Everything the board writes goes here, and this is the only path --clean
# will ever remove.
REMOTE_WORK="$PREFIX/out_accept"

PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "no local python: set PY=/path/to/python3"

SSH_BIN="${SSH:-ssh}"
# Not BatchMode: the board may want a password. ConnectTimeout is what turns
# "the board is not on this network" into an error in ten seconds instead of
# a two-minute stall.
read -r -a SSH_ARGS <<< "${SSH_OPTS:--o ConnectTimeout=10 -o ServerAliveInterval=15}"

# Ship a script to the board without quoting it through two shells: base64 is
# the one encoding every login shell (bash, dash, busybox) passes untouched,
# and the board's login shell is not something this script gets to choose.
# $1 is the interpreter command line including its end-of-options marker,
# $2 the script, the rest the script's own arguments.
remote_run() {
    local interpreter="$1" script="$2"; shift 2
    local encoded args="" arg
    encoded=$(printf '%s' "$script" | base64 | tr -d '\n')
    for arg in "$@"; do args="$args $(q "$arg")"; done
    # Braces, not a bare pipeline: an interpreter reached through `cd x && ...`
    # would otherwise bind the pipe to the `cd` and leave the interpreter
    # reading an empty stdin -- which exits 0 and writes nothing.
    "$SSH_BIN" "${SSH_ARGS[@]}" "$TARGET" \
        "printf %s '$encoded' | base64 -d | { $interpreter$args; }"
}

remote_bash() {
    local script="$1"; shift
    remote_run "bash -s --" "$script" "$@"
}

remote_python() {
    local script="$1"; shift
    # Into the prefix first. ServiceConfig resolves cad_path and the weights
    # with os.path.abspath, so the digest depends on the working directory,
    # and the unit's is WorkingDirectory=<prefix>. An ssh login lands in
    # $HOME, where the identical config file hashes differently -- which
    # would read as configuration drift that is not there.
    remote_run "cd $(q "$PREFIX") && $(q "$PREFIX/.venv/bin/python") -" \
        "$script" "$@"
}

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

printf 'target    %s\n' "$TARGET"
printf 'prefix    %s   (work dir %s)\n' "$PREFIX" "$REMOTE_WORK"
printf 'release   %s   split %s   scenes %s\n' "$RELEASE" "$SPLIT" "$SCENES"
printf 'out       %s\n' "$OUT"
printf 'profile   %s only, imgsz %s, --profile %s, --pick, --repeat %s\n' \
    "$(basename "$SEG_MODEL")" "$IMGSZ" "$BENCH_PROFILE" "$REPEAT"
printf '          passed explicitly: bench.py defaults the weights to the\n'
printf '          desktop configuration (part-seg.pt + part-seg-synthetic.pt\n'
printf '          at 960), which would bench a different model on the same\n'
printf '          board and read as hardware.\n'

# ---------------------------------------------------------------- preflight

step "reaching the board"
if ! "$SSH_BIN" "${SSH_ARGS[@]}" "$TARGET" "true" </dev/null; then
    die "cannot reach $TARGET over ssh.
   fix: check the board is powered and on a subnet this machine can route to
        (\`ip route get ${TARGET#*@}\`), that sshd is running on it, and that
        \`ssh $TARGET true\` succeeds by hand before re-running this script."
fi
printf 'ssh ok\n'

step "preflight"
PREFLIGHT_SH=$(cat <<'PREFLIGHT_SH_EOF'
set -u
prefix="$1"; release="$2"; split="$3"; shift 3
missing=0
need() {
    if [ ! -e "$1" ]; then
        printf "MISSING  %s\n         %s\n         fix: %s\n" "$1" "$2" "$3"
        missing=1
    fi
}
need "$prefix" "the install prefix" \
     "unpack the release there, or pass --prefix"
need "$prefix/.venv/bin/python" "the runtime venv" \
     "PYTHON=python3.8 ./setup.sh --jetson-nano in the prefix (runbook step 2)"
need "$prefix/deploy/jetson-nano/bench.py" "the benchmark" \
     "copy the repo, not just the weights, into the prefix"
need "$prefix/weights/part-seg-nano.pt" "the board segmenter" \
     "copy weights/ into the prefix (runbook step 3)"
need "$release/model/3d_model.ply" "the CAD model" \
     "copy model/ from the release into $release, or pass --release"
for scene in "$@"; do
    need "$release/$split/$scene" "a benchmark scene" \
         "copy the $split split into $release, or pass --scenes"
done
exit $missing
PREFLIGHT_SH_EOF
)
if ! remote_bash "$PREFLIGHT_SH" "$PREFIX" "$RELEASE" "$SPLIT" $SCENES </dev/null; then
    die "the board is not ready to be measured -- fix the paths above and re-run.
   Nothing was written to the board."
fi
printf 'prefix, venv, weights, model and scenes are all present\n'

# ------------------------------------------------------------ fingerprint

ENV_SH=$(cat <<'ENV_SH_EOF'
set -u
prefix="$1"; phase="$2"
section() { printf "\n== %s ==\n" "$1"; }
try() {
    printf "$ %s\n" "$*"
    "$@" 2>&1 || printf "(unavailable: %s exited %d)\n" "$1" "$?"
}
thermals() {
    found=0
    # /sys/class/thermal is the canonical view; /sys/devices/virtual/thermal
    # is the same set of zones through the other path, so listing both would
    # report every sensor twice.
    for zone in /sys/class/thermal/thermal_zone*; do
        [ -r "$zone/temp" ] || continue
        milli="$(cat "$zone/temp" 2>/dev/null || true)"
        [ -n "$milli" ] || continue
        found=1
        printf "%s %s = %s milli-degC\n" "$(basename "$zone")" \
               "$(cat "$zone/type" 2>/dev/null || echo unknown)" "$milli"
    done
    [ "$found" = 1 ] || printf "(no thermal zones: not a Tegra, or not readable)\n"
}

if [ "$phase" = before ]; then
    printf "======== board fingerprint, before the benchmark ========\n"
    printf "collected %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ) (wall clock; the Nano has no RTC battery)"
    section "identity"
    printf "$ cat /proc/device-tree/model\n"
    if [ -r /proc/device-tree/model ]; then
        tr -d "\000" < /proc/device-tree/model
        printf "\n"
    else
        printf "(no device tree model: this is not a Jetson)\n"
    fi
    try cat /etc/nv_tegra_release
    try uname -a
    try lsb_release -a
    section "power and clocks"
    printf "timings are meaningless without nvpmodel -m 0 and jetson_clocks\n"
    try nvpmodel -q
    try sh -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"
    try sh -c "cat /sys/devices/system/cpu/online"
    section "memory"
    try free -m
    try swapon --show
    try df -h "$prefix"
    section "python and packages"
    try "$prefix/.venv/bin/python" -V
    try "$prefix/.venv/bin/pip" --disable-pip-version-check freeze
    section "service unit"
    try systemctl is-active pose-service
    try systemctl is-enabled pose-service
    section "thermals (before the benchmark)"
    thermals
else
    printf "\n\n======== board fingerprint, after the benchmark ========\n"
    section "thermals (after the benchmark)"
    thermals
    section "memory"
    try free -m
    try swapon --show
    section "power and clocks"
    try sh -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"
fi
ENV_SH_EOF
)

step "board fingerprint"
if remote_bash "$ENV_SH" "$PREFIX" before </dev/null > "$OUT/env.txt"; then
    sed -n '1,12p' "$OUT/env.txt"
    printf '... full fingerprint in %s\n' "$OUT/env.txt"
else
    printf 'WARNING: the fingerprint collection failed; %s may be partial\n' \
        "$OUT/env.txt" >&2
fi

# --------------------------------------------------------------- service

SERVICE_PY=$(cat <<'SERVICE_PY_EOF'
"""Ask the running service what it is and what it has done."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

prefix, config_path, unit = sys.argv[1], sys.argv[2], sys.argv[3]
record = {"collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "unit_name": unit, "config_path": config_path}

def fetch(url):
    """Status and body. A 503 from /healthz is an answer, not a failure:
    it is how the service says it is still loading."""
    try:
        # getcode(), not .status: HTTPResponse.status arrived in 3.9 and the
        # board runs 3.8.
        with urllib.request.urlopen(url, timeout=10) as response:
            return {"status": response.getcode(),
                    "body": response.read().decode("utf-8", "replace")}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode("utf-8", "replace")}
    except Exception as exc:
        return {"status": None, "error": "%s: %s" % (type(exc).__name__, exc)}

# The bind address comes from the config the unit points at, so a board that
# moved the service off 127.0.0.1:8080 is still probed where it actually is.
host, port = "127.0.0.1", 8080
sys.path.insert(0, prefix)
try:
    from deploy.pose_service.config import ServiceConfig
    config = ServiceConfig.from_file(config_path)
    host, port = config.host, config.port
    record["config"] = {"digest_on_disk": config.digest(),
                        "summary": config.summary()}
except Exception as exc:
    record["config"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

base = "http://%s:%d" % (host, port)
record["base_url"] = base
record["endpoints"] = {path: fetch(base + path)
                       for path in ("/healthz", "/readyz", "/metrics")}

health = record["endpoints"]["/healthz"]
try:
    health["json"] = json.loads(health.get("body", ""))
except Exception:
    pass
running = (health.get("json") or {}).get("config_digest")
on_disk = (record.get("config") or {}).get("digest_on_disk")
record["config_digest_running"] = running
# A mismatch is not necessarily a fault: /etc/default/pose-service can
# override single settings. It does mean the file on disk is not the whole
# story for this record, which is exactly what the record has to say.
record["config_digest_matches"] = (running == on_disk) if (running and on_disk) else None

def systemctl(*args):
    try:
        out = subprocess.run(["systemctl"] + list(args), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=20)
        return out.stdout.decode("utf-8", "replace").strip()
    except Exception as exc:
        return "(unavailable: %s: %s)" % (type(exc).__name__, exc)

properties = {}
raw = systemctl("show", unit, "--no-pager",
                "--property=ActiveState,SubState,Result,NRestarts,MemoryCurrent,"
                "MemoryMax,MemoryAccounting,ExecMainStartTimestamp,ExecMainPID")
for line in raw.splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        properties[key] = value
record["unit"] = properties or {"error": raw}
json.dump(record, sys.stdout, indent=1)
sys.stdout.write("\n")
SERVICE_PY_EOF
)

step "service"
remote_python "$SERVICE_PY" "$PREFIX" \
    "$PREFIX/deploy/jetson-nano/config.nano.json" "$UNIT" </dev/null \
    > "$OUT/service.json" || true
if "$PY" - "$OUT/service.json" 2>/dev/null <<'PY'; then
import json, sys
record = json.load(open(sys.argv[1]))
health = (record.get("endpoints") or {}).get("/healthz") or {}
body = health.get("json") or {}
print("healthz  %s  status=%s  ready=%s  uptime=%ss"
      % (health.get("status"), body.get("status"), body.get("ready"),
         body.get("uptime_s")))
print("digest   running=%s  on disk=%s"
      % (record.get("config_digest_running"),
         (record.get("config") or {}).get("digest_on_disk")))
unit = record.get("unit") or {}
print("unit     %s/%s  restarts=%s  MemoryMax=%s"
      % (unit.get("ActiveState"), unit.get("SubState"),
         unit.get("NRestarts"), unit.get("MemoryMax")))
PY
    :
else
    printf 'WARNING: the service query came back empty or unreadable.\n' >&2
    printf '         %s holds:\n' "$OUT/service.json" >&2
    head -c 400 "$OUT/service.json" >&2 || true
    printf '\n         The bench below still runs; the summary will say the\n' >&2
    printf '         service was not queried.\n' >&2
fi

# ---------------------------------------------------------------- benchmark

step "benchmark on the board"
BENCH_NOTE="board acceptance run, $(date -u +%Y-%m-%dT%H:%M:%SZ), service left running"
# rm -f first: a bench that dies before it writes would otherwise leave the
# previous run's board.json in place, and everything below would report it as
# this run's evidence. One file, inside the work directory, named in full.
BENCH_CMD="cd $(q "$PREFIX") && mkdir -p $(q "$REMOTE_WORK") && \
rm -f $(q "$REMOTE_WORK/board.json") && \
.venv/bin/python deploy/jetson-nano/bench.py \
--root $(q "$RELEASE") --split $(q "$SPLIT") --scenes $SCENES \
--profile $BENCH_PROFILE --pick --repeat $REPEAT \
--seg-model $(q "$SEG_MODEL") --extra-seg-model $(q "$EXTRA_SEG_MODEL") \
--imgsz $IMGSZ \
--out $(q "$REMOTE_WORK/board.json") --note $(q "$BENCH_NOTE")"
printf 'running on %s:\n  %s\n\n' "$TARGET" "$BENCH_CMD"
printf 'The first frame loads the CAD cloud and the segmenter -- tens of\n'
printf 'seconds on four A57 cores -- and is not the cycle time.\n\n'

BENCH_OK=1
if ! "$SSH_BIN" "${SSH_ARGS[@]}" "$TARGET" "$BENCH_CMD" </dev/null 2>&1 \
        | tee "$OUT/bench.log"; then
    BENCH_OK=0
    printf '\nWARNING: the benchmark exited non-zero; see %s\n' "$OUT/bench.log" >&2
fi

if ! "$SSH_BIN" "${SSH_ARGS[@]}" "$TARGET" "cat $(q "$REMOTE_WORK/board.json")" \
        </dev/null > "$OUT/board.json" 2>/dev/null; then
    rm -f "$OUT/board.json"
    printf '\n' >&2
    printf 'The board left %s in place; log in and look at it.\n' "$REMOTE_WORK" >&2
    die "no board.json came back from $TARGET. The benchmark did not finish --
   the tail of $OUT/bench.log names the failure. A board that ran out of
   memory says so in the journal (journalctl -u $UNIT -n 200)."
fi
printf '\nboard.json  %s bytes\n' "$(wc -c < "$OUT/board.json")"

step "board fingerprint, after"
if remote_bash "$ENV_SH" "$PREFIX" after </dev/null >> "$OUT/env.txt"; then
    printf 'thermals appended to %s\n' "$OUT/env.txt"
else
    printf 'WARNING: post-benchmark fingerprint failed\n' >&2
fi

# ----------------------------------------------------------------- journal

step "service journal"
JOURNAL_SH=$(cat <<'JOURNAL_SH_EOF'
set -u
unit="$1"; lines="$2"
printf "======== journalctl -u %s -n %s ========\n" "$unit" "$lines"
journalctl -u "$unit" -n "$lines" --no-pager 2>&1 \
    || printf "(unavailable: no journalctl, or no permission to read this unit)\n"
printf "\n======== kernel out-of-memory events ========\n"
journalctl -k --no-pager 2>/dev/null | grep -iE "out of memory|oom-kill|killed process" \
    || printf "(none in the kernel journal, or the kernel journal is not readable)\n"
JOURNAL_SH_EOF
)
if ! remote_bash "$JOURNAL_SH" "$UNIT" "$JOURNAL_LINES" </dev/null > "$OUT/journal.txt"; then
    printf 'WARNING: the journal tail failed; %s may be partial\n' "$OUT/journal.txt" >&2
fi
printf '%s lines in %s\n' "$(wc -l < "$OUT/journal.txt" 2>/dev/null || echo 0)" \
    "$OUT/journal.txt"

# -------------------------------------------------------------- comparison

compare_against() {
    local baseline="$1" name="$2" rc=0
    if [ ! -f "$baseline" ]; then
        printf 'missing baseline %s\n' "$baseline" > "$OUT/compare-$name.txt"
        return 3
    fi
    # Emulated/native record first, board second: compare_bench.py's ratio
    # reads "left takes Nx right", and the board is the reference.
    "$PY" "$ROOT/deploy/jetson-nano/compare_bench.py" \
        "$baseline" "$OUT/board.json" > "$OUT/compare-$name.txt" 2>&1 || rc=$?
    cat "$OUT/compare-$name.txt"
    return $rc
}

step "against results/bench/board_nano640.json (the measured board)"
CMP_BOARD_RC=0
compare_against "$BASELINE_BOARD" board || CMP_BOARD_RC=$?

step "against results/bench/emulated_nano640.json"
CMP_EMULATED_RC=0
compare_against "$BASELINE_EMULATED" emulated || CMP_EMULATED_RC=$?

step "against results/bench/native_nano640.json"
CMP_NATIVE_RC=0
compare_against "$BASELINE_NATIVE" native || CMP_NATIVE_RC=$?

# ------------------------------------------------------------------ summary

LOCAL_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

SUMMARY_PY=$(cat <<'SUMMARY_PY_EOF'
"""One page: did the board pass, and on what evidence."""
import json
import os
import re
import sys

(out, target, prefix, budget_low, budget_high, memory_max_mb, accept_score,
 pick_score, board_rc, emulated_rc, native_rc, local_commit, bench_ok,
 board_reference_s, regression_factor) = sys.argv[1:16]
budget_low, budget_high = float(budget_low), float(budget_high)
memory_max_mb, accept_score = float(memory_max_mb), float(accept_score)
pick_score = float(pick_score)
board_reference_s, regression_factor = (float(board_reference_s),
                                        float(regression_factor))

def load_json(name):
    try:
        with open(os.path.join(out, name)) as handle:
            return json.load(handle)
    except Exception:
        return None

def load_text(name):
    try:
        with open(os.path.join(out, name)) as handle:
            return handle.read()
    except Exception:
        return ""

board = load_json("board.json") or {}
service_record = load_json("service.json")
service = service_record or {}
journal = load_text("journal.txt")
env = load_text("env.txt")

lines, warnings, gates = [], [], []

def gate(label, verdict, text):
    gates.append(verdict)
    lines.append("%-7s %-5s %s" % (label, verdict, text))

# -- what was measured ------------------------------------------------------
host = board.get("host") or {}
config = board.get("config") or {}
versions = host.get("versions") or {}
model = host.get("device_tree_model") or "(no device-tree model)"
tegra = host.get("nv_tegra_release") or "(no /etc/nv_tegra_release)"
lines.append("pose acceptance -- %s  (%s)" % (target, prefix))
lines.append("collected into %s" % out)
lines.append("")
lines.append("board    %s | %s" % (model.strip(), tegra))
lines.append("         %s, %s, python %s, torch %s, open3d %s"
             % (host.get("machine"), host.get("kernel"), host.get("python"),
                versions.get("torch"), versions.get("open3d")))
lines.append("code     commit %s on the board (this checkout %s)"
             % (board.get("git_commit") or "unknown", local_commit))
lines.append("profile  %s%s, imgsz %s, pick=%s, repeat %s, %s threads"
             % (config.get("seg_model"),
                " + " + str(config.get("extra_seg_model"))
                if config.get("extra_seg_model") else " only",
                config.get("imgsz"), config.get("pick"), config.get("repeat"),
                config.get("threads")))
lines.append("")

if host.get("platform_kind") != "jetson":
    warnings.append("this record did not come from a Jetson: bench.py calls the "
                    "host %r (no /proc/device-tree/model, no /etc/nv_tegra_release). "
                    "Every number below describes that host, not a board."
                    % host.get("platform_kind"))
if host.get("emulation_evidence"):
    warnings.append("the host bench.py measured is emulated (%s) -- its timings "
                    "are qemu, not silicon."
                    % "; ".join(host["emulation_evidence"]))
if not board.get("git_commit"):
    warnings.append("the board record carries no commit: bench.py found no git "
                    "checkout there, so the code that produced these numbers is "
                    "not identified. Set BENCH_GIT_COMMIT on the board.")
elif local_commit != "unknown" and board["git_commit"] != local_commit:
    warnings.append("the board runs commit %s and this checkout is %s: the "
                    "baselines here may not describe the code that ran."
                    % (board["git_commit"], local_commit))
if bench_ok != "1":
    warnings.append("bench.py exited non-zero; board.json is whatever it "
                    "managed to write (see bench.log).")

# -- pose agreement, the pass/fail ------------------------------------------
def verdict_line(name, rc):
    text = load_text("compare-%s.txt" % name)
    output = ""
    for line in text.splitlines():
        if line.startswith("OUTPUT"):
            output = line.split(":", 1)[1].strip()
            break
    if rc == 0:
        return "PASS", "vs %s: %s" % (name, output or "agrees")
    if rc == 1:
        return "FAIL", "vs %s: %s" % (name, output or "disagrees")
    if rc == 2:
        return "FAIL", ("vs %s: the two runs were not asked the same question, "
                        "or share no scenes -- see compare-%s.txt"
                        % (name, name))
    return "n/a", "vs %s: no baseline to compare against" % name

first = True
for name, rc in (("board", int(board_rc)), ("emulated", int(emulated_rc)),
                 ("native", int(native_rc))):
    verdict, text = verdict_line(name, rc)
    gate("POSE" if first else "", verdict, text)
    first = False
    # "n/a" is not a pass. A deleted baseline means the correctness claim was
    # not made at all, and the page has to say so rather than stay quiet.
    if rc == 3:
        warnings.append("results/bench/%s_nano640.json is missing, so the "
                        "poses were never checked against it -- restore it "
                        "from the repository and re-run before accepting "
                        "this board." % name)
for name in ("board", "emulated", "native"):
    for line in load_text("compare-%s.txt" % name).splitlines():
        if line.startswith("NOISE") and "unmeasured" in line:
            warnings.append("%s comparison: %s" % (name, line.split(":", 1)[1].strip()))

# -- cycle time -------------------------------------------------------------
scenes = board.get("scenes") or {}
timed = []
for scene_id in sorted(scenes):
    scene = scenes[scene_id]
    if scene.get("wall_s_median") is not None:
        timed.append((scene_id, scene["wall_s_min"], scene["wall_s_median"]))
    for error in scene.get("errors") or []:
        warnings.append("scene %s failed a repeat: %s" % (scene_id, error))
if timed:
    worst = max(timed, key=lambda row: row[2])
    best = min(timed, key=lambda row: row[2])
    if worst[2] <= budget_low:
        verdict, note = "PASS", "under the %.0f s floor of the budget, with headroom" % budget_low
    elif worst[2] <= budget_high:
        verdict, note = "PASS", "inside the %.0f-%.0f s budget" % (budget_low, budget_high)
    else:
        verdict, note = "FAIL", "over the %.0f s budget" % budget_high
    gate("CYCLE", verdict,
         "worst scene %s at %.1f s median (min %.1f), best %s at %.1f s -- %s"
         % (worst[0], worst[2], worst[1], best[0], best[2], note))
    lines.append("        %.1fx the measured board (%.1f s, "
                 "results/bench/board_nano640.json, MAXN and headless)"
                 % (worst[2] / board_reference_s, board_reference_s))
    lines.append("        model load %s s, once per process, not part of a cycle"
                 % board.get("model_load_s"))
    # Inside the budget and still much slower than the reference board is the
    # signature of a board in the wrong state, not of a broken one.
    if worst[2] > board_reference_s * regression_factor:
        warnings.append("this board takes %.1fx the %.1f s a Jetson Nano 4 GB "
                        "measured on the same profile: check nvpmodel -m 0 and "
                        "jetson_clocks in env.txt, and that no desktop session "
                        "is running (both are in ACCEPTANCE.md, CYCLE)."
                        % (worst[2] / board_reference_s, board_reference_s))
else:
    gate("CYCLE", "n/a", "no scene produced a timing")

# -- score gate -------------------------------------------------------------
scores = [repeat["top_score"]
          for scene in scenes.values()
          for repeat in scene.get("repeats") or []
          if repeat.get("top_score") is not None]
if scores:
    lowest = min(scores)
    if lowest >= pick_score:
        verdict = "PASS"
    elif lowest >= accept_score:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    gate("SCORE", verdict,
         "lowest top score %.3f over %d runs -- accept gate %.2f, pick stops at %.2f"
         % (lowest, len(scores), accept_score, pick_score))
    if verdict == "WARN":
        warnings.append("a top score between %.2f and %.2f still picks, but the "
                        "sweep ran to the end to find it: the margin is thin."
                        % (accept_score, pick_score))
else:
    gate("SCORE", "n/a", "no scores in board.json")

# -- memory -----------------------------------------------------------------
unit = service.get("unit") or {}
endpoints = service.get("endpoints") or {}
health = endpoints.get("/healthz") or {}
body = health.get("json") or {}
limit_mb, limit_from = memory_max_mb, "pose-service.service as shipped"
raw_limit = unit.get("MemoryMax")
# systemd reports bytes, and its "M" suffix is MiB -- MemoryMax=2600M is
# 2726297600, not 2.6e9. bench.py's peak_rss_mb is ru_maxrss/1024, also MiB,
# so dividing by 1e6 here would inflate the denominator by 4.9 % and quietly
# flatter every board near the cap.
if raw_limit and raw_limit.isdigit() and int(raw_limit) < (1 << 60):
    limit_mb, limit_from = int(raw_limit) / 1048576.0, "the running unit"
peak = board.get("peak_rss_mb")
if peak is not None:
    share = 100.0 * peak / limit_mb
    verdict = "PASS" if share < 80 else ("WARN" if share < 100 else "FAIL")
    gate("MEMORY", verdict,
         "bench peak RSS %.0f MB against MemoryMax %.0f MB (%.0f %%, from %s)"
         % (peak, limit_mb, share, limit_from))
else:
    gate("MEMORY", "n/a", "board.json carries no peak RSS")
current = unit.get("MemoryCurrent")
service_mb = None
if current and current.isdigit() and int(current) < (1 << 60):
    service_mb = int(current) / 1048576.0
elif isinstance((body.get("memory") or {}).get("rss_mb"), (int, float)):
    # No systemd here, or no accounting: /healthz reports the RSS the
    # service is holding, which is the number the reader wanted anyway.
    service_mb = body["memory"]["rss_mb"]
if service_mb is not None:
    lines.append("        the service itself held %.0f MB while the bench ran"
                 % service_mb)

# -- service ----------------------------------------------------------------
if service_record is None:
    gate("SERVICE", "n/a",
         "service.json is missing or unreadable -- the service was never "
         "queried, so this run says nothing about it")
    warnings.append("no service.json: re-run once the unit is up, or the "
                    "record does not describe a serving board.")
elif health.get("status") == 200 and body.get("ready"):
    problems = []
    if service.get("config_digest_matches") is False:
        problems.append("the running digest %s is not the config file on disk %s"
                        % (service.get("config_digest_running"),
                           (service.get("config") or {}).get("digest_on_disk")))
    for path in ("/readyz", "/metrics"):
        if (endpoints.get(path) or {}).get("status") != 200:
            problems.append("%s answered %s" % (path, (endpoints.get(path) or {}).get("status")))
    # Something answering on the port is not the same as the cell's service
    # being under systemd. A hand-started process passes every probe here and
    # is gone after the next reboot, which is the failure a board acceptance
    # exists to catch before the cell is trusted.
    active = unit.get("ActiveState")
    if active and not active.startswith("(") and active != "active":
        problems.append("systemd reports %s as %s/%s while %s answers: what is "
                        "serving was not started by the unit and will not "
                        "survive a reboot"
                        % (service.get("unit_name"), active,
                           unit.get("SubState"), service.get("base_url")))
    gate("SERVICE", "WARN" if problems else "PASS",
         ("ready, digest %s, uptime %s s" % (body.get("config_digest"), body.get("uptime_s")))
         + ("; " + "; ".join(problems) if problems else ""))
    warnings.extend(problems)
elif health.get("status") is None:
    gate("SERVICE", "FAIL",
         "/healthz did not answer: %s" % health.get("error", "no reason recorded"))
else:
    gate("SERVICE", "FAIL",
         "/healthz answered %s, status %r -- the service is up but not picking"
         % (health.get("status"), body.get("status")))

metrics = (endpoints.get("/metrics") or {}).get("body") or ""
counters = {}
for key in ("pose_frames_total", "pose_failures_total", "pose_rejected_total",
            "pose_picks_total"):
    found = re.search(r"^%s (\S+)$" % key, metrics, re.M)
    if found:
        counters[key] = found.group(1)
if counters:
    lines.append("        %s" % "  ".join("%s=%s" % (k.replace("pose_", "").replace("_total", ""), v)
                                          for k, v in counters.items()))

# -- journal ----------------------------------------------------------------
hits = [line for line in journal.splitlines()
        if re.search(r"out of memory|oom-kill|Killed process|Traceback|"
                     r"Failed with result|Main process exited|Scheduled restart",
                     line, re.I)]
restarts = unit.get("NRestarts")
if not journal.strip() or "unavailable" in journal[:400]:
    gate("JOURNAL", "n/a", "no journal came back -- see journal.txt")
elif hits or (restarts and restarts.isdigit() and int(restarts) > 0):
    gate("JOURNAL", "WARN",
         "%d line(s) worth reading, %s restart(s) recorded by systemd"
         % (len(hits), restarts))
    for line in hits[:5]:
        warnings.append("journal: %s" % line.strip()[:160])
else:
    gate("JOURNAL", "PASS", "no OOM, no restart, no traceback in the tail")

# -- thermals ---------------------------------------------------------------
def thermals(header):
    block = env.split(header, 1)
    if len(block) < 2:
        return []
    return [(name, int(milli) / 1000.0) for name, milli in
            re.findall(r"^\S+ (\S+) = (-?\d+) milli-degC$",
                       block[1].split("\n== ", 1)[0], re.M)]
after = thermals("== thermals (after the benchmark) ==")
if after:
    hottest = max(after, key=lambda row: row[1])
    lines.append("")
    lines.append("thermals after the run: hottest %s at %.1f C"
                 % (hottest[0], hottest[1]))
    # The Nano throttles around 87 C and shuts down above 97 C; a board that
    # is already hot after three picks will be slower after an hour of them.
    if hottest[1] >= 80.0:
        warnings.append("%s reached %.1f C: the Nano throttles near 87 C, so a "
                        "sustained shift will be slower than this record."
                        % (hottest[0], hottest[1]))

swap = re.search(r"^Swap:\s+(\d+)", env, re.M)
if swap and int(swap.group(1)) == 0:
    warnings.append("the board has no swap: a transient peak is an OOM kill "
                    "rather than a slow cycle (runbook, Resource budget).")

# -- the page ---------------------------------------------------------------
lines.append("")
if warnings:
    lines.append("worth acting on")
    for warning in warnings:
        lines.append("  - %s" % warning)
    lines.append("")
lines.append("results/bench/board_nano640.json is a Jetson Nano 4 GB measured "
             "at MAXN and")
lines.append("headless (2.6-2.7 s a pick, 624 MB peak): the POSE and CYCLE "
             "lines above are")
lines.append("read against it. The emulated and x86 records are the same poses "
             "on other")
lines.append("machines, not board timings. Thresholds and what to do about "
             "each line:")
lines.append("deploy/jetson-nano/ACCEPTANCE.md")

page = "\n".join(lines)
with open(os.path.join(out, "summary.txt"), "w") as handle:
    handle.write(page + "\n")
print(page)
sys.exit(1 if "FAIL" in gates else 0)
SUMMARY_PY_EOF
)

step "summary"
SUMMARY_RC=0
"$PY" - "$OUT" "$TARGET" "$PREFIX" "$BUDGET_LOW_S" "$BUDGET_HIGH_S" \
    "$MEMORY_MAX_MB" "$ACCEPT_SCORE" "$PICK_SCORE" \
    "$CMP_BOARD_RC" "$CMP_EMULATED_RC" "$CMP_NATIVE_RC" \
    "$LOCAL_COMMIT" "$BENCH_OK" "$BOARD_REFERENCE_S" "$CYCLE_REGRESSION_FACTOR" \
    <<PY || SUMMARY_RC=$?
$SUMMARY_PY
PY

# ------------------------------------------------------------------ cleanup

printf '\n'
if [ "$CLEAN" = 1 ]; then
    step "removing the board-side work directory"
    printf 'on %s: rm -rf %s\n' "$TARGET" "$REMOTE_WORK"
    printf 'this is the only path this script will remove, and it is inside %s\n' \
        "$PREFIX"
    CLEAN_SH=$(cat <<'CLEAN_SH_EOF'
set -u
prefix="$1"; work="$2"
resolved="$(readlink -f "$work" 2>/dev/null || echo "$work")"
case "$resolved" in
    "$prefix"/?*)
        rm -rf -- "$resolved"
        printf "removed %s\n" "$resolved" ;;
    *)
        printf "REFUSED: %s resolves to %s, which is outside the install prefix %s\n" \
            "$work" "$resolved" "$prefix" >&2
        exit 1 ;;
esac
CLEAN_SH_EOF
)
    remote_bash "$CLEAN_SH" "$PREFIX" "$REMOTE_WORK" </dev/null \
        || printf 'WARNING: the work directory is still on the board at %s\n' \
                  "$REMOTE_WORK" >&2
else
    printf 'The board still holds %s (board.json and nothing else).\n' "$REMOTE_WORK"
    printf 'Re-run with --clean to remove it; that is the only path this script\n'
    printf 'will delete, and it refuses anything outside %s.\n' "$PREFIX"
fi

printf '\nartefacts in %s:\n' "$OUT"
ls -1 "$OUT"
printf '\nsend the whole directory back -- summary.txt is the page to read first.\n'

exit "$SUMMARY_RC"
