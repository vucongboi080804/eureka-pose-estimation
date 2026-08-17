# Bringing the cell up on a Jetson Nano 4 GB (JetPack 4.6)

The camera hands one registered RGB-D frame to **PoseService**
(`deploy/pose_service/`), a long-lived process that holds the CAD cloud and
the segmenter in memory and answers `POST /v1/estimate` with the ranked poses
for that frame; the cell reads the top pose, compares its `score` against the
accept gate, and either grabs or rescans. Everything runs on the board over
`127.0.0.1` — no network, no cloud, no GPU required. This file is the order
in which a board goes from a freshly flashed SD card to a cell that picks.

Target: the original Jetson Nano developer kit — 4x Cortex-A57, 128-core
Maxwell GPU, 4 GB shared RAM, JetPack 4.6.x (L4T r32.7, Ubuntu 18.04,
glibc 2.27, CUDA 10.2). The stock Python is 3.6, which ultralytics no longer
supports, so the runtime goes into a **Python 3.8** venv. Everything except
the CUDA torch wheel is on PyPI as `cp38 aarch64` wheels that install on
glibc 2.27 (`requirements-jetson-nano.txt`; open3d must be 0.18.0 — 0.19
ships no aarch64 wheel at all).

---

## Bring-up

### 0. Board prep

Flash JetPack 4.6.x (SD-card image or SDK Manager) and boot once to finish
the OEM setup. Then, before anything else:

```bash
sudo nvpmodel -m 0        # MAXN: all 4 A57 cores, 10 W
sudo jetson_clocks        # pin CPU/GPU/EMC clocks -- timings are meaningless without it
swapon --show             # JetPack enables zram by default; confirm it is there
free -m                   # this is the memory budget the service has to live inside
```

A cell board should boot headless — `sudo systemctl set-default multi-user.target`
gives back the RAM the desktop session holds. If the board keeps the desktop,
add file swap on top of zram (see *Resource budget*).

The Nano has no RTC battery. Off the network its clock restarts at the last
shutdown time, so wall-clock timestamps are not trustworthy; anything that
measures durations must use a monotonic clock, and `sudo apt install
fake-hwclock` at least keeps the journal ordered.

### 1. Python 3.8

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-distutils \
                        libgl1 libglib2.0-0 libgomp1
```

`libgl1`/`libglib2.0-0`/`libgomp1` are load-time dependencies of the opencv
and open3d wheels; nothing is rendered on screen.

### 2. Wheelhouse (air-gapped board)

Build it **inside the aarch64 image**, which resolves natively as the board
would, and against the lock file rather than the loose one:

```bash
docker run --rm --platform linux/arm64 -v "$PWD/wheelhouse-nano:/wh" \
    --entrypoint pip pose-est:nano \
    download -r deploy/jetson-nano/requirements-jetson-nano.lock.txt -d /wh
```

Do **not** reach for `pip download --platform manylinux2014_aarch64` on an
x86 host. `--platform` is a wheel-tag filter, not a description of the
target: environment markers still evaluate against the build host, so
torch's `nvidia-*; platform_machine == "x86_64"` pins fire and the download
fails on versions PyPI no longer carries; and any wheel whose tag is not in
the list is **silently walked back** to an older version instead of being
reported — which is how `polars` falls to the 0.19.x line and produces a
message that reads as though ultralytics were incompatible with Python 3.8.
Two wheels in this stack cannot be expressed by any manylinux list at all:
`polars` is `manylinux_2_24_aarch64` and `torchvision` carries the bare
`linux_aarch64` tag.

Carry `wheelhouse-nano/` over with the repo and `weights/`, then on the board:

```bash
sudo install -d -o pose -g pose /opt/pose-estimation     # see step 5 for the user
cd /opt/pose-estimation                                  # repo unpacked here
python3.8 -m venv .venv
.venv/bin/pip install --no-index --find-links wheelhouse-nano \
    -r deploy/jetson-nano/requirements-jetson-nano.lock.txt
```

With internet on the board, `PYTHON=python3.8 ./setup.sh --jetson-nano` does
the same thing in one command.

### 3. Weights and CAD model

The service reads three files that are **not** produced at run time:

| File | Size | What it is |
| --- | --- | --- |
| `weights/part-seg.pt` | 56 MB | YOLO11l-seg fine-tuned on the 20 train scenes |
| `weights/part-seg-synthetic.pt` | 45 MB | YOLO11m-seg on 1140 synthetic renders — *off* on the Nano, see below |
| `model/3d_model.ply` | — | the CAD part, sampled into the registration cloud at startup |

Copy `weights/` from the repo and `model/` from the release folder into
`/opt/pose-estimation/`. A missing or unreadable path is a startup failure,
not a run-time one: the service refuses to come up rather than answering
requests it cannot serve.

### 4. Config

`deploy/jetson-nano/config.nano.json` is the board profile: one segmenter
(`extra_seg_weights: null`), `seg_imgsz` 640, `pick` on, `accept_score` 0.7,
`max_concurrency` 1, `omp_threads` 4, bound to `127.0.0.1:8080`. That is the
configuration `analysis/nano_profile.md` prices as the board's stop, and the
one every service number in this file was measured on; check the paths match
where you unpacked, and leave the rest alone until the board has been
measured — the knobs and what they cost are in *Resource budget*.

**One open call before the board is committed to.** `analysis/edge_model.md`
measures a 2.84 M-parameter `weights/part-seg-nano.pt` trained at 640 and
finds it indistinguishable from the 27.62 M `part-seg.pt` on end-to-end pose
accuracy (AR 0.837 vs 0.838, top-1 1.000 for both) for 16.9x less weight and
a 5.8x faster forward pass on CPU. If that holds, the board profile becomes
`"seg_weights": "weights/part-seg-nano.pt"` — a one-line change, no other
setting moves. It is not the default here because the service and emulation
numbers below were taken on `part-seg.pt`, and a profile whose documented
latency and memory belong to a different weight is worse than a slower one
that is honest. Switch it, then re-run step 8 on both sides.

The service validates the whole file before it reads a single weight, so a
typo, a missing `.pt`, or an `accept_score` above `pick_score` (a cell that
would never pick) fails the launch instead of the first cycle. Every setting
is hashed into a short digest that comes back with every pose, which is how
a pick is traced to the configuration that produced it. The digest covers the
*resolved absolute* paths, so a board that unpacks the release somewhere else
carries its own digest; what matters operationally is that it does not change
between two picks. Single values can be overridden without editing the file —
`POSE_SEG_IMGSZ=960`, `POSE_HOST=0.0.0.0` — which is what
`/etc/default/pose-service` is for, and `--host`/`--port` on the command line
override both, for a second instance beside the cell's own.

### 5. systemd unit

```bash
sudo useradd --system --home-dir /var/lib/pose-service --shell /usr/sbin/nologin pose
sudo chown -R pose:pose /opt/pose-estimation
sudo install -m 0644 deploy/jetson-nano/pose-service.service \
    /etc/systemd/system/pose-service.service
sudo systemctl daemon-reload
sudo systemctl enable --now pose-service
journalctl -u pose-service -f
```

The unit runs as `pose`, restarts on failure, caps its own memory and gives
up after five crashes in five minutes — a bad config or a missing weight
file must leave the unit in `failed`, loudly, instead of restarting forever.
Per-site overrides (a different config path, a different `OMP_NUM_THREADS`)
belong in `/etc/default/pose-service`, not in the unit.

### 6. Health check

```bash
curl -s http://127.0.0.1:8080/healthz          # or:
.venv/bin/python -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8080/healthz').read())"
```

The service binds its socket before it even imports the estimator, so
`/healthz` is the readiness signal and the open port is not: from the first
moment it answers, it answers `503` with `"status": "starting"`, then
`"loading"`, then `200` once the CAD cloud and the segmenter are in memory
and the warmup inference has run. **The cell must poll it until ready before
the first pick** — that is seconds on a desktop and minutes on four A57-class
cores. Anything that scripts it can poll with the shipped client instead,
which needs no `curl`:

```bash
until .venv/bin/python -m deploy.pose_service.client health >/dev/null 2>&1
do sleep 2; done            # it prints the 503 body on each miss otherwise
.venv/bin/python -m deploy.pose_service.client health
```

### 7. First pick

Before wiring the camera SDK in, prove the whole chain on a recorded scene —
this is the check to re-run after any recalibration or any board swap:

```bash
.venv/bin/python deploy/pick_demo.py --root /path/to/release --split test --scene 000001
```

`pick_demo.py` runs the same code path in-process (arrays in, grasp pose
out). To prove the *service* rather than the pipeline, send the same frame
over HTTP with the shipped client (`--help` for its exact flags; the
request/response contract itself is `deploy/pose_service/schema.py`):

```bash
.venv/bin/python -m deploy.pose_service.client estimate \
    --scene /path/to/release/test/000001
```

It prints `gate=pick`, one pose and a `score` around 0.83, and exits 0 (2 is
`rescan`, 1 a failed request). The client's own watchdog defaults to 120 s;
a cold board, or the emulated run below, can take longer than that on the
first frame, and the client then says so and tells you to raise `--timeout`. Pick mode commits to the first instance that
clears `PICK_SCORE`, which is not usually `submission.json`'s top-ranked one,
so the check is that the returned pose *coincides with an instance the
submission holds* for that scene: on the x86 development machine test/000001
came back 0.03 mm and 0.28 deg from `submission["000001"][4]`, and test/000015
0.09 mm and 0.48 deg from `submission["000015"][3]`.

### 8. Benchmark, and compare against the emulated baseline

The board is the only place the timing question gets a real answer. The
emulated numbers are *not* a board estimate — on the shipped board profile
qemu-user inflates one pick 71x overall and does it very unevenly: **541x on
the segmenter against 14x on registration**. Note what happens when the
weight changes: on the old desktop profile the same comparison read 172x
overall / 307x segmenter, and swapping in the 9x smaller nano model *halved*
the overall ratio while nearly *doubling* the segmenter's, because at 0.3 s
native there is no longer a heavy registration stage for the segmenter to
hide behind. A single scaling factor carried from one profile to another
would be wrong twice over. The board run replaces these numbers; it does not
confirm them.

The committed baseline is `results/bench/emulated_nano640.json` (board
limits, one scene) against `results/bench/native_nano640.json` (this machine,
three scenes, three repeats), both on `part-seg-nano.pt` at 640.

```bash
# on the board, quiet, jetson_clocks on:
mkdir -p out_bench
.venv/bin/python deploy/jetson-nano/bench.py --root /path/to/release --split test \
    --scenes 000001 000002 000015 --pick --repeat 3 \
    --seg-model weights/part-seg-nano.pt --extra-seg-model "" --imgsz 640 \
    --out out_bench/board.json \
    --note "board, 2 GB swap, no desktop session"
```

`--profile nano` is the default (4 threads, CPU only) and is what makes a
desktop or emulated run comparable at all. The three weight arguments are not
optional: `bench.py` defaults to the shipped desktop configuration
(`part-seg.pt`, both segmenters, 960), so a run without them benches a
27 M-parameter model while `config.nano.json` serves a 2.84 M one — and the
difference would read as hardware when the board record is diffed against the
emulated baseline. `--seg-model weights/part-seg-nano.pt --extra-seg-model ""
--imgsz 640` is what the board actually runs.
`--repeat 3` reports min and median, because RANSAC is stochastic. The
record carries the host fingerprint, the pins, the config and the peak RSS
next to the times — that is what makes two files comparable.

The emulated baseline comes from an x86 host, which applies the board's
limits to the same command:

```bash
deploy/jetson-nano/emulate.sh --check     # prints the command and the caveat
deploy/jetson-nano/emulate.sh             # -> out_bench/emulated.json
```

Then diff the two — emulated first, board second:

```bash
.venv/bin/python deploy/jetson-nano/compare_bench.py \
    out_bench/emulated.json out_bench/board.json --md
```

It gates on the poses agreeing within 2 mm and 2 degrees (exit non-zero if
they do not) and reports the speed ratio *without* a verdict, because qemu
emulates the instruction set and not a Cortex-A57: no A57 caches, no LPDDR4
bandwidth, no thermal envelope. Same answer is the claim emulation can
support; same speed is not.

Both sides must be asked the same question. `emulate.sh` defaults to the
board profile (one segmenter, `--imgsz 640`) so that it matches
`config.nano.json` and the board command above; `EXTRA_SEG` and `IMGSZ`
override it, and `compare_bench.py` refuses to compare two records that
disagree on `pick`, `imgsz` or the segmenter set.

---

## Resource budget

4 GB is shared between the CPU, the GPU and the display. The pipeline's
appetite is dominated by the registration working set, not by the network:

| Item | Cost | Where the number comes from |
| --- | --- | --- |
| One pipeline worker, both segmenters, CPU only | 1.58 GB peak RSS | `analysis/runtime.md` (x86, `/usr/bin/time -v`) |
| Same, inside the CPU container | 1.8 GB peak | `deploy/OFFLINE.md` |
| Segmenter weights on disk | 56 MB + 45 MB | `weights/` |
| Segmenter activations | scale with `imgsz`<sup>2</sup> | `src/detect_seg.py:masks_from_model` |
| Board profile (one segmenter, 640), 4 cores pinned | 1.31 GB peak RSS | `analysis/nano_profile.md` (x86, 4 pinned cores) |
| **Same profile on the aarch64 stack** | **0.74 GB peak RSS** | `results/bench/emulated_nano640.json` — emulated, but the board's own torch 2.4.1 / open3d 0.18 / numpy 1.24 rather than x86's much newer set, so this is the closer estimate |
| The service holding that profile across frames | 0.95 GB after load, 1.71 GB RSS | measured here (x86, `/healthz` after two frames) |
| The same on the Nano | *unmeasured — the board answers this* | |

**Why one worker.** One worker already needs ~1.6 GB before the desktop,
the camera buffers and the page cache; two would swap, and swapping to an SD
card costs more than the second worker returns. Registration is where the
time goes and it is already OpenMP-parallel across the four cores, so the
board's parallelism belongs *inside* one scene (`omp_threads: 4`), not across
scenes. On a cell there is only ever one frame in flight anyway, which is what
`max_concurrency: 1` says in the config and `WORKERS=1` says for the batch
image.

**Swap.** JetPack enables zram by default (~half of RAM, striped across the
cores); that is enough for a headless board. If the desktop stays up, add
file swap so a transient peak degrades instead of getting killed
(2 GB on top of 4 GB of RAM is the envelope `emulate.sh` reproduces):

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile \
    && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Swap is insurance, not headroom: a cycle that actually swaps blows the
latency budget. `MemoryMax` in the unit is the real guard.

**The two nano knobs**, in the order to reach for them:

1. **One segmenter** (`extra_seg_weights: null`, the shipped board profile).
   Drops 45 MB of weights, one full network pass per frame and its
   activations. Costs recall: at the board profile's `conf` 0.25 the single
   YOLO11l segmenter reaches AR 0.829 against 0.851 for the pool, at
   precision 0.866 instead of 0.767 and top-1 still 1.000
   (`analysis/nano_profile.md`, leave-scenes-out CV). Top-1 is the number a
   picking cell is actually governed by, so this is a cheap trade here and an
   expensive one for the assignment metric. What it costs the cell is pick
   margin: the worst CV scene's best score falls from 0.854 to 0.812, so a
   single-segmenter cell should expect the occasional fall-through to the
   geometric safety net — the expensive path.
2. **Lower `seg_imgsz`** (960 is what the weights were fine-tuned at and
   what the submission uses). Masks are resized back to the full frame
   either way, so lowering it costs segmentation detail and nothing else,
   while activation memory falls with the square. Measured: 960 -> 640 costs
   0.002 AR with one segmenter and 0.007 with two, both inside the ±0.005
   draw-to-draw band, for 0.13 GB less peak RSS — so `config.nano.json`
   ships **640**, the knee `analysis/nano_profile.md` recommends. Below 640
   is not worth measuring: the segmenters are ~2 % of scene time to begin
   with.
   *(`scripts/run_pipeline.py` has no `--imgsz` flag — there the constant is
   `SEG_IMGSZ` in `src/detect_seg.py`. `bench.py` and the service both take
   it as a setting.)*

**GPU.** JetPack 4.6 ships CUDA torch for Python 3.6 only; for 3.8 use a
community CUDA-10.2 build (e.g. Qengineering's *PyTorch-Jetson-Nano* wheels,
torch 1.13 / 2.0 for aarch64 cp38) *instead of* the `torch`/`torchvision`
lines of the requirements file — nothing in this repo depends on the torch
build beyond ultralytics' inference call. It is worth little: the GPU
accelerates only the segmenters, which are ~2 % of scene time
(`analysis/runtime.md`), and RANSAC/ICP stay on the CPU. TensorRT export is
not an option here (the JetPack 4.6 TensorRT Python bindings are 3.6-only).
Not exercised in this repository — no board was available.

---

## Operating the cell

**The gate.** `score` = segmenter confidence x depth verification. Gate at
**0.7**: on cross-validated train scenes that keeps precision 0.99 at 5 mm
and 0.99 at 10 mm, at recall 0.786; 0.6 buys recall back (0.872 at 10 mm)
for precision 0.94 at 5 mm, and below 0.4 a prediction is wrong more often
than right
(`analysis/score_calibration.md`). Read it as a rank, not a probability —
it is only roughly calibrated (ECE 0.143), and its ranking is what is
trustworthy (AUROC 0.96 against correct-at-5-mm).

Note the two thresholds are different on purpose: **pick mode stops the
sweep at the first pose scoring >= 0.8** (`PICK_SCORE`), while the cell
accepts at 0.7. So a normal cycle returns one pose well above the gate; when
the sweep finishes without reaching 0.8 you get whatever it found, and the
gate is what decides.

**One cycle.** The service does the gating: every frame comes back with
`gate` = `pick` or `rescan`, so the cell reads one field rather than
re-implementing the threshold. `pick` means grab `poses[0]`; then rescan,
because a bin changes after every pick and ranking the whole scene is wasted
work (pick mode is 7-17x less work than a full sweep). Pin `schema_version`
in the cell and log `config_digest` with every grasp — that pair is what
lets you prove afterwards which weights and thresholds moved the robot.

**Nothing verifies.** `gate` = `rescan` — an empty `poses`, or a top score
under 0.7 — is a normal outcome, not an error, and so is a frame that failed
outright (`error` set, `poses` empty, still `rescan`): one bad frame must
look to the cell exactly like a bin it cannot pick from. Rescan, and if the
second scan agrees, shake or re-orient the bin and scan again. Parts standing on end, buried, or leaning
steeply are the population that does not verify; shaking is the cheapest fix
available to the cell. Count these — the *rate* is the signal, not the event.

**Watchdog.** Wrap each cycle in a client-side timeout (the service has its
own `request_timeout_s`, but the cell must not depend on the service being
alive to time out). A cycle that exceeds the board's measured p99 by a wide
margin is a hang: abandon the frame, rescan, and let systemd restart the
service if `/healthz` also stops answering. One bad frame must never stop the
cell — the pipeline already isolates per-scene exceptions into an empty
result rather than a crash.

**What to log per cycle.** The response already carries it: `n_proposals`
(the count that collapses first when the scene leaves the training domain),
`gate`, the top `score` with its two factors, and `timings_ms`. `/metrics`
exposes the running counters; the journal (`journalctl -u pose-service`)
carries the per-request lines.

**Domain shift or mechanics?** Both look like "picks are failing". They
separate on the numbers above:

| Symptom | Reading |
| --- | --- |
| `n_proposals` collapses; scores drop across *all* bin states; low `seg_confidence` with a *healthy* `depth_verification` | The segmenter has left its training domain — new lighting, a new tray, a repainted part — while geometry still confirms the parts. Re-check illumination first, then collect site scenes and re-fine-tune. |
| Confident masks (`seg_confidence` high) that the depth map refuses (`depth_verification` low) | Not the segmenter. Geometry is objecting: parts are where the camera cannot see them, or the depth stream disagrees with the colour frame. Worth logging the frame before the bin is disturbed. |
| Proposals and scores are normal, the gripper still misses | Not perception. Hand-eye calibration, gripper wear, or the pick strategy. A constant offset across every scene is calibration by definition. |
| Depth mostly empty, scores near zero everywhere | Sensor: dirty or occluded optics, exposure, or the depth stream drifting out of registration with the colour image. |
| Scores fine, occasional 180-degree wrong grasp | The part's near-symmetry. The flip disambiguation is what handles this; a rise in the rate is worth a look at `analysis/failure_analysis.md`. |

---

## Docker on the board

`deploy/jetson-nano/Dockerfile` (Ubuntu 18.04 aarch64 + deadsnakes 3.8 + the
pinned wheels) serves both modes. Do not build it on the Nano — build it on
an x86 host with `--platform linux/arm64` and `docker save` it over.

The image is built *from the tree*, so rebuild it after any change under
`deploy/`, `src/` or `weights/`; with a warm cache that is under a minute.

```bash
docker build --platform linux/arm64 -f deploy/jetson-nano/Dockerfile -t pose-est:nano .

# the service (the cell): host networking, so 127.0.0.1:8080 is the board's
docker run --rm --network host -v /path/to/release/model:/app/model:ro \
    --entrypoint python pose-est:nano \
    -m deploy.pose_service.server --config /app/deploy/jetson-nano/config.nano.json

# the batch pipeline (a split, offline): the default entrypoint
docker run --rm --network none --no-healthcheck \
    -v /path/to/release:/data:ro -v "$PWD/out:/out" pose-est:nano /data test /out
```

Single settings can be tuned without a second config file — the service
overlays `POSE_*` environment variables on top of it, so
`-e POSE_HOST=0.0.0.0 -p 8080:8080` replaces `--network host` where host
networking is not wanted.

`--no-healthcheck` on the batch run: the image's `HEALTHCHECK` probes the
service port and would otherwise mark a perfectly healthy batch container
unhealthy. For the GPU inside a container use NVIDIA's `l4t-base:r32.7.1`
image with the community torch wheel and the same pip lines — not provided
here because it cannot be tested off-board.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Service killed mid-request; `journalctl -k` shows the OOM killer, or `systemctl status` shows `oom-kill` | 4 GB shared, `MemoryMax` hit, or a second worker | `max_concurrency: 1`, `extra_seg_weights: null`, drop `seg_imgsz`; boot headless; add file swap. Confirm with `systemctl show pose-service -p MemoryCurrent` under load. |
| `ERROR: Could not find a version that satisfies the requirement open3d` | open3d 0.19+ publishes no aarch64 wheel | Stay on **0.18.0**. Do not "upgrade" the pin; there is nothing to upgrade to. |
| `pip` tries to *build* a wheel on the board, or the venv install takes an hour | A pin resolved to an sdist, or the wheelhouse was built with `--platform` on an x86 host and is missing wheels | Install with `--no-index --find-links` from a wheelhouse built inside the aarch64 image (step 2). A `--platform` wheelhouse silently omits `polars` and `torchvision`. |
| `ERROR: Could not find a version that satisfies the requirement polars>=0.20.0` | The wheelhouse was built with `--platform`, which excluded polars' `manylinux_2_24_aarch64` wheel and walked the resolver back to 0.19.x | Not an ultralytics/Python 3.8 incompatibility — 8.4.120 installs fine on 3.8. Rebuild the wheelhouse inside the image. |
| `No matching distribution found for nvidia-cuda-nvrtc-cu12==12.1.105` | `pip download` run on an x86 host: torch's CUDA markers evaluate against the build machine, not the target | Same fix — resolve inside the aarch64 image, where `platform_machine` is `aarch64` and the markers are false. |
| First request after start hangs or takes far longer than the rest | Model load plus lazy first inference | Expected. `/healthz` stays not-ready until warmup completes — poll it, do not race it. If it is still slow after ready, `warmup` is off in the config. |
| Startup pauses ~5 s, or logs a DNS/`pip install` attempt | ultralytics probing the network | `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=0` and `yolo settings sync=False` — all three are set by the unit and the image; a hand-made venv needs them too. |
| `torch.cuda.is_available()` is `False` | The PyPI aarch64 torch wheel is CPU-only; JetPack's CUDA torch is Python 3.6 | Expected on the shipped pins, and cheap (the GPU serves ~2 % of scene time). For the GPU, swap in a community cp38 CUDA-10.2 wheel — untested here. |
| `ImportError: libGL.so.1` / `libgomp.so.1` | Missing shared libraries the wheels link against | `sudo apt-get install libgl1 libglib2.0-0 libgomp1` |
| Two benchmark runs disagree wildly; timestamps go backwards | No RTC battery, or clocks unpinned | `jetson_clocks` + `nvpmodel -m 0` before timing; `fake-hwclock`; compare monotonic durations, never wall-clock timestamps. |
| Poses correct offline, wrong through the service | Intrinsics or depth scale wrong on the live path | The service needs the *colour* camera's `K` and the `depth_scale` that turns ticks into metres, with depth registered to the colour image — same contract as `deploy/live_adapter.py`. Re-run `pick_demo.py` on a recorded scene to separate the camera from the pipeline. |

---

## Verified here vs. not verified on the board

- **Verified (the image, which is the validated path).** `pose-est:nano`
  resolves and installs the whole stack natively on aarch64 / glibc 2.27 /
  CPython 3.8.0 — torch 2.4.1, torchvision 0.19.1, open3d 0.18.0,
  ultralytics 8.4.120, polars 1.8.2, opencv 4.10.0.84 — and
  `requirements-jetson-nano.lock.txt` is that resolution, extracted from the
  image after it ran the pipeline. A dry-run resolve of the loose
  requirements file inside the image agrees with the lock on all 64 packages
  today, but only 9 are pinned there, so the lock is what the board should
  install.

- **Not verified: the bare-metal wheelhouse as previously documented.** The
  old `pip download --platform …` line does not produce a wheelhouse from an
  x86 host at all (see step 2). The Docker path is the one with evidence
  behind it.

- **Verified (x86 development machine).** The pipeline runs on Python 3.8
  with open3d 0.18 (x86 container: test/000001 and 000002 give the same 9
  and 5 poses as the submission).

- **Verified (x86, natively, against this config).** The service comes up on
  `config.nano.json`, reports ready 6.0 s after launch (3.9 s load, 1.0 s
  warmup) and answers `/v1/estimate` for test/000001 in 0.55 s and for the
  dense test/000015 in 1.0 s, both `gate=pick`. Each returned pose coincides
  with an instance `submission.json` holds for its scene — 0.03 mm / 0.28 deg
  and 0.09 mm / 0.48 deg — although not with the submission's *top-ranked*
  one, because pick mode stops at the first instance over `PICK_SCORE`.
  `/healthz`, `/readyz` and `/metrics` answer; a truncated body, a
  non-string `scene_dir` and an empty body are 400; a scene folder missing
  `depth.png` is 422 with `poses: []`, `gate: rescan` and the error named;
  `SIGTERM` drains and exits 0. Resident memory settles at 1.18 GB
  (CPU-only; 1.70 GB when a CUDA device is visible, which the board's CPU
  torch never is).

- **Verified (arm64 image under qemu-user, the board's limits applied).**
  The image was rebuilt from this Dockerfile, so it carries `deploy/` and the
  service. `docker run --platform linux/arm64 --network none --cpus 4
  --memory 4g --memory-swap 6g` with the service as the entry point (the
  command in *Docker on the board*, minus `--network host`, which has nothing
  to reach on this machine): ready after
  **166 s** (51.0 s load + 115.2 s warmup), one pick for test/000001 in
  **111.8 s** (107.7 s of it the segmenter), service RSS **878 MB**, cgroup
  `memory.peak` **864 MB**. The pose is the native one to **0.02 mm /
  0.07 deg** (score 0.8235 vs 0.8250). `bench.py` inside the same container
  against a native record, both at the board profile:

  ```
  scene    qemu s  x86_64 s  ratio  poses  max mm  max deg  d score  output
  000001   20.9    0.3       71.2x  1/1    0.02    0.04     0.003    ok

  OUTPUT : AGREES on all 1 scene -- worst 0.02 mm / 0.04 deg (tolerance 2 mm / 2 deg)
  NOISE  : same machine, repeated -- worst 0.01 mm / 0.01 deg
  STAGES : io 5x, setup 19x, segmenter 541x, register 14x
  SPEED  : qemu takes 71.2x x86_64
  ```

  **The result is the agreement, not the speed.** The aarch64 build returns
  the same pose as x86 to 0.02 mm, which is the repeat-to-repeat noise of a
  single machine — so the port is proven and the cross-machine delta is
  indistinguishable from RANSAC. Memory is the one magnitude worth carrying:
  the aarch64 stack peaks at **0.74 GB against x86's 1.09 GB on the same
  profile**, and it is the board's own library versions, so the unit's
  `MemoryMax=2600M` has more headroom than the x86 estimate suggested. Time
  transfers not at all — 541x on the segmenter against 14x on registration
  cannot be one factor.

- **Not verified — the board answers these, and the runbook is built so it
  can.** Wall-clock and memory on the actual Nano; the CUDA-10.2 torch path
  on JetPack 4.6; memory headroom inside 4 GB with the desktop session
  running; the unit under the board's systemd 237 and cgroup v1, where
  `MemoryMax` becomes `memory.limit_in_bytes` — `systemd-analyze verify`
  here (on a newer systemd) reports nothing against the unit except that
  `/opt/pose-estimation/.venv/bin/python` does not exist on a development
  machine, which is the one thing only the board can satisfy; a real camera
  on the live path.

For the x86 and container recipes, and what the runtime touches on disk and
on the network, see `deploy/OFFLINE.md`.
