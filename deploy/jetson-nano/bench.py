"""Measure this pipeline where it will run, and record what it ran on.

The Jetson Nano numbers in ``deploy/jetson-nano/README.md`` are two
extrapolations: an x86 desktop, and a qemu-user emulation that took 259 s for
a pick the desktop does in 1.5 s. Neither is a board measurement, and a
timing without its hardware attached is a rumour. This script produces one
machine-readable record -- fingerprint, per-stage timings, peak RSS, and the
poses themselves -- from the identical command line on the desktop, under
emulation and on the board, so ``compare_bench.py`` can turn "how close was
the emulation" into a table.

    cd <repo> && .venv/bin/python deploy/jetson-nano/bench.py \
        --root . --split test --scenes 000001 000002 000015 --out bench.json

Emulated runs go through ``deploy/jetson-nano/emulate.sh``, which applies the
board's CPU and memory limits to the same command. The pose output is the
part that must agree between machines; the wall clock is the part that will
not, so the two are reported separately.

Stage timings come from wrapping the segmenter objects, not from editing
``src/``: the shipped ``detect_scene_hybrid`` runs unmodified, and every
forward pass it makes is charged to the segmenter stage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
import traceback

#: Bump when a field changes meaning. compare_bench.py reads older files by
#: treating every field as optional, so additions do not need a bump.
SCHEMA_VERSION = 1

#: The board: 4 Cortex-A57 cores, CPU torch (JetPack 4.6 ships no CUDA torch
#: for Python 3.8). Running the desktop under this profile is the closest
#: like-for-like an x86 machine can offer.
NANO_THREADS = 4

_REPO_MARKER = os.path.join("src", "scene_io.py")


def _repo_root() -> str:
    """Directory holding ``src/`` -- the checkout, or /app inside the image.

    ``emulate.sh`` bind-mounts this file into the aarch64 image rather than
    rebuilding it, so the path relative to ``__file__`` is the answer on a
    checkout and the working directory is the answer in the container.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for candidate in (here, os.getcwd()):
        if os.path.exists(os.path.join(candidate, _REPO_MARKER)):
            return candidate
    raise SystemExit("cannot locate the repository: no %s next to %s or in %s"
                     % (_REPO_MARKER, __file__, os.getcwd()))


def _read_text(path: str):
    """File contents, or None when it does not exist on this platform."""
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace").replace("\x00", "")
    except OSError:
        return None


def _cpu_model() -> str:
    """First human-readable CPU name /proc/cpuinfo offers.

    x86 calls it "model name", a Cortex-A57 under JetPack calls it "model
    name" too but some aarch64 kernels only fill "Processor" or "Hardware".
    """
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    for key in ("model name", "Model", "Processor", "Hardware", "CPU part"):
        match = re.search(r"^%s\s*:\s*(.+)$" % key, cpuinfo, re.M)
        if match:
            return match.group(1).strip()
    return "unknown"


def _total_ram_mb():
    meminfo = _read_text("/proc/meminfo") or ""
    match = re.search(r"^MemTotal:\s*(\d+) kB", meminfo, re.M)
    return round(int(match.group(1)) / 1024.0) if match else None


def _cgroup_budget() -> dict:
    """CPU and memory the container actually grants.

    Inside ``docker run --cpus 4 --memory 4g`` the kernel still reports the
    host's 20 cores and 32 GB through ``os.cpu_count`` and /proc/meminfo, so
    without this the emulated record would claim an envelope it never had.
    """
    budget = {"cpu_quota": None, "memory_limit_mb": None}
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")            # cgroup v2
    if cpu_max and cpu_max.split()[0] != "max":
        quota, period = cpu_max.split()[:2]
        budget["cpu_quota"] = round(int(quota) / int(period), 2)
    else:                                                     # cgroup v1
        quota = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period and int(quota.strip()) > 0:
            budget["cpu_quota"] = round(int(quota) / int(period), 2)
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read_text(path)
        if raw and raw.strip() != "max":
            limit = int(raw.strip())
            # cgroup v1 spells "unlimited" as a number near 2**63.
            if limit < (1 << 60):
                budget["memory_limit_mb"] = round(limit / 1e6)
            break
    return budget


def _emulation_evidence() -> list:
    """Signs that this aarch64 process is qemu-user, not silicon.

    qemu-user synthesises /proc/cpuinfo rather than passing the host's
    through, so the usual "the CPU says x86" trick does not work; what it
    cannot fake convincingly is the identity of a chip that does not exist.
    A real ARM part reports an implementer of 0x41 (ARM), 0x4e (NVIDIA) and
    so on, and a measured BogoMIPS; qemu reports implementer 0x00 and a
    constant 100.00, and advertises a feature set (SVE2 + SME2 + MTE) no
    shipped A57 has.

    Limits, stated here because a benchmark must not overclaim: this detects
    qemu-user only. A full-machine emulator (qemu-system) or a KVM guest
    looks like real hardware from inside and will report as native. The
    positive identification of the board does not depend on any of this --
    it comes from /proc/device-tree/model and /etc/nv_tegra_release, which
    only a Tegra provides.
    """
    evidence = []
    if any(key.startswith("QEMU_") for key in os.environ):
        evidence.append("QEMU_* in the environment")
    maps = _read_text("/proc/self/maps") or ""
    if "qemu" in maps.lower():
        evidence.append("qemu mapped into /proc/self/maps")
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    if re.search(r"^CPU implementer\s*:\s*0x0+$", cpuinfo, re.M):
        evidence.append("CPU implementer 0x00 (no such vendor)")
    if re.search(r"^BogoMIPS\s*:\s*100\.00$", cpuinfo, re.M):
        evidence.append("BogoMIPS exactly 100.00 (qemu constant)")
    return evidence


#: Where the board identity is read from inside a container. /proc/device-tree
#: is not visible in one, so the runbook bind-mounts the model file to this
#: path; without it a run on real hardware records itself as "not a Jetson",
#: which is the one thing a board record exists to prove.
CONTAINER_DEVICE_MODEL = "/etc/device-tree-model"


def describe_host() -> dict:
    """Everything needed to decide whether two bench files are comparable."""
    device_model = _read_text("/proc/device-tree/model") \
        or _read_text(CONTAINER_DEVICE_MODEL)
    tegra_release = _read_text("/etc/nv_tegra_release")
    is_jetson = bool(device_model and "jetson" in device_model.lower()) \
        or tegra_release is not None
    evidence = _emulation_evidence()
    host = {
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity") else None,
        "total_ram_mb": _total_ram_mb(),
        "cgroup": _cgroup_budget(),
        "jetson": is_jetson,
        "device_tree_model": device_model.strip() if device_model else None,
        "nv_tegra_release": tegra_release.splitlines()[0].strip()
                            if tegra_release else None,
        "emulation_evidence": evidence,
        "python": platform.python_version(),
        "versions": {},
        "torch_cuda": False,
        "cuda_device": None,
    }
    if is_jetson:
        host["platform_kind"] = "jetson"
    elif evidence:
        host["platform_kind"] = "qemu-user"
        # Under qemu-user /proc/cpuinfo describes a machine that does not
        # exist; keep the string but do not let a reader take it for silicon.
        host["cpu_model"] = "%s (synthesised by qemu)" % host["cpu_model"]
    else:
        host["platform_kind"] = "native"
    for name in ("torch", "open3d", "ultralytics", "cv2", "numpy", "scipy",
                 "trimesh"):
        module = sys.modules.get(name)
        host["versions"][name] = getattr(module, "__version__", None) \
            if module else None
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        host["torch_cuda"] = True
        host["cuda_device"] = torch.cuda.get_device_name(0)
    return host


def _git_commit(repo_root: str):
    """Short HEAD, or None. The image carries no .git, so emulate.sh passes
    the checkout's commit in BENCH_GIT_COMMIT instead."""
    override = os.environ.get("BENCH_GIT_COMMIT")
    if override:
        return override.strip()
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _peak_rss_mb() -> float:
    """High-water RSS of this process, in MB.

    ``ru_maxrss`` is a high-water mark: it never falls, so it belongs to the
    run rather than to the scene that happened to be running when it was
    read. On Linux the unit is kibibytes. Reaped children are excluded on
    purpose -- this harness is single-process, so counting them would add a
    subprocess peak that never coexisted with the parent's; the number that
    matters on a 4 GB board is what one worker holds at once.
    """
    kibibytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(kibibytes / 1024.0, 1)


class SegmenterTimer:
    """A YOLO model that keeps the bill for its own forward passes.

    ``detect_scene_hybrid`` calls the segmenters internally, so the only way
    to split segmenter time from registration time without editing ``src/``
    is to hand it models that time themselves. The wrapper forwards the call
    untouched, which keeps the benchmark on the shipped code path.
    """

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.seconds = 0.0
        self.proposals = 0

    def reset(self) -> None:
        self.seconds = 0.0
        self.proposals = 0

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        results = self.model(*args, **kwargs)
        self.seconds += time.perf_counter() - start
        first = results[0]
        if getattr(first, "masks", None) is not None:
            self.proposals += len(first.masks.data)
        return results


def _apply_profile(profile: str, threads) -> dict:
    """Pin the thread and device budget before the heavy imports.

    Order matters and is the same trap ``scripts/run_pipeline.py`` documents:
    libgomp reads OMP_NUM_THREADS once, when Open3D loads, and torch reads
    CUDA_VISIBLE_DEVICES once, when it initialises. Setting either after the
    import is a no-op that would silently invalidate the comparison.
    """
    if threads is None:
        threads = NANO_THREADS if profile == "nano" else (os.cpu_count() or 1)
    config = {"profile": profile, "threads": threads,
              "cpu_only": profile == "nano"}
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    if profile == "nano":
        # The board runs CPU torch, so a desktop GPU would measure a machine
        # the Nano is not.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return config


def _run_once(root: str, split: str, scene_id: str, model_cloud,
              segmenters: list, pick: bool, deps, imgsz=None) -> dict:
    """One timed pass over one scene, on the shipped code path.

    The four stages sum to the wall time exactly: reading the scene off
    disk (the stage a board's SD card makes expensive), preparing the
    estimator, the segmenter forward passes, and everything the detector
    does with the proposals -- back-projection, RANSAC, ICP, flips, the
    depth verifier, the polish, and the geometric safety net when it fires.
    """
    load_scene, part_pixel_mask, PoseEstimator, detect_scene_hybrid = deps
    for timer in segmenters:
        timer.reset()

    start = time.perf_counter()
    scene = load_scene(root, split, scene_id)
    after_io = time.perf_counter()
    estimator = PoseEstimator(model_cloud, scene.depth, scene.K,
                              part_mask=part_pixel_mask(scene.rgb))
    after_setup = time.perf_counter()
    # imgsz is passed only when asked for, so the default run is exactly the
    # shipped call and the harness does not depend on the knob existing.
    extras = {"imgsz": imgsz} if imgsz else {}
    found = detect_scene_hybrid(
        scene, estimator, segmenters[0],
        extra_model=segmenters[1] if len(segmenters) > 1 else None,
        pick=pick, **extras)
    end = time.perf_counter()

    segmenter_s = sum(timer.seconds for timer in segmenters)
    poses = [{"R": [round(v, 9) for v in est.R.ravel().tolist()],
              "t": [round(v, 9) for v in est.t.tolist()],
              "score": round(est.submission_score, 4)}
             for est in found]
    return {
        "wall_s": round(end - start, 3),
        "stages_s": {
            "io": round(after_io - start, 3),
            "setup": round(after_setup - after_io, 3),
            "segmenter": round(segmenter_s, 3),
            "register": round(end - after_setup - segmenter_s, 3),
        },
        "proposals": {timer.name: timer.proposals for timer in segmenters},
        "n_poses": len(poses),
        "top_score": max((p["score"] for p in poses), default=0.0),
        "poses": poses,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _summarise(repeats: list) -> dict:
    """min and median of the successful repeats.

    Open3D's RANSAC is stochastic -- its OpenMP threads share one random
    engine, so a seed does not make it reproducible -- and the pose a scene
    yields can differ between draws on one machine. min is the machine's
    best case, median is what to quote; the spread between them is the noise
    a cross-machine comparison has to clear.
    """
    times = [r["wall_s"] for r in repeats]
    summary = {"wall_s_min": min(times),
               "wall_s_median": round(statistics.median(times), 3),
               "stages_s_min": {}}
    for stage in repeats[0]["stages_s"]:
        summary["stages_s_min"][stage] = min(r["stages_s"][stage]
                                             for r in repeats)
    return summary


def _parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".",
                        help="Release folder: the split directories and "
                             "model/3d_model.ply")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Scene ids to time; default is the whole split, "
                             "which is minutes on a desktop and hours under "
                             "emulation")
    parser.add_argument("--out", default="bench.json")
    parser.add_argument("--profile", choices=("nano", "desktop"),
                        default="nano",
                        help="nano: 4 threads, CPU only -- the board's "
                             "envelope, and what makes a desktop run "
                             "comparable to it. desktop: all cores, GPU if "
                             "present")
    parser.add_argument("--threads", type=int, default=None,
                        help="Override the profile's thread budget")
    parser.add_argument("--pick", action="store_true",
                        help="Deployment latency mode: stop each scene at "
                             "the first pose the cell would act on")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Timed passes per scene; RANSAC is stochastic, "
                             "so min and median are reported")
    parser.add_argument("--workers", type=int, default=1,
                        help="Only 1 is measured here (see below)")
    parser.add_argument("--seg-model", default=None,
                        help="Default: <repo>/weights/part-seg.pt")
    parser.add_argument("--extra-seg-model", default=None,
                        help="Second segmenter; pass an empty string to "
                             "measure the single-segmenter configuration "
                             "the 4 GB board may prefer. Default: "
                             "<repo>/weights/part-seg-synthetic.pt")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Segmenter input side. Default: whatever the "
                             "pipeline ships, which is what the weights "
                             "were trained at; lower it to measure the "
                             "memory/accuracy trade a 4 GB board may want")
    parser.add_argument("--timestamp", type=float, default=None,
                        help="UTC epoch seconds to stamp the record with; "
                             "default is now")
    parser.add_argument("--note", default="",
                        help="Free text carried into the record, e.g. "
                             "'board, 6 GB swap, no desktop session'")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    repo_root = _repo_root()
    sys.path.insert(0, repo_root)

    if args.workers != 1:
        raise SystemExit(
            "bench.py measures one worker, because the board runs one: at "
            "~1.6 GB RSS per worker a 4 GB Nano has room for nothing else, "
            "and a pool hides the per-stage split this file exists to "
            "report. For multi-worker throughput on a desktop use\n"
            "    scripts/run_pipeline.py --workers %d" % args.workers)
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    seg_weights = args.seg_model or os.path.join(repo_root, "weights",
                                                 "part-seg.pt")
    extra_weights = args.extra_seg_model
    if extra_weights is None:
        extra_weights = os.path.join(repo_root, "weights",
                                     "part-seg-synthetic.pt")
    ply = os.path.join(args.root, "model", "3d_model.ply")
    split_dir = os.path.join(args.root, args.split)
    # Fail at startup, not three scenes into a half-hour emulated run.
    for path in [ply, split_dir, seg_weights] + ([extra_weights]
                                                 if extra_weights else []):
        if not os.path.exists(path):
            raise SystemExit("missing: %s" % path)

    config = _apply_profile(args.profile, args.threads)
    config.update({
        "root": args.root, "split": args.split, "pick": args.pick,
        "repeat": args.repeat, "workers": 1,
        "seg_model": os.path.basename(seg_weights),
        "extra_seg_model": os.path.basename(extra_weights)
                           if extra_weights else None,
        "imgsz": args.imgsz,
    })

    # Imported here, after the profile is applied: see _apply_profile.
    from src.detect import part_pixel_mask
    from src.detect_seg import detect_scene_hybrid
    from src.model_cloud import load_model_cloud
    from src.register import PoseEstimator
    from src.scene_io import list_scenes, load_scene
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(config["threads"])
    deps = (load_scene, part_pixel_mask, PoseEstimator, detect_scene_hybrid)

    scene_ids = list_scenes(args.root, args.split)
    if args.scenes:
        wanted = set(args.scenes)
        unknown = sorted(wanted - set(scene_ids))
        if unknown:
            raise SystemExit("no such scene in %s/%s: %s"
                             % (args.root, args.split, " ".join(unknown)))
        scene_ids = [s for s in scene_ids if s in wanted]

    load_start = time.perf_counter()
    model_cloud = load_model_cloud(ply)
    segmenters = [SegmenterTimer(YOLO(seg_weights),
                                 os.path.basename(seg_weights))]
    if extra_weights:
        segmenters.append(SegmenterTimer(YOLO(extra_weights),
                                         os.path.basename(extra_weights)))
    load_s = round(time.perf_counter() - load_start, 3)
    print("loaded models in %.1fs  (%s)"
          % (load_s, ", ".join(t.name for t in segmenters)), flush=True)

    scenes = {}
    for scene_id in scene_ids:
        repeats, errors = [], []
        for index in range(args.repeat):
            tag = "%s  repeat %d/%d" % (scene_id, index + 1, args.repeat)
            try:
                result = _run_once(args.root, args.split, scene_id,
                                   model_cloud, segmenters, args.pick, deps,
                                   imgsz=args.imgsz)
            except Exception as exc:
                # A board that runs out of memory on the densest pile must
                # still hand back the scenes it managed.
                errors.append("%s: %s" % (type(exc).__name__, exc))
                traceback.print_exc()
                print("%s  FAILED: %s" % (tag, errors[-1]), flush=True)
                continue
            repeats.append(result)
            print("%s  %.1fs  %d poses  top %.2f  rss %.0f MB"
                  % (tag, result["wall_s"], result["n_poses"],
                     result["top_score"], result["peak_rss_mb"]), flush=True)
        record = {"repeats": repeats, "errors": errors}
        if repeats:
            record.update(_summarise(repeats))
        scenes[scene_id] = record

    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(args.timestamp if args.timestamp is not None
                        else time.time())),
        "git_commit": _git_commit(repo_root),
        "argv": argv,
        "note": args.note,
        "config": config,
        "host": describe_host(),
        "model_load_s": load_s,
        "peak_rss_mb": _peak_rss_mb(),
        "determinism": "Open3D RANSAC is stochastic (its OpenMP threads "
                       "share one random engine), so repeats of one scene "
                       "differ on one machine. Per-repeat poses are kept so "
                       "the cross-machine tolerance can be judged against "
                       "the same-machine spread.",
        "scenes": scenes,
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=False)
    failed = sum(1 for s in scenes.values() if not s["repeats"])
    print("wrote %s  (%d scenes, %d failed, peak rss %.0f MB, %s)"
          % (args.out, len(scenes), failed, record["peak_rss_mb"],
             record["host"]["platform_kind"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
