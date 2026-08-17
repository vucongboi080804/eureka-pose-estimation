# Runtime of the pose pipeline

Every number below comes from the submitted configuration
(`scripts/run_pipeline.py --seg-model weights/part-seg.pt --extra-seg-model
weights/part-seg-synthetic.pt`, i.e. what `run_all.sh` runs) on the 40 test
scenes, measured on a quiet machine (1-min load below 4 before each timed run;
a desktop session with a browser and ~1.1 GB of GPU memory in use stayed up).

**Hardware / software.** Intel i5-14600K (6 P + 8 E cores, 20 threads),
NVIDIA RTX 4070 Ti SUPER 16 GB, 31 GB RAM; Python 3.12.3, torch 2.13.0+cu130,
ultralytics 8.4.120, open3d 0.19.0, numpy 2.5.2. CPU-only rows set
`CUDA_VISIBLE_DEVICES=""`. `--workers N` runs N scene processes; each gets
`20 // N` Open3D threads (`OMP_NUM_THREADS`), a single worker takes all 20.

## Throughput: full sweep over the test split

| Setting | Wall | Per scene: mean / median / max [s] | Poses |
|---|---|---|---|
| GPU, `WORKERS=6 ./run_all.sh . test` (pipeline + overlays) | 147 s | 17.6 / 16.8 / 48.3 | 389 |
| GPU, `run_pipeline.py --workers 6` (pipeline only) | 135 s | 17.9 / 16.7 / 47.8 | 396 |
| CPU only, `--workers 6` | 160 s | 21.8 / 20.4 / 60.3 | 389 |
| CPU only, `--workers 2` (small industrial PC) | 280 s | 13.4 / 12.8 / 34.2 | 389 |
| GPU, `--workers 1` (one scene at a time) | 488 s | 12.1 / 12.1 / 31.4 | 389 |

The overlays (`visualize.py`) alone take 15 s. Per-scene time at 6 workers
is inflated by thread sharing (3 Open3D threads each) — 17.9 s per scene
but 3.4 s of wall per scene — while the single worker with 20 threads
needs 12.1 s per scene: parallelism over scenes pays, threads inside
Open3D pay little (next section). The GPU only serves the two segmenters, so
CPU-only costs 19 % more wall at 6 workers. The pose count moves 389–396
between identical runs (the RANSAC draws differ); the committed
`submission.json` has 391.

## Latency: pick mode (`--pick`, one confident pose per scene)

| Setting | Scenes | Wall | Per scene: mean / median / max [s] |
|---|---|---|---|
| GPU, `--workers 1 --pick` | 40 | 34 s | 0.7 / 0.3 / 2.1 |
| CPU only, `--workers 1 --pick` | 10 (000001–000016) | 24 s | 1.8 / 1.8 / 2.4 |
| CPU only, 4 P-cores (`taskset -c 0-7`, 8 threads), `--pick` | same 10 | 22 s | 1.6 / 1.4 / 2.1 |
| CPU only, 4 P-cores, full sweep (for comparison) | same 10 | 119 s | 11.3 / 11.9 / 18.4 |
| GPU, 20 threads, full sweep (same 10 scenes, from the run above) | same 10 | – | 11.4 / 12.9 / 18.7 |

Agreement with the committed `submission.json` (GPU pick run, 40 scenes):
the picked pose coincides (within 2 mm and 2°) with one of the committed
poses in **40/40** scenes and with the *top-scoring* committed pose in
**17/40** — pick mode stops at the first proposal that verifies at ≥ 0.8
(picked scores 0.80–0.94, mean 0.88) rather than ranking the whole scene, so
it returns a correct instance, not necessarily the best-scoring one. The
27 scenes returned exactly one pose, 12 returned two and one returned three
(masks tried before the pick that scored below 0.8 stay in the list).

## Where the time goes

Single worker, GPU present, scenes 000001 000003 000015 000043 000052
(60 poses, 59.7 s of scene time after a one-off 3.9 s model + segmenter
load). Wall-clock wrappers around the stage functions, cumulative and
inclusive of callees, so the nested rows do not add up to 100 %. `cProfile`
on the same run agrees on the C-level totals (`registration_icp` 29.9 s,
`evaluate_registration` 4.3 s, `verify_pose` 3.1 s own time of 68.3 s) but
attributes cumulative time to the wrong Python callers once torch is
imported (`_global_init` showed 0 s), hence the wrappers.

| Stage | Calls | Time [s] | Share |
|---|---|---|---|
| `PoseEstimator.estimate` (whole registration chain per mask) | 115 | 57.4 | 96.1 % |
| ├ `_refine` — coarse-to-fine ICP | 1296 | 24.0 | 40.1 % |
| ├ `_global_init` — FPFH RANSAC | 230 | 20.0 | 33.4 % |
| ├ `_grid_candidates` — rotation-grid fallback | 25 | 6.5 | 10.9 % |
| ├ `_judge` — fitness + depth verification | 1181 | 5.1 | 8.6 % |
| │ └ `verify_pose` | 4708 | 3.5 | 5.8 % |
| ├ `_select_and_package` (incl. polish) | 115 | 2.1 | 3.6 % |
| │ └ `_polish` — edge / hole refinement | 115 | 1.6 | 2.7 % |
| └ `compute_fpfh_feature` | 116 | 0.1 | 0.2 % |
| `masks_from_model` — both segmenters, GPU | 10 | 1.2 | 2.1 % |
| `distance_to_model` — own-mask check | 110 | 0.7 | 1.2 % |
| `PoseEstimator.__init__` (per scene) | 5 | 0.1 | 0.1 % |
| `load_scene` | 5 | 0.05 | 0.1 % |
| `nms` | 5 | 0.00 | 0.0 % |

Each RANSAC start feeds one ICP chain plus three flip rivals, and the grid
fallback adds more, so ICP runs 5.6 chains per RANSAC start; ICP and RANSAC
together are three quarters of the scene time, the grid fallback a tenth,
verification under a tenth, the learned segmentation two per cent.

## Memory

| Measure | Value |
|---|---|
| Peak RSS, one worker, 2-scene run (`/usr/bin/time -v`), GPU | 1.87 GB |
| Peak RSS, one worker, 2-scene run, CPU only | 1.58 GB |
| GPU memory per worker (`nvidia-smi`, 5 s samples, 6 workers) | 0.86–0.96 GB |
| GPU memory, whole device during the 6-worker run | ≈ 6.5 GB (incl. 1.1 GB desktop baseline) |

## What this means for deployment

- Registration dominates: ICP + RANSAC + grid fallback are ~85 % of scene
  time and run on the CPU (Open3D); the GPU accelerates only the 2 % spent
  in the segmenters, so a GPU buys 19 % throughput and no latency.
- Throughput comes from scene-level processes, not threads: 6 workers give
  3.4 s of wall per scene; Open3D's OpenMP gain saturates around 4–8
  threads (4 P-cores CPU-only 11.3 s ≈ 20 threads + GPU 11.4 s on the same
  scenes).
- Pick mode is the deployment path: 0.7 s mean / 2.1 s max per scene on the
  desktop, 1.6–2.4 s CPU-only on 4 cores — 7–17× less than the full sweep —
  and coincided with a committed full-sweep pose in 40/40 test scenes.
- A GPU-less industrial PC with 4 physical cores handles ~2 s per pick or
  ~11 s (max 18 s) per full-scene enumeration at 1.6 GB RAM per worker;
  budget the one-off 4 s model load per process.
- Cheapest further speed-ups, if needed: the flip rivals cost three extra
  ICP chains per RANSAC start for equal AR (`analysis/ablation.md`: 197 s
  vs 132 s wall on train), the rotation grid is 11 % and only fires when
  RANSAC fails to verify, and `MAX_ATTEMPTS` (5 RANSAC restarts) is the
  other lever.
