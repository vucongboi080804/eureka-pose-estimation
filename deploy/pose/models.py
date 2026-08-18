"""The weights the service holds for the life of the process.

Sampling the CAD surface, computing its FPFH features and loading two
segmenters costs several seconds and a gigabyte; a robot cycle is a
second. So the bundle is loaded once, warmed once, and then reused for
every frame -- and its memory is measured, because the smallest target
board this ships to has 4 GB shared between the model, the frame buffers
and the rest of the system.

Lifecycle is explicit rather than lazy: ``load()`` then ``warmup()``, and
``close()`` when the process is winding down. A service that silently
loaded a 100 MB checkpoint on its first request would answer that request
seconds late and report the delay as pose latency.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.model_cloud import load_model_cloud
from src.register import PoseEstimator

from .config import ServiceConfig


class ModelBundle:
    """CAD clouds and segmenter weights, loaded once per process."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._cad = None            # type: Any
        self._primary = None        # type: Any
        self._extra = None          # type: Any
        self._warm = False
        self._load_seconds = 0.0
        self._warmup_seconds = 0.0
        self._baseline_kb = _rss_kb()
        self._loaded_kb = self._baseline_kb

    # -- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """Read the CAD model and the segmenter weights.

        Raises whatever the loaders raise: a bundle that cannot be built
        is a dead service, and it should say so on the launch line rather
        than on the first pick.
        """
        if self._cad is not None:
            return
        started = time.perf_counter()
        self._cad = load_model_cloud(self.config.cad_path)
        # Imported here, not at module scope: ultralytics pulls in torch,
        # which costs a second and a few hundred megabytes even when no
        # segmenter is configured.
        from ultralytics import YOLO
        self._primary = YOLO(self.config.seg_weights)
        if self.config.extra_seg_weights:
            self._extra = YOLO(self.config.extra_seg_weights)
        self._load_seconds = time.perf_counter() - started
        self._loaded_kb = _rss_kb()

    def warmup(self, frame_shape: Tuple[int, int]) -> float:
        """Run one dummy frame so the first real one is not the slow one.

        Ultralytics defers most of its initialisation -- kernel selection,
        the fused forward pass, the first CUDA context -- to the first
        inference, and Open3D/SciPy resolve their BLAS threads the first
        time the estimator is built. Paying that on a blank frame of the
        camera's own shape keeps the first pick honest.

        Args:
            frame_shape: ``(height, width)`` of the camera's colour frame.

        Returns:
            Seconds spent warming.
        """
        if self._cad is None:
            raise RuntimeError("warmup() before load()")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        started = time.perf_counter()
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for model in self.segmenters:
            model(blank, imgsz=self.config.seg_imgsz,
                  conf=self.config.seg_conf, verbose=False,
                  retina_masks=True)
        # Building an estimator touches the KD-trees, the mesh loader and
        # the depth-slope pass; a blank depth map exercises all of them
        # and proposes nothing, so no registration runs.
        PoseEstimator(self._cad, np.zeros((height, width), dtype=np.float64),
                      np.array([[1000.0, 0.0, width / 2.0],
                                [0.0, 1000.0, height / 2.0],
                                [0.0, 0.0, 1.0]]),
                      part_mask=np.zeros((height, width), dtype=bool))
        self._warmup_seconds = time.perf_counter() - started
        self._warm = True
        return self._warmup_seconds

    def close(self) -> None:
        """Drop the weights and let the allocator reclaim what it can."""
        self._cad = None
        self._primary = None
        self._extra = None
        self._warm = False
        gc.collect()

    # -- state -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._cad is not None

    @property
    def is_warm(self) -> bool:
        return self._warm

    @property
    def cad(self) -> Any:
        if self._cad is None:
            raise RuntimeError("model bundle is not loaded")
        return self._cad

    @property
    def primary(self) -> Any:
        return self._primary

    @property
    def extra(self) -> Any:
        return self._extra

    @property
    def segmenters(self) -> List[Any]:
        """Every configured segmenter, primary first."""
        return [m for m in (self._primary, self._extra) if m is not None]

    def memory_footprint(self) -> Dict[str, float]:
        """Resident memory around the load, in MB.

        ``delta_mb`` is what the weights and the CAD clouds cost on top of
        the interpreter that was already running; ``rss_mb`` is what the
        process occupies now, which is the number that has to fit the
        board. Both come from ``/proc/self/status``; on a platform without
        it they fall back to the peak reported by ``resource``, which
        never decreases -- so read ``delta_mb`` as an upper bound there.
        """
        current = _rss_kb()
        return {"baseline_mb": round(self._baseline_kb / 1024.0, 1),
                "loaded_mb": round(self._loaded_kb / 1024.0, 1),
                "rss_mb": round(current / 1024.0, 1),
                "delta_mb": round((self._loaded_kb - self._baseline_kb)
                                  / 1024.0, 1),
                "load_s": round(self._load_seconds, 2),
                "warmup_s": round(self._warmup_seconds, 2)}


def _rss_kb() -> float:
    """Resident set size in kB."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except OSError:
        pass
    import resource
    # ru_maxrss is kB on Linux, bytes on macOS; the peak either way.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) if sys.platform != "darwin" else float(peak) / 1024.0
