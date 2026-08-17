"""The camera of a bin-picking cell, as a service.

The estimator already accepts a frame by value (base64 PNGs, ``K``,
``depth_scale``); this package is what puts a frame on the other end of
that socket, whether the pixels come from a RealSense on the cell or from
a folder of recorded scenes:

    server.py    HTTP transport, health, metrics, MJPEG preview
    sources.py   FrameSource -- the replay stream and the RealSense driver
    frame.py     Frame -- one RGB-D capture and its wire form
    config.py    CameraConfig -- resolved and validated at startup
    client.py    a stdlib CLI for the same endpoints

Splitting the camera off from the estimator is what lets the whole cell be
demonstrated and regression-tested without hardware: the recorded scenes
enter the pose service through the same socket, the same JSON and the same
PNG bytes a sensor would use, so nothing on the estimator's side can be
accidentally built around the fact that the frames came off a disk.
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: Repository root on the path, so ``src`` resolves however the package
#: was reached (``python -m``, a systemd unit started from ``/``). Only
#: :meth:`Frame.to_scene` needs it, and only in a process that also runs
#: the pipeline.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

CAMERA_VERSION = "1.0.0"

#: Bumped on any change to the frame body served by ``/v1/frame``. The
#: four estimator fields inside it are pinned by the pose service's own
#: schema version instead.
WIRE_VERSION = "1.0"

from .config import LOG_LEVELS, SOURCE_KINDS, CameraConfig, ConfigError

__all__ = ["CAMERA_VERSION", "WIRE_VERSION", "LOG_LEVELS", "SOURCE_KINDS",
           "CameraConfig", "ConfigError", "Frame", "FrameSource",
           "SceneFolderSource", "RealSenseSource", "FrameReadError",
           "FrameSourceError", "SourceExhausted"]


def __getattr__(name: str) -> Any:
    """Resolve the frame and source names on first use (PEP 562).

    They import NumPy and OpenCV; the entry point reads and validates a
    configuration before paying for that, and the client's ``info`` and
    ``grab`` commands never pay for it at all.
    """
    if name == "Frame":
        from .frame import Frame
        return Frame
    if name in ("FrameSource", "SceneFolderSource", "RealSenseSource",
                "FrameReadError", "FrameSourceError", "SourceExhausted"):
        from . import sources
        return getattr(sources, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
