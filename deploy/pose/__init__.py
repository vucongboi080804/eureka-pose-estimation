"""A long-running 6-DoF pose service for a bin-picking cell.

The cell hands the service one registered RGB-D frame and gets back the
ranked poses for it, with the score to gate on and the configuration
digest that produced them:

    server.py    HTTP transport, health, metrics, graceful shutdown
    service.py   PoseService -- the estimator, transport-free
    adapter.py   one registered RGB-D frame -> Scene -> poses; the seam a
                 camera SDK plugs into
    models.py    the CAD cloud and segmenter weights, loaded once
    config.py    ServiceConfig -- resolved, validated, digested
    schema.py    the request/response contract
    client.py    a stdlib CLI for the same endpoints

Importing this package is cheap on purpose: it pulls in nothing heavier
than the standard library, so ``server.py`` can read and validate the
configuration -- and set ``OMP_NUM_THREADS``, which libgomp reads once,
when Open3D first loads -- before NumPy, Open3D or torch are imported.
The two names that would break that are resolved lazily below.
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: Repository root on the path, so ``src`` resolves however the package
#: was reached (``python -m``, an installed console script, a systemd
#: unit with a different working directory).
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

SERVICE_VERSION = "1.0.0"

from .config import LOG_LEVELS, ConfigError, ServiceConfig
from .schema import (GATE_PICK, GATE_RESCAN, SCHEMA_VERSION, EstimateRequest,
                     FrameResult, PoseEstimateDTO, RequestError)

__all__ = ["SERVICE_VERSION", "SCHEMA_VERSION", "LOG_LEVELS", "ServiceConfig",
           "ConfigError", "EstimateRequest", "FrameResult", "PoseEstimateDTO",
           "RequestError", "GATE_PICK", "GATE_RESCAN", "PoseService",
           "ServiceBusy", "ModelBundle"]


def __getattr__(name: str) -> Any:
    """Resolve the estimator names on first use (PEP 562).

    They import NumPy, Open3D and torch; the entry point must be able to
    read a configuration without paying for that.
    """
    if name in ("PoseService", "ServiceBusy"):
        from . import service
        return getattr(service, name)
    if name == "ModelBundle":
        from .models import ModelBundle
        return ModelBundle
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
