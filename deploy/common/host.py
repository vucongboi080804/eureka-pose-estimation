"""What machine this is, from the files Linux fills in for every board.

A timing without its hardware attached is a rumour, so the bench record,
the HUD's hardware line and the acceptance report all describe the host the
same way, from here. Two files decide whether this is a Jetson: only a
Tegra puts a model name in ``/proc/device-tree/model``, and inside a
container -- where ``/proc/device-tree`` is invisible -- the runbook
bind-mounts that name to ``CONTAINER_DEVICE_MODEL``.
"""

from __future__ import annotations

import os
import platform
import re
from typing import Any, Dict, Optional

#: Where the runbook bind-mounts /proc/device-tree/model inside a container.
CONTAINER_DEVICE_MODEL = "/etc/device-tree-model"


def read_text(path: str) -> Optional[str]:
    """File contents, NULs dropped, or None when it does not exist here."""
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace").replace("\x00", "")
    except OSError:
        return None


def cpu_model() -> Optional[str]:
    """First human-readable CPU name /proc/cpuinfo offers, else None.

    x86 calls it "model name"; a Cortex-A57 under JetPack does too, but some
    aarch64 kernels only fill "Processor", "Hardware" or "CPU part".
    """
    cpuinfo = read_text("/proc/cpuinfo") or ""
    for key in ("model name", "Model", "Processor", "Hardware", "CPU part"):
        match = re.search(r"^%s\s*:\s*(.+)$" % key, cpuinfo, re.M)
        if match:
            return match.group(1).strip()
    return None


def total_ram_mb() -> Optional[int]:
    """MemTotal from /proc/meminfo in MB, or None off Linux."""
    match = re.search(r"^MemTotal:\s*(\d+) kB", read_text("/proc/meminfo") or "", re.M)
    return round(int(match.group(1)) / 1024.0) if match else None


def device_tree_model() -> Optional[str]:
    """The board's own name (e.g. "NVIDIA Jetson Nano Developer Kit"), or None."""
    model = read_text("/proc/device-tree/model") or read_text(CONTAINER_DEVICE_MODEL)
    return model.strip() or None if model else None


def describe_local_host() -> Dict[str, Any]:
    """This machine as the bench record's ``host`` block describes one."""
    return {"device_tree_model": device_tree_model(),
            "cpu_model": cpu_model(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
            "total_ram_mb": total_ram_mb()}
