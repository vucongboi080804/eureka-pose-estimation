"""Elapsed time, the one way every stage timer measures it."""

from __future__ import annotations

import time
from typing import Optional


def ms_since(mark: float, ndigits: Optional[int] = None) -> float:
    """Milliseconds since a ``time.perf_counter()`` mark.

    Wall clock, inclusive of everything that happened in between, which is
    what a cell integrator wants from a stage latency; ``ndigits`` rounds it
    for a log line and leaves it exact for arithmetic when omitted.
    """
    ms = (time.perf_counter() - mark) * 1000.0
    return round(ms, ndigits) if ndigits is not None else ms
