"""Small statistics the services report about themselves."""

from __future__ import annotations

from typing import Sequence


def percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted sequence, one decimal.

    Nearest rank rather than interpolation: a latency percentile should be
    a value that actually happened, and 0.0 for an empty window keeps
    ``/metrics`` numeric before the first request.
    """
    if not ordered:
        return 0.0
    rank = max(1, int(round(q * len(ordered))))
    return round(ordered[rank - 1], 1)
