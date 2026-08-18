"""One JSON object per log line, for a log collector to read.

Both services log this way so a collector on the cell sees one shape:
``ts``, ``level``, ``event`` and ``thread`` on every line, plus whatever
fields the caller attached through ``extra={"fields": {...}}``.
"""

from __future__ import annotations

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """Format one record as one JSON object; unknown values fall back to str."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3),
                   "level": record.levelname,
                   "event": record.getMessage(),
                   "thread": record.threadName}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure_logger(logger: logging.Logger, level: str) -> None:
    """Send ``logger`` to stderr as JSON lines at ``level``, and nowhere else."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(getattr(logging, level))
    logger.propagate = False
