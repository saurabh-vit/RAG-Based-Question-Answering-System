from __future__ import annotations

import logging
import sys
from typing import Any


class _KeyValueFormatter(logging.Formatter):
    """
    Lightweight structured-ish logs without extra dependencies.
    Example:
      level=INFO msg="..." request_id=... latency_ms=...
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extras (anything not in LogRecord defaults)
        extras: dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            extras[k] = v

        parts = [f'{k}={_quote(v)}' for k, v in base.items()]
        parts.extend([f"{k}={_quote(v)}" for k, v in sorted(extras.items())])

        if record.exc_info:
            parts.append("exc_info=true")
        return " ".join(parts)


def _quote(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, (int, float, bool)):
        return str(v).lower() if isinstance(v, bool) else str(v)
    s = str(v).replace('"', '\\"')
    return f'"{s}"'


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_KeyValueFormatter())

    root.handlers.clear()
    root.addHandler(handler)

