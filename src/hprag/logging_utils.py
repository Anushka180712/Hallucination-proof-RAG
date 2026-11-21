"""Structured, trace-aware logging for every stage of the pipeline.

Why this module exists
----------------------
A RAG system that refuses to hallucinate has to be *auditable*: when it answers,
you should be able to see which chunks it retrieved, how confident retrieval was,
which claims were checked and why each one passed or failed. Plain ``print``
statements cannot give you that, so every component logs through here.

Design choices
--------------
* **Standard library only.** ``logging`` is everywhere, works with every log
  shipper, and adds no dependency. We layer two things on top of it:
* **A trace id per question** (stored in a :class:`contextvars.ContextVar`) so
  every log line produced while answering one question can be grepped together,
  even under concurrent API requests.
* **Two output formats.** ``pretty`` is for humans reading a terminal; ``json``
  is one object per line for machines (Loki, Datadog, CloudWatch, ``jq``).
* **``stage`` timing helper.** Wrapping a step in ``with stage("retrieve"):``
  logs its start, its end, and its duration in milliseconds, so latency per
  stage is visible without a profiler.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("hprag_trace_id", default="-")

# Attributes that every LogRecord has. Anything else passed through ``extra=``
# is treated as a structured field and printed by the formatters below.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "trace_id"}


def new_trace_id() -> str:
    """Start a new trace (one per user question) and return its id."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


class _TraceFilter(logging.Filter):
    """Injects the current trace id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Safe for log aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", "-"),
            "event": record.getMessage(),
        }
        payload.update(_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    """Readable, aligned, optionally coloured output for terminals."""

    COLORS = {"DEBUG": "\033[90m", "INFO": "\033[36m", "WARNING": "\033[33m", "ERROR": "\033[31m"}
    RESET = "\033[0m"

    def __init__(self, color: bool = True) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl = record.levelname[:4]
        if self.color:
            lvl = f"{self.COLORS.get(record.levelname, '')}{lvl}{self.RESET}"
        name = record.name.removeprefix("hprag.")
        extras = " ".join(f"{k}={_short(v)}" for k, v in _fields(record).items())
        line = f"{t} {lvl} [{getattr(record, 'trace_id', '-')}] {name:<22} {record.getMessage()}"
        if extras:
            line += f"  {extras}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _short(v: Any, limit: int = 80) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def configure_logging(level: str = "INFO", fmt: str = "pretty", stream: Any = None) -> None:
    """Configure the ``hprag`` logger tree. Idempotent: safe to call repeatedly."""
    root = logging.getLogger("hprag")
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = stream or sys.stderr
    handler = logging.StreamHandler(stream)
    is_tty = hasattr(stream, "isatty") and stream.isatty()
    handler.setFormatter(JsonFormatter() if fmt == "json" else PrettyFormatter(color=is_tty))
    handler.addFilter(_TraceFilter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``hprag`` logger, e.g. ``hprag.retrieval.bm25``."""
    return logging.getLogger(name if name.startswith("hprag") else f"hprag.{name}")


@contextmanager
def stage(name: str, logger: logging.Logger | None = None, **fields: Any) -> Iterator[dict]:
    """Time a pipeline stage and log its start/end.

    The yielded dict can be filled with result fields (e.g. ``n_hits``) which are
    attached to the ``stage.end`` log line.
    """
    log = logger or get_logger("pipeline")
    out: dict[str, Any] = {}
    log.debug(f"{name}.start", extra=fields)
    t0 = time.perf_counter()
    try:
        yield out
    except Exception:
        log.exception(f"{name}.error", extra={**fields, "ms": _ms(t0)})
        raise
    log.info(f"{name}.end", extra={**fields, **out, "ms": _ms(t0)})


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
