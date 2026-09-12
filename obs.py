"""Structured logging for every boundary in the pipeline. Spec section 21.

One JSON line per instrumented boundary, keyed on the `trace_id` D-38 already
made deterministic, so nothing here invents a second correlation key.

Two rules bound what this module may do, and both are the reason it exists as a
module rather than as scattered `logging` calls:

Determinism (D-70). A log line carries a clock and the ground rules say the same
input produces the same bytes. Both stay true only because log output never
reaches `transcripts/`, the README number blocks or any committed file. Lines go
to stderr and to a gitignored `logs/`, and durations are rounded so clock noise
never becomes the reason two runs look different.

Redaction (D-71). Observability is the usual way a PII rule gets broken, because
the rule is normally written for responses and not for diagnostics. Every query
logged from here passes through the masker the guardrails already use; the API
key is never logged in any form; prompts and retrieved context are logged as doc
ids, counts and lengths, never as text.

Stdlib only, by D-70.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

import config

LOGGER_NAME = "loan_support_agent"

# Keys whose value is never written, whatever a caller passes. A deny list is
# the wrong shape for PII in general, which is why free text goes through
# mask_pii instead; this is only the belt for the named secrets.
_NEVER_LOG = frozenset({"api_key", "key", "authorization", "token", "secret"})

_configured = False


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the event's own fields flattened in."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "level": record.levelname,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            line.update(fields)
        if record.exc_info:
            line["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(line, default=str, sort_keys=True)


def configure(level: str | None = None) -> logging.Logger:
    """Attach the stderr and file handlers once, and return the logger."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(getattr(logging, (level or config.LOG_LEVEL).upper(), logging.WARNING))
    logger.propagate = False

    stream = logging.StreamHandler()
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)

    # The file handler is best-effort: a read-only checkout still logs to
    # stderr rather than failing the run over a directory it cannot create.
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    except OSError:
        pass

    _configured = True
    return logger


def mask(text: str) -> str:
    """Run free text through the masker the guardrails already use.

    Imported lazily because `agent` imports config and this module sits below
    both; a module-level import would close the cycle.
    """
    if not text:
        return text
    try:
        from agent.guardrails import mask_pii
    except ImportError:
        return text
    masked, _ = mask_pii(text)
    return masked


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if k.lower() not in _NEVER_LOG and v is not None}


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one line. Callers pass doc ids, counts and lengths, never text."""
    configure().log(level, name, extra={"fields": _clean(fields)})


@contextmanager
def timed(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a boundary and emit one line for it.

    The yielded dict is the line under construction: a caller adds what it
    learns as it learns it, which keeps the instrumentation to one call at a
    boundary rather than one before and one after.
    """
    line: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    try:
        yield line
    except Exception as exc:
        line["outcome"] = "error"
        line["error_type"] = type(exc).__name__
        _emit(name, line, start, logging.ERROR)
        raise
    else:
        line.setdefault("outcome", "ok")
        _emit(name, line, start, logging.INFO)


def _emit(name: str, line: dict[str, Any], start: float, level: int) -> None:
    # Rounded, per D-70: a duration is for spotting a boundary that got slow,
    # not for distinguishing two runs of the same input.
    line["duration_ms"] = round((time.perf_counter() - start) * 1000, config.LOG_DURATION_PLACES)
    configure().log(level, name, extra={"fields": _clean(line)})
