import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from config.settings import settings


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Generate and bind a fresh correlation ID for the current context.

    Called once per inbound request (Phase 8 middleware) or per CLI invocation.
    Every subsequent log line in that context inherits it.
    """
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid

def set_request_id(rid: str) -> None:
    """Bind an externally supplied ID, e.g. from an X-Request-ID header."""
    _request_id.set(rid)


def get_request_id() -> str | None:
    return _request_id.get()



#: Attributes present on every LogRecord. Anything not in here was passed via
#: `extra=` and should be surfaced as structured context.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class StructuredJSONFormatter(logging.Formatter):
    """Renders log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        rid = _request_id.get()
        if rid:
            payload["request_id"] = rid

        # Anything passed as logger.info("...", extra={"chunk_id": x}) lands on the
        # record as a plain attribute. Promote those into a nested context object.
        context = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if context:
            payload["context"] = context

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # default=str so datetimes, Paths, and Enums serialise instead of raising.
        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Compact coloured output for local development.

    JSON is correct for machines and miserable to read while coding. We select
    between the two on ENVIRONMENT rather than making anyone choose.
    """

    _COLOURS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        rid = _request_id.get()
        prefix = f"[{rid}] " if rid else ""
        line = f"{colour}{stamp} {record.levelname:<8}{self._RESET} {prefix}{record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logger(name: str = "legalrag") -> logging.Logger:
    """Configure and return the application logger. Idempotent."""
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

    # Guard against duplicate handlers — this module may be imported many times,
    # and each extra handler duplicates every line of output.
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            StructuredJSONFormatter() if settings.is_production else HumanFormatter()
        )
        log.addHandler(handler)

    # Do not forward to the root logger, which would print everything twice
    # once uvicorn installs its own handlers in Phase 8.
    log.propagate = False
    return log


logger = setup_logger()