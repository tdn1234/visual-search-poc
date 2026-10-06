"""Centralized logging setup.

Responsibility: configure the root logger exactly once (`configure_logging`,
called from `main.py` before anything else runs) and own the
request-correlation mechanism every log line uses -- a `ContextVar`
holding the current request's short ID, injected into every record via
a `logging.Filter` so ordinary `logger.info(...)` calls throughout the
codebase don't need to know a request is even happening.

`app.middleware.RequestContextMiddleware` is the only thing that ever
*sets* `request_id_var`; every other module just logs normally and
gets correlation for free.
"""

from __future__ import annotations

import logging
import logging.handlers
from contextvars import ContextVar
from pathlib import Path

from app.config import LOG_FILE, LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES, LOG_LEVEL

# Empty string outside a request (startup/shutdown logs, background
# scripts) -- the format string below renders that as a blank field
# rather than the word "None".
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# These libraries' own DEBUG logs are protocol-level noise (every
# multipart chunk, every HTTP connection-pool event) that drowns out
# the app's own DEBUG timing logs without adding debugging value here.
# Setting LOG_LEVEL=DEBUG is meant to surface *our* per-request timing
# breakdown, not third-party internals -- so these are pinned to
# WARNING regardless of LOG_LEVEL.
_NOISY_LOGGERS = ("multipart", "urllib3", "PIL", "httpcore", "httpx", "pika")


class _RequestIdFilter(logging.Filter):
    """Attach the current request's ID (if any) to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    """Set up the root logger's format, level, request-ID filter and handlers.

    Logs go to the console and, unless `LOG_FILE` is empty, to a rotating
    file (`LOG_FILE_MAX_BYTES` each, `LOG_FILE_BACKUP_COUNT` old files kept).

    Call this once, as early as possible in `main.py` -- every module
    in the app calls `logging.getLogger(__name__)` and inherits this
    configuration rather than configuring its own handlers.
    """
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    file_error: OSError | None = None
    if LOG_FILE:
        try:
            Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    LOG_FILE, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT, encoding="utf-8"
                )
            )
        except OSError as exc:  # e.g. read-only filesystem: keep console logging working
            file_error = exc

    for handler in handlers:
        handler.addFilter(_RequestIdFilter())
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.handlers = handlers

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if file_error is not None:
        logging.getLogger(__name__).warning("File logging disabled: cannot write %s (%s)", LOG_FILE, file_error)
