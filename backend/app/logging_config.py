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
from contextvars import ContextVar

from app.config import LOG_LEVEL

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
_NOISY_LOGGERS = ("multipart", "urllib3", "PIL", "httpcore", "httpx")


class _RequestIdFilter(logging.Filter):
    """Attach the current request's ID (if any) to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    """Set up the root logger's format, level, and request-ID filter.

    Call this once, as early as possible in `main.py` -- every module
    in the app calls `logging.getLogger(__name__)` and inherits this
    configuration rather than configuring its own handlers.
    """
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.handlers = [handler]

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
