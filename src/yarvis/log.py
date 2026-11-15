"""Logging setup. Every handler redacts secrets; log files are private and rotated.

Handlers are attached to the *root* logger, so messages from third-party libraries
(HTTP clients, model loaders) are redacted too. Yarvis logs at the configured level;
everything else only at WARNING and above. On the console, third-party warnings are
hidden (they would interrupt the conversation); they still go to the log file.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Final, override

from rich.logging import RichHandler

from yarvis.paths import PRIVATE_FILE_MODE, ensure_private_dir
from yarvis.security.redact import RedactingFilter, Redactor, default_redactor

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from yarvis.config.models import LogLevel

ROOT_LOGGER: Final = "yarvis"
LOG_FILE_NAME: Final = "yarvis.log"
MAX_LOG_BYTES: Final = 5 * 1024 * 1024
LOG_BACKUPS: Final = 3
_MARKER: Final = "_yarvis_handler"
QUIET_LIBRARIES: Final = ("phonemizer",)
"""Libraries whose warnings are harmless noise ("words count mismatch" on every reply)."""


def configure_logging(
    *,
    level: LogLevel = "INFO",
    log_dir: Path | None = None,
    console: Console | None = None,
    redactor: Redactor = default_redactor,
) -> logging.Logger:
    """Configure logging. Safe to call more than once; only replaces its own handlers."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in tuple(root.handlers):
        if getattr(handler, _MARKER, False):
            root.removeHandler(handler)
            handler.close()

    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    logger.propagate = True
    for noisy in QUIET_LIBRARIES:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    redacting = RedactingFilter(redactor)

    # Tracebacks are rendered by RedactingFilter as plain text; rich tracebacks with
    # local variables could leak secrets, so they stay off.
    console_handler = RichHandler(
        console=console,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    console_handler.addFilter(_ConsoleFilter())
    _install(root, console_handler, redacting)

    if log_dir is not None:
        ensure_private_dir(log_dir)
        log_file = log_dir / LOG_FILE_NAME
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
        log_file.chmod(PRIVATE_FILE_MODE)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"),
        )
        _install(root, file_handler, redacting)

    return logger


class _ConsoleFilter(logging.Filter):
    """Show Yarvis's own messages, and only errors from other libraries."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        own = record.name == ROOT_LOGGER or record.name.startswith(f"{ROOT_LOGGER}.")
        return own or record.levelno >= logging.ERROR


def _install(root: logging.Logger, handler: logging.Handler, redacting: RedactingFilter) -> None:
    handler.addFilter(redacting)
    setattr(handler, _MARKER, True)
    root.addHandler(handler)
