"""Logging setup. Every handler redacts secrets; log files are private and rotated."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Final

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


def configure_logging(
    *,
    level: LogLevel = "INFO",
    log_dir: Path | None = None,
    console: Console | None = None,
    redactor: Redactor = default_redactor,
) -> logging.Logger:
    """Configure the ``yarvis`` logger hierarchy. Safe to call more than once."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    redacting = RedactingFilter(redactor)

    # Tracebacks are rendered by RedactingFilter as plain text; rich tracebacks with
    # local variables could leak secrets, so they stay off.
    console_handler = RichHandler(
        console=console,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    console_handler.addFilter(redacting)
    logger.addHandler(console_handler)

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
        file_handler.addFilter(redacting)
        logger.addHandler(file_handler)

    return logger
