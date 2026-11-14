from __future__ import annotations

import io
import stat
from typing import TYPE_CHECKING

from rich.console import Console

from yarvis.log import LOG_FILE_NAME, configure_logging

if TYPE_CHECKING:
    from pathlib import Path

    from yarvis.security.redact import Redactor


def test_logs_are_redacted_everywhere(tmp_path: Path, redactor: Redactor) -> None:
    redactor.register("my-very-secret-key")
    screen = io.StringIO()
    logger = configure_logging(
        level="DEBUG",
        log_dir=tmp_path / "logs",
        console=Console(file=screen, width=200),
        redactor=redactor,
    )
    logger.getChild("llm").warning("request with %s failed", "my-very-secret-key")
    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / "logs" / LOG_FILE_NAME
    assert "my-very-secret-key" not in screen.getvalue()
    assert "[REDACTED]" in screen.getvalue()
    assert "my-very-secret-key" not in log_file.read_text()
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600


def test_reconfiguring_replaces_handlers(redactor: Redactor) -> None:
    console = Console(file=io.StringIO())
    configure_logging(console=console, redactor=redactor)
    logger = configure_logging(console=console, redactor=redactor)
    assert len(logger.handlers) == 1
    assert not logger.propagate
