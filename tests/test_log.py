from __future__ import annotations

import io
import logging
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
    for handler in ours():
        handler.flush()

    log_file = tmp_path / "logs" / LOG_FILE_NAME
    assert "my-very-secret-key" not in screen.getvalue()
    assert "[REDACTED]" in screen.getvalue()
    assert "my-very-secret-key" not in log_file.read_text()
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600


def ours() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "_yarvis_handler", False)]


def test_reconfiguring_replaces_only_our_handlers(redactor: Redactor) -> None:
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    console = Console(file=io.StringIO())
    configure_logging(console=console, redactor=redactor)
    configure_logging(console=console, redactor=redactor)
    assert len(ours()) == 1
    assert foreign in logging.getLogger().handlers
    logging.getLogger().removeHandler(foreign)


def test_third_party_logs_are_redacted_and_kept_off_the_console(
    tmp_path: Path, redactor: Redactor
) -> None:
    redactor.register("third-party-secret-value")
    screen = io.StringIO()
    configure_logging(
        level="DEBUG",
        log_dir=tmp_path,
        console=Console(file=screen, width=200),
        redactor=redactor,
    )
    logging.getLogger("httpx2").info("info noise")
    logging.getLogger("onnxruntime").warning("a library warning")
    logging.getLogger("httpx2").error("token third-party-secret-value leaked")
    logging.getLogger("yarvis.test").debug("our debug line")
    for handler in ours():
        handler.flush()
    console = screen.getvalue()
    log_file = (tmp_path / LOG_FILE_NAME).read_text()
    assert "info noise" not in console + log_file
    assert "a library warning" not in console
    assert "a library warning" in log_file
    assert "third-party-secret-value" not in console + log_file
    assert "[REDACTED]" in console
    assert "our debug line" in console


def test_harmless_library_warnings_are_silenced(redactor: Redactor) -> None:
    configure_logging(redactor=redactor)
    assert not logging.getLogger("phonemizer").isEnabledFor(logging.WARNING)
    assert logging.getLogger("phonemizer").isEnabledFor(logging.ERROR)
