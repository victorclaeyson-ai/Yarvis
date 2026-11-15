"""Open the Yarvis window and run the assistant behind it."""

from __future__ import annotations

import asyncio
import logging
import threading
from importlib import resources
from typing import TYPE_CHECKING, Any

from yarvis.app.bridge import AppApi, AppUi, CommandQueue, EventPump
from yarvis.assistant import AssistantControls
from yarvis.llm.anthropic import DEFAULT_MODEL
from yarvis.runtime import run_assistant

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import SecretStr

    from yarvis.config.models import Config
    from yarvis.runtime import Backends

logger = logging.getLogger(__name__)

WINDOW_SIZE = (480, 820)
MIN_SIZE = (400, 600)
BACKGROUND = "#07030f"


def index_html() -> str:
    """The page, loaded from package data and passed to the window as a string.

    Passing HTML directly (instead of a URL) means pywebview starts no HTTP server, so
    the app opens no network port.
    """
    return (resources.files("yarvis.app") / "static" / "index.html").read_text(encoding="utf-8")


class AppError(RuntimeError):
    """The window could not be opened (e.g. no GUI backend on Linux)."""


def run_app(
    config: Config,
    *,
    api_key: SecretStr,
    cache_dir: Path,
    backends: Backends,
    voice_input: bool = True,
    voice_output: bool = True,
    spotify: SecretStr | None = None,
) -> None:  # pragma: no cover - needs a real display; the parts are unit-tested
    """Show the window (blocks until it is closed)."""
    try:
        import webview  # noqa: PLC0415 - heavy GUI import, only for the app
    except ImportError as exc:
        msg = "pywebview is not installed"
        raise AppError(msg) from exc
    gui: Any = webview  # pywebview's annotations are incomplete for strict type checking

    commands = CommandQueue()
    controls = AssistantControls()
    pump = EventPump()
    ui = AppUi(pump.post)
    details = {
        "llm": config.llm.model or DEFAULT_MODEL,
        "stt": f"whisper {config.voice.stt_model}" if voice_input else "off",
        "tts": config.voice.voice if voice_output else "muted",
    }
    api = AppApi(commands, controls, name=config.assistant.name, voice=voice_input, details=details)

    window = gui.create_window(
        config.assistant.name,
        html=index_html(),
        js_api=api,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=MIN_SIZE,
        background_color=BACKGROUND,
        text_select=True,
    )
    if window is None:
        msg = "could not create the window"
        raise AppError(msg)
    window.events.loaded += pump.page_loaded
    pump.start(window)

    def assistant_thread() -> None:
        try:
            asyncio.run(
                run_assistant(
                    config,
                    api_key=api_key,
                    cache_dir=cache_dir,
                    lines=commands.lines(),
                    ui=ui,
                    backends=backends,
                    voice_input=voice_input,
                    voice_output=voice_output,
                    on_loading=ui.notice,
                    controls=controls,
                    spotify=spotify,
                )
            )
        except Exception as exc:
            logger.exception("The assistant stopped")
            ui.error(f"Yarvis stopped: {exc}")

    worker = threading.Thread(target=assistant_thread, name="yarvis-assistant", daemon=True)
    try:
        gui.start(worker.start, private_mode=True, debug=False, http_server=False)
    except Exception as exc:
        msg = f"could not open the window: {exc}"
        raise AppError(msg) from exc
    finally:
        controls.stop()
        commands.close()
        pump.stop()
        worker.join(timeout=5)
