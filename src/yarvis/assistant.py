"""The interactive loop: listen or read typed text, reply, repeat.

Every command runs as its own task so it can be stopped from another thread (the app's
stop button) without ending the loop. Unexpected errors are reported and the loop
continues: a voice assistant should never silently die mid-conversation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Protocol

from yarvis.audio.devices import AudioBackendError
from yarvis.llm.base import LLMError
from yarvis.pipeline import TurnObserver

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from yarvis.listen import Utterance
    from yarvis.pipeline import Turn, TurnResult

logger = logging.getLogger(__name__)

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


class AssistantUi(TurnObserver, Protocol):
    """Everything the user sees."""

    def ready(self, *, voice: bool) -> None:
        """Waiting for input."""
        ...

    def listening(self) -> None:
        """The microphone is open."""
        ...

    def speech_started(self) -> None:
        """The user started talking."""
        ...

    def level(self, value: float) -> None:
        """Microphone input level (RMS) while listening."""
        ...

    def nothing_heard(self) -> None:
        """No speech, or nothing intelligible."""
        ...

    def reply_finished(self, result: TurnResult) -> None:
        """The reply has been fully spoken."""
        ...

    def error(self, message: str) -> None:
        """Something went wrong; the loop continues."""
        ...

    def stopped(self) -> None:
        """The current command was stopped by the user."""
        ...


class Assistant:
    """Reads input lines; an empty line means "listen", any text is sent as typed input."""

    def __init__(
        self,
        *,
        turn: Turn,
        lines: AsyncIterator[str],
        ui: AssistantUi,
        listen: Callable[[], Awaitable[Utterance | None]] | None,
    ) -> None:
        self._turn = turn
        self._lines = lines
        self._ui = ui
        self._listen = listen
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Run until the user quits or input ends."""
        self._loop = asyncio.get_running_loop()
        voice = self._listen is not None
        while True:
            self._ui.ready(voice=voice)
            line = await anext(self._lines, None)
            if line is None:
                return
            text = line.strip()
            if text.lower() in QUIT_COMMANDS:
                return
            await self._run_command(text)

    def stop(self) -> None:
        """Stop whatever is happening (listening, thinking, speaking). Thread-safe."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._cancel_current)

    def _cancel_current(self) -> None:
        if self._current is not None and not self._current.done():
            logger.info("Stopping the current command")
            self._current.cancel()

    async def _run_command(self, text: str) -> None:
        task = asyncio.create_task(self._handle(text))
        self._current = task
        try:
            await asyncio.wait({task})
        finally:
            self._current = None
            if not task.done():  # ``run`` itself was cancelled
                task.cancel()
        if task.cancelled():
            self._ui.stopped()
            return
        error = task.exception()
        if error is None:
            return
        if isinstance(error, LLMError | AudioBackendError):
            logger.warning("%s: %s", type(error).__name__, error)
            self._ui.error(str(error))
        else:
            logger.error("Unexpected error while handling a command", exc_info=error)
            self._ui.error(
                f"Something went wrong ({type(error).__name__}). Details are in the log."
            )

    async def _handle(self, text: str) -> None:
        if text:
            logger.info("Typed message (%d characters)", len(text))
            self._ui.reply_finished(await self._turn.from_text(text))
            return
        if self._listen is None:
            return
        logger.info("Listening")
        self._ui.listening()
        utterance = await self._listen()
        result = None
        if utterance is not None:
            result = await self._turn.from_audio(
                utterance.audio,
                speech_end=utterance.speech_end,
                turn_detected=utterance.turn_detected,
            )
        if result is None:
            self._ui.nothing_heard()
        else:
            self._ui.reply_finished(result)


class AssistantControls:
    """A thread-safe handle on a running ``Assistant``, e.g. for a GUI's stop button."""

    def __init__(self) -> None:
        self._assistant: Assistant | None = None
        self._lock = threading.Lock()

    def attach(self, assistant: Assistant) -> None:
        """Connect to the assistant that is about to run."""
        with self._lock:
            self._assistant = assistant

    def stop(self) -> None:
        """Stop the current command, if any."""
        with self._lock:
            assistant = self._assistant
        if assistant is not None:
            assistant.stop()
