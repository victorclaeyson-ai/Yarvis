"""The interactive loop: listen or read typed text, reply, repeat.

Every command runs as its own task so it can be stopped from another thread (the app's
stop button) without ending the loop. Unexpected errors are reported and the loop
continues: a voice assistant should never silently die mid-conversation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from yarvis.audio.devices import AudioBackendError
from yarvis.listen import PauseHooks
from yarvis.llm.base import LLMError
from yarvis.pipeline import TurnObserver
from yarvis.speech.turn_taking import sounds_finished

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from yarvis.listen import PauseControl, Utterance
    from yarvis.pipeline import Speculation, Turn, TurnResult
    from yarvis.speech.base import AudioArray

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


class Listener(Protocol):
    """Opens the microphone and captures one utterance."""

    def __call__(self, hooks: PauseHooks, *, follow_up: bool) -> Awaitable[Utterance | None]:
        """``follow_up`` listens briefly after a reply instead of waiting for the user."""
        ...


@dataclass(frozen=True, slots=True)
class TurnTaking:
    """How the assistant decides when you are done and whether to keep listening."""

    speculate: bool = True
    """Start transcribing and drafting a reply during pauses."""
    thinking_pause_ms: int | None = 2_000
    """Silence allowed when the sentence sounds unfinished (None: no adaptive waiting)."""
    follow_up: bool = True
    """After a spoken reply, listen again without a key press."""


class Assistant:
    """Reads input lines; an empty line means "listen", any text is sent as typed input."""

    def __init__(
        self,
        *,
        turn: Turn,
        lines: AsyncIterator[str],
        ui: AssistantUi,
        listen: Listener | None,
        turn_taking: TurnTaking | None = None,
    ) -> None:
        self._turn = turn
        self._lines = lines
        self._ui = ui
        self._listen = listen
        self._turn_taking = turn_taking or TurnTaking()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current: asyncio.Task[bool] | None = None

    async def run(self) -> None:
        """Run until the user quits or input ends."""
        self._loop = asyncio.get_running_loop()
        voice = self._listen is not None
        follow_up = False
        while True:
            if follow_up:
                follow_up = await self._run_command("", follow_up=True)
                continue
            self._ui.ready(voice=voice)
            line = await anext(self._lines, None)
            if line is None:
                return
            text = line.strip()
            if text.lower() in QUIT_COMMANDS:
                return
            follow_up = await self._run_command(text)

    def stop(self) -> None:
        """Stop whatever is happening (listening, thinking, speaking). Thread-safe."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._cancel_current)

    def _cancel_current(self) -> None:
        if self._current is not None and not self._current.done():
            logger.info("Stopping the current command")
            self._current.cancel()

    async def _run_command(self, text: str, *, follow_up: bool = False) -> bool:
        """Run one command; returns True if the assistant should listen again by itself."""
        task = asyncio.create_task(self._handle(text, follow_up=follow_up))
        self._current = task
        try:
            await asyncio.wait({task})
        finally:
            self._current = None
            if not task.done():  # ``run`` itself was cancelled
                task.cancel()
        if task.cancelled():
            self._ui.stopped()
            return False
        error = task.exception()
        if error is None:
            return task.result()
        if isinstance(error, LLMError | AudioBackendError):
            logger.warning("%s: %s", type(error).__name__, error)
            self._ui.error(str(error))
        else:
            logger.error("Unexpected error while handling a command", exc_info=error)
            self._ui.error(
                f"Something went wrong ({type(error).__name__}). Details are in the log."
            )
        return False

    async def _handle(self, text: str, *, follow_up: bool = False) -> bool:
        if text:
            logger.info("Typed message (%d characters)", len(text))
            self._ui.reply_finished(await self._turn.from_text(text))
            return False
        if self._listen is None:
            return False
        logger.info("Listening for a follow-up" if follow_up else "Listening")
        self._ui.listening()
        speculation: Speculation | None = None
        thinking_pause_ms = self._turn_taking.thinking_pause_ms

        def on_pause(audio: AudioArray, control: PauseControl) -> None:
            nonlocal speculation
            if speculation is not None:
                speculation.discard()
            speculation = None
            if not self._turn_taking.speculate:
                return
            speculation = self._turn.speculate(audio)
            if thinking_pause_ms is not None:
                patience = thinking_pause_ms

                def adapt(transcript: str) -> None:
                    if not sounds_finished(transcript):
                        control.require_silence_ms(patience)

                speculation.when_transcribed(adapt)

        def on_resume() -> None:
            nonlocal speculation
            if speculation is not None:
                speculation.discard()
                speculation = None

        result = None
        utterance = None
        try:
            hooks = PauseHooks(on_pause=on_pause, on_resume=on_resume)
            utterance = await self._listen(hooks, follow_up=follow_up)
            if utterance is not None:
                result = await self._turn.from_audio(
                    utterance.audio,
                    speech_end=utterance.speech_end,
                    turn_detected=utterance.turn_detected,
                    speculation=speculation,
                )
        finally:
            if speculation is not None:
                speculation.discard()  # no-op once used; stops it if we were interrupted
        if result is not None:
            self._ui.reply_finished(result)
            return self._turn_taking.follow_up
        if not follow_up or utterance is not None:
            self._ui.nothing_heard()  # silence after a reply simply ends the conversation
        return False


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
