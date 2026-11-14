"""The bridge between the assistant (Python, asyncio) and the window (JavaScript).

Threads involved:

* the **GUI thread** runs the native window (macOS requires the main thread);
* the **assistant thread** runs the asyncio event loop with the voice pipeline;
* pywebview calls ``AppApi`` methods on its own worker threads;
* the **event pump** thread delivers events to the page, batching them so a burst of
  reply fragments costs one JavaScript call instead of dozens.

Security: everything sent to the page is JSON data (never HTML or code), and the page
only ever inserts it with ``textContent``. The JavaScript API is four small methods
with validated arguments (``hello``, ``talk``, ``stop``, ``send``).
"""

from __future__ import annotations

import json
import queue
import threading
from typing import TYPE_CHECKING, Any, Final, Protocol

from yarvis.cli.ui import format_timings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from yarvis.assistant import AssistantControls
    from yarvis.pipeline import TurnResult

Event = dict[str, Any]

MAX_MESSAGE_CHARS: Final = 2_000
LEVEL_EVERY_N_FRAMES: Final = 2  # ~15 updates per second is plenty for an animation


class CommandQueue:
    """Thread-safe input for the assistant: ``""`` means "listen", text means "send"."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()

    def put(self, command: str) -> None:
        """Queue a command (callable from any thread)."""
        self._queue.put(command)

    def close(self) -> None:
        """End the stream of commands; the assistant loop then exits."""
        self._queue.put(None)

    async def lines(self) -> AsyncIterator[str]:
        """Commands as an async stream, for ``Assistant``."""
        import asyncio  # noqa: PLC0415 - only needed in the assistant thread

        while (command := await asyncio.to_thread(self._queue.get)) is not None:
            yield command


class AppApi:
    """Methods JavaScript may call via ``window.pywebview.api``. Keep this surface tiny."""

    def __init__(
        self,
        commands: CommandQueue,
        controls: AssistantControls,
        *,
        name: str,
        voice: bool,
    ) -> None:
        self._commands = commands
        self._controls = controls
        self._name = name
        self._voice = voice

    def hello(self) -> Event:
        """Initial information for the page."""
        return {"name": self._name, "voice": self._voice}

    def talk(self) -> None:
        """Start listening (ignored in text-only mode)."""
        if self._voice:
            self._commands.put("")

    def stop(self) -> None:
        """Stop listening, thinking or speaking."""
        self._controls.stop()

    def send(self, text: object) -> None:
        """Send a typed message."""
        if not isinstance(text, str):
            return
        message = " ".join(text.split())[:MAX_MESSAGE_CHARS]
        if message:
            self._commands.put(message)


class Window(Protocol):
    """The part of ``webview.Window`` the pump needs."""

    def run_js(self, script: str) -> object:
        """Execute JavaScript without waiting for a result."""
        ...


class EventPump:
    """Delivers events to the page from a background thread, in batches.

    Events posted before the page has loaded are kept and delivered once it has.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Event | None] = queue.Queue()
        self._loaded = threading.Event()
        self._window: Window | None = None
        self._thread = threading.Thread(target=self._run, name="yarvis-ui-events", daemon=True)

    def start(self, window: Window) -> None:
        """Begin delivering to ``window`` (events wait until ``page_loaded``)."""
        self._window = window
        self._thread.start()

    def page_loaded(self) -> None:
        """Called when the page is ready to receive events."""
        self._loaded.set()

    def post(self, event: Event) -> None:
        """Queue an event (callable from any thread)."""
        self._queue.put(event)

    def stop(self, timeout: float = 2.0) -> None:
        """Deliver what is queued, then stop."""
        self._queue.put(None)
        self._loaded.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        self._loaded.wait()
        while True:
            first = self._queue.get()
            batch = [] if first is None else [first]
            stopping = first is None
            while not stopping:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                else:
                    batch.append(item)
            if batch and self._window is not None:
                self._window.run_js(script_for(batch))
            if stopping:
                return


def script_for(events: list[Event]) -> str:
    """JavaScript that hands ``events`` to the page as data.

    ``ensure_ascii`` escapes every non-ASCII character (including U+2028/U+2029, which
    are line terminators in older JavaScript), so the JSON is always a safe literal.
    """
    payload = json.dumps(events, ensure_ascii=True, separators=(",", ":"))
    return f"window.yarvis && window.yarvis.receive({payload});"


class AppUi:
    """``AssistantUi`` that turns every update into an event for the page."""

    def __init__(self, emit: Callable[[Event], None]) -> None:
        self._emit = emit
        self._frames = 0

    def notice(self, message: str) -> None:
        """Neutral status text."""
        self._emit({"type": "notice", "text": message})

    def ready(self, *, voice: bool) -> None:
        """Waiting for input."""
        self._emit({"type": "state", "state": "idle", "voice": voice})

    def listening(self) -> None:
        """The microphone is open."""
        self._frames = 0
        self._emit({"type": "state", "state": "listening"})

    def speech_started(self) -> None:
        """The user started talking."""
        self._emit({"type": "speech"})

    def level(self, value: float) -> None:
        """Microphone level, throttled."""
        self._frames += 1
        if self._frames % LEVEL_EVERY_N_FRAMES == 0:
            self._emit({"type": "level", "value": round(min(value, 1.0), 4)})

    def on_transcript(self, text: str) -> None:
        """The user's words."""
        self._emit({"type": "user", "text": text})
        self._emit({"type": "state", "state": "thinking"})

    def on_reply_delta(self, text: str) -> None:
        """A fragment of the reply."""
        self._emit({"type": "delta", "text": text})

    def on_audio_started(self) -> None:
        """The reply is being spoken."""
        self._emit({"type": "state", "state": "speaking"})

    def reply_finished(self, result: TurnResult) -> None:
        """The reply is complete."""
        self._emit(
            {
                "type": "done",
                "timing": format_timings(result.timings),
                "ms": result.timings.voice_to_voice_ms,
            }
        )

    def nothing_heard(self) -> None:
        """No speech detected."""
        self._emit({"type": "notice", "text": "Didn't catch that."})

    def error(self, message: str) -> None:
        """Something went wrong; the conversation continues."""
        self._emit({"type": "error", "text": message})

    def stopped(self) -> None:
        """The user stopped the current command."""
        self._emit({"type": "stopped"})
