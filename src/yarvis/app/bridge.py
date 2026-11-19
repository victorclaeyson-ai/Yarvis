"""The bridge between the assistant (Python, asyncio) and the window (JavaScript).

Threads involved:

* the **GUI thread** runs the native window (macOS requires the main thread);
* the **assistant thread** runs the asyncio event loop with the voice pipeline;
* pywebview calls ``AppApi`` methods on its own worker threads;
* the **event pump** thread delivers events to the page, batching them so a burst of
  reply fragments costs one JavaScript call instead of dozens.

Security: everything sent to the page is JSON data (never HTML or code), and the page
only ever inserts it with ``textContent``; the dashboard gets numbers, and connection
details already scrubbed of secrets (``telemetry.Link``). The JavaScript API is five small methods
with validated arguments (``hello``, ``talk``, ``stop``, ``send``, ``new_conversation``).
"""

from __future__ import annotations

import json
import queue
import threading
from typing import TYPE_CHECKING, Any, Final, Protocol

from yarvis.cli.ui import format_timings
from yarvis.telemetry import RECENT_TURNS
from yarvis.usage import PRICES_AS_OF, percentile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from yarvis.assistant import AssistantControls
    from yarvis.pipeline import TurnResult
    from yarvis.telemetry import Link, UsageUpdate
    from yarvis.tools.timers import Alarm
    from yarvis.tools.weather import Day, Outlook

Event = dict[str, Any]

MAX_MESSAGE_CHARS: Final = 2_000
MAX_LABEL_CHARS: Final = 60
NEW_CONVERSATION: Final = "new conversation"
"""What the dashboard's button sends: handled by code like the spoken command."""
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
        user: str | None = None,
        place: str | None = None,
        asks: bool = True,
    ) -> None:
        self._commands = commands
        self._controls = controls
        self._name = name
        self._voice = voice
        self._user = user
        self._place = place
        self._asks = asks

    def hello(self) -> Event:
        """What the page shows from the start: names, the input mode, the gate's setting.

        Everything else (the connections, the numbers, timers, weather) arrives as events.
        """
        return {
            "name": self._name,
            "voice": self._voice,
            "user": self._user,
            "place": self._place,
            "asks": self._asks,
        }

    def talk(self) -> None:
        """Start listening (ignored in text-only mode)."""
        if self._voice:
            self._commands.put("")

    def stop(self) -> None:
        """Stop listening, thinking or speaking."""
        self._controls.stop()

    def new_conversation(self) -> None:
        """Start a new conversation: the same as saying "new conversation"."""
        self._commands.put(NEW_CONVERSATION)

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
    """``AssistantUi`` and ``Dashboard`` that turn every update into an event for the page."""

    def __init__(self, emit: Callable[[Event], None]) -> None:
        self._emit = emit
        self._frames = 0

    def notice(self, message: str) -> None:
        """Neutral status text."""
        self._emit({"type": "notice", "text": message})

    def ready(self, *, voice: bool, wake_word: bool = False) -> None:
        """Waiting for input."""
        self._emit({"type": "state", "state": "idle", "voice": voice, "wake": wake_word})

    def woke(self) -> None:
        """The wake word was heard."""
        self._emit({"type": "wake"})

    def listening(self) -> None:
        """Capturing a request."""
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

    def said(self, text: str) -> None:
        """A fixed phrase, such as a greeting."""
        self._emit({"type": "said", "text": text})

    def alert(self, text: str) -> None:
        """A timer or reminder is due."""
        self._emit({"type": "alert", "text": text})

    def activity(self, text: str) -> None:
        """Yarvis is doing something."""
        self._emit({"type": "activity", "text": text})

    def nothing_heard(self) -> None:
        """No speech detected."""
        self._emit({"type": "notice", "text": "Didn't catch that."})

    def error(self, message: str) -> None:
        """Something went wrong; the conversation continues."""
        self._emit({"type": "error", "text": message})

    def stopped(self) -> None:
        """The user stopped the current command."""
        self._emit({"type": "stopped"})

    def interrupted(self) -> None:
        """The user talked over the reply."""
        self._emit({"type": "interrupted"})

    def links(self, links: Sequence[Link]) -> None:
        """The state of every connection."""
        self._emit(
            {
                "type": "links",
                "links": [
                    {"key": c.key, "label": c.label, "state": c.state, "detail": c.detail}
                    for c in links
                ],
            }
        )

    def usage(self, update: UsageUpdate) -> None:
        """Today's cost and speed (and the cost of the turn that just ended)."""
        day = update.today
        latencies = day.latencies_ms
        self._emit(
            {
                "type": "usage",
                "turn_usd": update.turn_usd,
                "as_of": PRICES_AS_OF,
                "today": {
                    "turns": day.turns,
                    "requests": day.requests,
                    "input_tokens": day.input_tokens,
                    "output_tokens": day.output_tokens,
                    "web_searches": day.web_searches,
                    "speech_seconds": round(day.speech_seconds, 1),
                    "claude_usd": day.claude_usd,
                    "search_usd": day.search_usd,
                    "speech_usd": day.speech_usd,
                    "total_usd": day.total_usd,
                    "unpriced": day.unpriced,
                    "p50_ms": percentile(latencies, 0.5),
                    "p90_ms": percentile(latencies, 0.9),
                    "recent_ms": list(latencies[-RECENT_TURNS:]),
                },
            }
        )

    def timers(self, alarms: Sequence[Alarm]) -> None:
        """The running timers and reminders; the page counts down from ``due_ms``."""
        self._emit(
            {
                "type": "timers",
                "timers": [
                    {
                        "id": alarm.id,
                        "kind": alarm.kind,
                        "label": alarm.label[:MAX_LABEL_CHARS],
                        "due_ms": round(alarm.due.timestamp() * 1000),
                        "seconds": alarm.seconds,
                    }
                    for alarm in alarms
                ],
            }
        )

    def weather(self, outlook: Outlook) -> None:
        """The weather at home (refreshed twice an hour)."""
        self._emit(
            {
                "type": "weather",
                "weather": {
                    "place": outlook.place[:MAX_LABEL_CHARS],
                    "temperature": outlook.temperature,
                    "feels_like": outlook.feels_like,
                    "condition": outlook.condition,
                    "wind_kmh": outlook.wind_kmh,
                    "today": _day(outlook.today),
                    "tomorrow": _day(outlook.tomorrow),
                },
            }
        )


def _day(day: Day | None) -> Event | None:
    if day is None:
        return None
    return {"condition": day.condition, "low": day.low, "high": day.high, "rain": day.rain_chance}
