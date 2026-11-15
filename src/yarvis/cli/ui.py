"""Terminal presentation of the conversation."""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING

from yarvis.speech.turn_taking import WAKE_PHRASE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rich.console import Console

    from yarvis.pipeline import TurnResult, TurnTimings


def format_timings(timings: TurnTimings) -> str | None:
    """One-line latency summary, or None if there is nothing to report."""
    stages = [
        ("end of turn", timings.end_of_turn_ms),
        ("speech-to-text", timings.stt_ms),
        ("Claude", timings.llm_ms),
        ("voice", timings.tts_ms),
    ]
    parts = [f"{label} {value / 1000:.2f}" for label, value in stages if value is not None]
    if not parts:
        return None
    total = timings.voice_to_voice_ms
    head = f"{total / 1000:.2f} s to first word" if total is not None else "timings"
    return f"{head}  ({' · '.join(parts)})"


class ConsoleUi:
    """Prints the conversation with streaming replies and a latency line per turn."""

    def __init__(self, console: Console, *, name: str, show_timings: bool = True) -> None:
        self._console = console
        self._name = name
        self._show_timings = show_timings
        self._replying = False

    def notice(self, message: str) -> None:
        """A neutral status message."""
        self._console.print(f"[dim]{message}[/]", highlight=False)

    def ready(self, *, voice: bool, wake_word: bool = False) -> None:
        """Show the input prompt."""
        hint = "Enter = talk · type to chat · q = quit" if voice else "type a message · q = quit"
        if wake_word:
            hint = f"Say “{WAKE_PHRASE}” · {hint}"
        self._console.print(f"\n[dim]{hint}[/]")
        self._console.print("[bold cyan]›[/] ", end="")  # noqa: RUF001

    def woke(self) -> None:
        """The wake word was heard."""
        self._console.print()  # the prompt line was waiting for typed input

    def listening(self) -> None:
        """Capturing a request."""
        self._console.print("[red]●[/] listening…")

    def speech_started(self) -> None:
        """The user started talking (nothing to show yet)."""

    def level(self, value: float) -> None:
        """Microphone level: not shown in the terminal."""

    def on_audio_started(self) -> None:
        """Reply audio started: not shown in the terminal."""

    def on_transcript(self, text: str) -> None:
        """Show what was heard."""
        self._console.print(f"[bold]You:[/] {_escape(text)}", highlight=False)

    def on_reply_delta(self, text: str) -> None:
        """Stream the reply as it is generated."""
        if not self._replying:
            self._console.print(f"[bold magenta]{_escape(self._name)}:[/] ", end="")
            self._replying = True
        self._console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def reply_finished(self, result: TurnResult) -> None:
        """End the reply line and show latency."""
        self._end_reply()
        if self._show_timings and (line := format_timings(result.timings)):
            self._console.print(f"[dim]⏱ {line}[/]", highlight=False)

    def said(self, text: str) -> None:
        """A fixed phrase, such as a greeting."""
        self._end_reply()
        name = _escape(self._name)
        self._console.print(f"[bold magenta]{name}:[/] {_escape(text)}", highlight=False)

    def alert(self, text: str) -> None:
        """A timer or reminder is due."""
        self._end_reply()
        self._console.print(f"\n[bold yellow]⏰ {_escape(text)}[/]", highlight=False)

    def activity(self, text: str) -> None:
        """Yarvis is doing something."""
        self._end_reply()
        self._console.print(f"[dim]• {_escape(text)}[/]", highlight=False)

    def nothing_heard(self) -> None:
        """Nothing intelligible was said."""
        self._console.print("[yellow]Didn't catch that.[/]")

    def stopped(self) -> None:
        """The user stopped the current command."""
        self._end_reply()
        self._console.print("[dim]Stopped.[/]")

    def interrupted(self) -> None:
        """The user talked over the reply."""
        self._end_reply()
        self._console.print("[dim]↳ interrupted[/]")

    def error(self, message: str) -> None:
        """Show an error; the conversation continues."""
        self._end_reply()
        self._console.print(f"[red]✗ {_escape(message)}[/]", highlight=False)

    def _end_reply(self) -> None:
        if self._replying:
            self._console.print()
            self._replying = False


def _escape(text: str) -> str:
    return text.replace("[", r"\[")


async def stdin_lines() -> AsyncIterator[str]:
    """Lines typed on stdin, without blocking the event loop.

    A daemon thread does the blocking reads, so Ctrl+C exits immediately even while
    waiting for input.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def post(item: str | None) -> None:
        if not loop.is_closed():
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def read() -> None:
        try:
            for line in sys.stdin:
                post(line)
        finally:
            post(None)

    threading.Thread(target=read, name="yarvis-stdin", daemon=True).start()
    while (line := await queue.get()) is not None:
        yield line
