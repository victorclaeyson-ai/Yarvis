"""In-memory fakes for external systems (keychain, audio hardware)."""

from __future__ import annotations

import asyncio
import stat
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import keyring.errors
import numpy as np
import pytest

from yarvis.audio.devices import AudioBackendError, AudioDevice

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
    from pathlib import Path

    from typer.testing import Result

    from yarvis.ears import Interruption
    from yarvis.listen import PauseHooks, Utterance
    from yarvis.llm.base import Message, SystemPrompt, ToolCall, ToolSpec, WebSearch
    from yarvis.speech.base import AudioArray
    from yarvis.telemetry import LinkState
    from yarvis.usage import ClaudeUsage

POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="about POSIX permission bits, which Windows does not have"
)


def posix_mode(path: Path) -> int | None:
    """The permission bits of ``path``; None on Windows, which has none (every file is 0o666)."""
    return None if sys.platform == "win32" else stat.S_IMODE(path.stat().st_mode)


def symlink_or_skip(link: Path, target: Path) -> None:
    """Make ``link`` point to ``target``, or skip: on Windows that takes Developer Mode."""
    try:
        link.symlink_to(target)
    except OSError as exc:  # WinError 1314: a required privilege is not held
        pytest.skip(f"symbolic links cannot be created here ({exc.strerror})")


VALID_KEY = "sk-ant-test-0123456789abcdefghij"  # gitleaks:allow
ASSEMBLYAI_KEY = "test" * 8
"""A well-formed AssemblyAI key that is obviously fake (and no secret scanner's match)."""


@dataclass
class FakeKeyring:
    """Dict-backed stand-in for the OS keychain."""

    store: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])
    available: bool = True
    fail_reads: bool = False

    def get_password(self, service: str, username: str) -> str | None:
        if self.fail_reads:
            raise keyring.errors.KeyringError("locked")
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self.store[(service, username)]
        except KeyError:
            raise keyring.errors.PasswordDeleteError("not found") from None

    def is_available(self) -> bool:
        return self.available

    def describe(self) -> str:
        return "fake.Keyring"


MIC = AudioDevice(
    index=0,
    name="Test Microphone",
    max_input_channels=1,
    max_output_channels=0,
    default_samplerate=48_000.0,
)
SPEAKER = AudioDevice(
    index=1,
    name="Test Speakers",
    max_input_channels=0,
    max_output_channels=2,
    default_samplerate=48_000.0,
)


@dataclass
class FakeAudioProbe:
    """Configurable set of audio devices."""

    all_devices: Sequence[AudioDevice] = (MIC, SPEAKER)
    input_device: AudioDevice | None = MIC
    output_device: AudioDevice | None = SPEAKER
    broken: bool = False

    def devices(self) -> Sequence[AudioDevice]:
        self._check()
        return self.all_devices

    def default_input(self) -> AudioDevice | None:
        self._check()
        return self.input_device

    def default_output(self) -> AudioDevice | None:
        self._check()
        return self.output_device

    def _check(self) -> None:
        if self.broken:
            raise AudioBackendError("the PortAudio library is not installed")


# --------------------------------------------------------------------------- speech + LLM fakes


class FakeLLM:
    """Streams scripted replies; records what it was asked."""

    def __init__(
        self, *replies: Sequence[str | ToolCall | WebSearch], error: Exception | None = None
    ) -> None:
        self.replies = [list(reply) for reply in replies]
        self.error = error
        self.calls: list[tuple[list[Message], str, int]] = []
        self.offered: list[list[str]] = []  # tool names offered on each call
        self.allowed: list[bool] = []  # whether each call let the model call tools
        self.systems: list[SystemPrompt] = []  # the system prompt of each call, in its parts
        self.warm_ups = 0

    async def warm_up(self) -> None:
        self.warm_ups += 1

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: SystemPrompt,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        allow_tool_calls: bool = True,
    ) -> AsyncGenerator[str | ToolCall | WebSearch]:
        self.calls.append((list(messages), system.text, max_tokens))
        self.systems.append(system)
        self.offered.append([tool.name for tool in tools])
        self.allowed.append(allow_tool_calls)
        deltas = self.replies.pop(0) if self.replies else []
        for delta in deltas:
            await asyncio.sleep(0)
            yield delta
        if self.error is not None:
            raise self.error


@dataclass
class RecordingReporter:
    """A ``telemetry.Reporter`` that keeps every report."""

    claude_calls: list[ClaudeUsage] = field(default_factory=list["ClaudeUsage"])
    speech_calls: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    health_calls: list[tuple[str, LinkState, str]] = field(
        default_factory=list[tuple[str, "LinkState", str]]
    )

    def claude(self, usage: ClaudeUsage) -> None:
        self.claude_calls.append(usage)

    def speech(self, model: str, seconds: float) -> None:
        self.speech_calls.append((model, seconds))

    def health(self, key: str, state: LinkState, detail: str) -> None:
        self.health_calls.append((key, state, detail))


class FakeSTT:
    def __init__(self, text: str = "hello there") -> None:
        self.text = text
        self.received: list[int] = []

    async def transcribe(self, audio: AudioArray) -> str:
        self.received.append(len(audio))
        return self.text


class FakeTTS:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AudioArray:
        self.spoken.append(text)
        return np.zeros(len(text), dtype=np.float32)


class FakeSpeaker:
    def __init__(self, device_name: str | None = None) -> None:
        self.device_name = device_name
        self.played: list[int] = []
        self.drained = 0
        self.stopped = 0

    def stop(self) -> None:
        self.stopped += 1

    async def play(self, audio: AudioArray) -> None:
        self.played.append(len(audio))

    async def drain(self) -> None:
        self.drained += 1


class RecordingObserver:
    def __init__(self) -> None:
        self.audio_started = 0
        self.transcripts: list[str] = []
        self.deltas: list[str] = []

    def on_transcript(self, text: str) -> None:
        self.transcripts.append(text)

    def on_reply_delta(self, text: str) -> None:
        self.deltas.append(text)

    def on_audio_started(self) -> None:
        self.audio_started += 1


class TickingClock:
    """Deterministic clock: every call advances by ``step`` seconds."""

    def __init__(self, start: float = 100.0, step: float = 0.01) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class Invoke(Protocol):
    """Runs the CLI in-process with test services."""

    def __call__(self, *args: str, stdin: str | None = None) -> Result: ...


class ListenFn(Protocol):
    def __call__(
        self, hooks: PauseHooks, *, follow_up: bool = False
    ) -> Awaitable[Utterance | None]: ...


class FakeEars:
    """``Ears`` driven by test callables; records what the assistant asked for."""

    def __init__(
        self,
        listen: ListenFn | None = None,
        *,
        wake: Callable[[], Awaitable[Sequence[AudioArray]]] | None = None,
        interruption: Callable[[], Awaitable[Interruption]] | None = None,
    ) -> None:
        self._listen = listen
        self._wake = wake
        self._interruption = interruption
        self.prefixes: list[Sequence[AudioArray] | None] = []
        self.timeouts: list[float | None] = []
        self.sessions = 0
        self.wake_word_only: list[bool] = []

    @property
    def wake_word(self) -> bool:
        return self._wake is not None

    @property
    def barge_in(self) -> bool:
        return self._interruption is not None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None]:
        self.sessions += 1
        yield

    async def listen(
        self,
        hooks: PauseHooks,
        *,
        follow_up: bool = False,
        prefix: Sequence[AudioArray] | None = None,
        timeout_s: float | None = None,
    ) -> Utterance | None:
        self.prefixes.append(prefix)
        self.timeouts.append(timeout_s)
        if self._listen is None:
            return None
        return await self._listen(hooks, follow_up=follow_up)

    async def wake(self) -> Sequence[AudioArray]:
        assert self._wake is not None
        return await self._wake()

    async def interruption(self, *, wake_word_only: bool = False) -> Interruption:
        assert self._interruption is not None
        self.wake_word_only.append(wake_word_only)
        return await self._interruption()


class FakeWakeWord:
    """Detects the wake word in any frame whose first sample is ``TRIGGER``."""

    TRIGGER = 0.5

    def __init__(self) -> None:
        self.loaded = False
        self.resets = 0
        self.heard = 0

    def load(self) -> None:
        self.loaded = True

    def reset(self) -> None:
        self.resets += 1

    def detected(self, frame: AudioArray) -> bool:
        self.heard += 1
        return bool(frame.size) and float(frame[0]) == self.TRIGGER
