"""In-memory fakes for external systems (keychain, audio hardware)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import keyring.errors
import numpy as np

from yarvis.audio.devices import AudioBackendError, AudioDevice

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from typer.testing import Result

    from yarvis.llm.base import Message
    from yarvis.speech.base import AudioArray

VALID_KEY = "sk-ant-test-0123456789abcdefghij"  # gitleaks:allow


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

    def __init__(self, *replies: list[str], error: Exception | None = None) -> None:
        self.replies = list(replies)
        self.error = error
        self.calls: list[tuple[list[Message], str, int]] = []

    async def stream(
        self, messages: Sequence[Message], *, system: str, max_tokens: int
    ) -> AsyncIterator[str]:
        self.calls.append((list(messages), system, max_tokens))
        deltas = self.replies.pop(0) if self.replies else []
        for delta in deltas:
            await asyncio.sleep(0)
            yield delta
        if self.error is not None:
            raise self.error


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
    def __init__(self) -> None:
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
