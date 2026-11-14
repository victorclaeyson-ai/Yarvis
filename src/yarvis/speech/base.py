"""Interfaces for the speech components. Every backend implements one of these."""

from __future__ import annotations

from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

AudioArray = NDArray[np.float32]
"""Mono audio, float32 samples in [-1, 1]."""

MIC_SAMPLE_RATE: Final = 16_000
"""Sample rate for everything on the input side (VAD, speech-to-text)."""

FRAME_SAMPLES: Final = 512
"""Samples per input frame: 32 ms at 16 kHz, the frame size Silero VAD expects."""


class VoiceActivityDetector(Protocol):
    """Estimates the probability that a frame contains speech."""

    def probability(self, frame: AudioArray) -> float:
        """Speech probability for one ``FRAME_SAMPLES`` frame at 16 kHz."""
        ...

    def reset(self) -> None:
        """Forget internal state before a new utterance."""
        ...


class SpeechToText(Protocol):
    """Transcribes a complete utterance."""

    async def transcribe(self, audio: AudioArray) -> str:
        """Return the text spoken in ``audio`` (16 kHz mono)."""
        ...


class TextToSpeech(Protocol):
    """Synthesizes speech for one sentence at a time."""

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this engine produces."""
        ...

    async def synthesize(self, text: str) -> AudioArray:
        """Return audio for ``text``."""
        ...


class AudioSink(Protocol):
    """Plays audio. ``play`` queues and returns immediately; ``drain`` waits for the end."""

    async def play(self, audio: AudioArray) -> None:
        """Queue audio for playback."""
        ...

    async def drain(self) -> None:
        """Wait until everything queued has been played."""
        ...

    def stop(self) -> None:
        """Stop playback immediately and discard anything queued."""
        ...
