"""Text-to-speech with Kokoro (82M parameters, ONNX, Apache-2.0)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from yarvis.models import KOKORO_MODEL, KOKORO_VOICES, kokoro_dir

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yarvis.speech.base import AudioArray

SAMPLE_RATE: Final = 24_000

# Kokoro voice ids start with a language letter: a = American, b = British English.
_LANGUAGES: Final = {"a": "en-us", "b": "en-gb"}


class KokoroLike(Protocol):
    """The part of ``kokoro_onnx.Kokoro`` that Yarvis uses."""

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        """Synthesize ``text``; returns (samples, sample_rate)."""
        ...

    def get_voices(self) -> list[str]:
        """Available voice ids."""
        ...


class UnknownVoiceError(ValueError):
    """The configured voice does not exist or is not an English voice."""


def _load_kokoro(model: Path, voices: Path) -> KokoroLike:
    from kokoro_onnx import Kokoro  # noqa: PLC0415 - heavy import, load lazily

    engine: KokoroLike = Kokoro(str(model), str(voices))
    return engine


class KokoroTTS:
    """``TextToSpeech`` backed by Kokoro. Synthesis runs in a worker thread."""

    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        cache_dir: Path,
        voice: str,
        speed: float = 1.0,
        loader: Callable[[Path, Path], KokoroLike] = _load_kokoro,
    ) -> None:
        language = _LANGUAGES.get(voice[:1])
        if language is None:
            msg = f"voice {voice!r} is not an English voice (ids start with 'a' or 'b')"
            raise UnknownVoiceError(msg)
        self.voice = voice
        self.speed = speed
        self._language = language
        self._directory = kokoro_dir(cache_dir)
        self._loader = loader
        self._engine: KokoroLike | None = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        """Load the model files (they must already be downloaded). Blocking."""
        if self._engine is not None:
            return
        engine = self._loader(
            self._directory / KOKORO_MODEL.name, self._directory / KOKORO_VOICES.name
        )
        voices = engine.get_voices()
        if self.voice not in voices:
            english = ", ".join(v for v in sorted(voices) if v[:1] in _LANGUAGES)
            msg = f"unknown voice {self.voice!r}. English voices: {english}"
            raise UnknownVoiceError(msg)
        self._engine = engine

    async def synthesize(self, text: str) -> AudioArray:
        """Return 24 kHz mono audio for ``text``."""
        async with self._lock:
            if self._engine is None:
                await asyncio.to_thread(self.load)
            return await asyncio.to_thread(self._synthesize_blocking, text)

    def _synthesize_blocking(self, text: str) -> AudioArray:
        if self._engine is None:
            msg = "Kokoro is not loaded"
            raise RuntimeError(msg)
        samples, rate = self._engine.create(
            text, voice=self.voice, speed=self.speed, lang=self._language
        )
        if rate != SAMPLE_RATE:  # pragma: no cover - Kokoro v1.0 is always 24 kHz
            msg = f"unexpected Kokoro sample rate {rate}"
            raise RuntimeError(msg)
        return np.asarray(samples, dtype=np.float32).reshape(-1)
