"""Which speech recognition engine runs on this machine.

``stt = "auto"`` picks Parakeet on Apple Silicon (more accurate and faster, on the GPU) and
Whisper everywhere else. If Parakeet cannot load, Yarvis falls back to Whisper rather
than going deaf.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import platform
from typing import TYPE_CHECKING, Literal

from yarvis.speech.parakeet import ParakeetSTT
from yarvis.speech.whisper import WhisperSTT

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yarvis.config.models import VoiceConfig
    from yarvis.runtime import LoadableSTT
    from yarvis.speech.base import AudioArray

logger = logging.getLogger(__name__)

type SttEngine = Literal["parakeet", "faster-whisper"]


def parakeet_supported(system: str, machine: str, *, installed: bool) -> bool:
    """Parakeet needs an Apple Silicon Mac and the ``parakeet-mlx`` package."""
    return system == "Darwin" and machine == "arm64" and installed


def parakeet_available() -> bool:
    """Whether Parakeet can run on this machine."""
    installed = importlib.util.find_spec("parakeet_mlx") is not None
    return parakeet_supported(platform.system(), platform.machine(), installed=installed)


def choose_engine(voice: VoiceConfig, *, parakeet_ok: bool) -> SttEngine:
    """The engine to use for the configured ``stt`` setting."""
    if voice.stt == "faster-whisper":
        return "faster-whisper"
    if parakeet_ok:
        return "parakeet"
    if voice.stt == "parakeet":
        logger.warning("Parakeet needs an Apple Silicon Mac; using Whisper instead")
    return "faster-whisper"


def describe(voice: VoiceConfig) -> str:
    """A short name for the engine a (resolved) configuration uses."""
    if voice.stt == "parakeet":
        return "parakeet"
    return f"whisper {voice.stt_model}"


def make_stt(voice: VoiceConfig, cache_dir: Path, *, parakeet_ok: bool) -> LoadableSTT:
    """Build the speech recognizer for this machine and configuration."""

    def whisper() -> LoadableSTT:
        return WhisperSTT(
            voice.stt_model,
            cache_dir=cache_dir,
            vocabulary=voice.vocabulary,
            device=voice.stt_device,
        )

    if choose_engine(voice, parakeet_ok=parakeet_ok) == "parakeet":
        return FallbackSTT(ParakeetSTT(cache_dir=cache_dir), whisper)
    return whisper()


class FallbackSTT:
    """Uses ``primary``, and switches to ``fallback`` for good if it breaks.

    Breaking means failing to load or to transcribe. The reason is logged, and a request
    that failed is transcribed again with the fallback, not lost.
    """

    def __init__(self, primary: LoadableSTT, fallback: Callable[[], LoadableSTT]) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active: LoadableSTT | None = None
        self._lock = asyncio.Lock()

    @property
    def active(self) -> LoadableSTT:
        """The engine in use (the primary until it has failed)."""
        return self._active or self._primary

    @property
    def device(self) -> str:
        """Where the engine in use runs."""
        return str(getattr(self.active, "device", "cpu"))

    @property
    def model_name(self) -> str:
        """The model in use."""
        return str(getattr(self.active, "model_name", "unknown"))

    def load(self) -> None:
        """Load the primary engine, or the fallback if that fails. Blocking."""
        if self._active is not None:
            return
        try:
            self._primary.load()
        except Exception as exc:  # noqa: BLE001 - any failure: Yarvis must still hear you
            self._switch(exc)
        else:
            self._active = self._primary

    async def transcribe(self, audio: AudioArray) -> str:
        """Transcribe with the engine in use (loading it first if needed)."""
        if self._active is None:
            async with self._lock:
                if self._active is None:
                    await asyncio.to_thread(self.load)
        engine = self.active
        if engine is not self._primary:
            return await engine.transcribe(audio)
        try:
            return await engine.transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - same: switch, and hear this request too
            async with self._lock:
                if self._active is self._primary:
                    await asyncio.to_thread(self._switch, exc)
        return await self.active.transcribe(audio)

    def _switch(self, reason: Exception) -> None:
        logger.warning("Speech recognition fell back to Whisper: %s", reason)
        fallback = self._fallback()
        fallback.load()
        self._active = fallback
