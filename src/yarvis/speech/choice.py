"""Which speech recognition engine runs on this machine.

With an AssemblyAI key stored (and ``cloud_stt`` on), AssemblyAI transcribes while you
talk, and the engine on this computer is its backup. That engine is chosen by
``stt = "auto"``: Parakeet, which mishears far less than the Whisper models that are fast
enough to run locally, on the GPU of an Apple Silicon Mac (MLX) and on the processor of
any other computer (ONNX Runtime). If Parakeet cannot load, Yarvis falls back to Whisper
rather than going deaf.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import platform
from typing import TYPE_CHECKING, Literal

from yarvis.security.secrets import SecretName
from yarvis.speech.assemblyai import AssemblyAISTT
from yarvis.speech.parakeet import ParakeetSTT
from yarvis.speech.whisper import COMMAND_WORDS, WhisperSTT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pydantic import SecretStr

    from yarvis.config.models import Config, VoiceConfig
    from yarvis.models import ParakeetBuild
    from yarvis.runtime import LoadableSTT
    from yarvis.security.secrets import SecretStore
    from yarvis.speech.base import AudioArray

logger = logging.getLogger(__name__)

type SttEngine = Literal["parakeet", "faster-whisper"]


def parakeet_build(system: str, machine: str) -> ParakeetBuild:
    """MLX on an Apple Silicon Mac (its GPU); ONNX on every other computer (the CPU)."""
    return "mlx" if system == "Darwin" and machine == "arm64" else "onnx"


def parakeet_package(build: ParakeetBuild) -> str:
    """The Python package that runs a Parakeet build."""
    return "parakeet_mlx" if build == "mlx" else "onnx_asr"


def parakeet_supported(system: str, machine: str, *, installed: bool) -> bool:
    """Parakeet runs anywhere its package is installed (``installed``)."""
    del system, machine  # every platform has a build now
    return installed


def parakeet_available() -> bool:
    """Whether Parakeet can run on this machine."""
    build = this_build()
    installed = importlib.util.find_spec(parakeet_package(build)) is not None
    return parakeet_supported(platform.system(), platform.machine(), installed=installed)


def this_build() -> ParakeetBuild:
    """The Parakeet build for this computer."""
    return parakeet_build(platform.system(), platform.machine())


def choose_engine(voice: VoiceConfig, *, parakeet_ok: bool) -> SttEngine:
    """The engine to use for the configured ``stt`` setting."""
    if voice.stt == "faster-whisper":
        return "faster-whisper"
    if parakeet_ok:
        return "parakeet"
    if voice.stt == "parakeet":
        logger.warning("Parakeet is not installed; using Whisper instead")
    return "faster-whisper"


def describe(voice: VoiceConfig, *, cloud: bool = False) -> str:
    """A short name for the engine a (resolved) configuration uses.

    ``cloud``: an AssemblyAI key is stored (it only counts when ``cloud_stt`` is on).
    """
    local = "parakeet" if voice.stt == "parakeet" else f"whisper {voice.stt_model}"
    if cloud and voice.cloud_stt:
        return f"assemblyai {voice.cloud_region} (backup: {local})"
    return local


def cloud_key(voice: VoiceConfig, secrets: SecretStore) -> SecretStr | None:
    """The AssemblyAI key, if it is stored and cloud speech recognition is on."""
    if not voice.cloud_stt:
        return None
    resolved = secrets.get(SecretName.ASSEMBLYAI)
    return resolved.value if resolved is not None else None


def spoken_names(config: Config) -> tuple[str, ...]:
    """Names the user will say that a recognizer should spell right (boosted in the cloud)."""
    assistant = config.assistant
    names = (assistant.name, assistant.user_name, assistant.location)
    return (*config.voice.vocabulary, *COMMAND_WORDS, *(name for name in names if name))


def make_stt(
    voice: VoiceConfig,
    cache_dir: Path,
    *,
    parakeet_ok: bool,
    build: ParakeetBuild | None = None,
    cloud: SecretStr | None = None,
    terms: Iterable[str] = (),
) -> LoadableSTT:
    """Build the speech recognizer for this machine and configuration.

    ``cloud`` is the AssemblyAI key (see ``cloud_key``): with it, AssemblyAI transcribes
    and the engine on this computer is the backup. ``terms`` are names to spell right.
    """
    local = _local_stt(voice, cache_dir, parakeet_ok=parakeet_ok, build=build)
    if cloud is None or not voice.cloud_stt:
        return local
    return AssemblyAISTT(cloud, local, region=voice.cloud_region, terms=terms)


def _local_stt(
    voice: VoiceConfig,
    cache_dir: Path,
    *,
    parakeet_ok: bool,
    build: ParakeetBuild | None,
) -> LoadableSTT:

    def whisper() -> LoadableSTT:
        return WhisperSTT(
            voice.stt_model,
            cache_dir=cache_dir,
            vocabulary=voice.vocabulary,
            device=voice.stt_device,
        )

    if choose_engine(voice, parakeet_ok=parakeet_ok) == "parakeet":
        parakeet = ParakeetSTT(cache_dir=cache_dir, build=build or this_build())
        return FallbackSTT(parakeet, whisper)
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
