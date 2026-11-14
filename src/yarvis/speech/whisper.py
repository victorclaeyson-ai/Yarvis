"""Speech-to-text with Whisper via ``faster-whisper`` (CTranslate2, CPU int8).

Models are pinned to an exact Hugging Face commit and loaded from the local cache when
present, so after the first download Yarvis needs no network for speech recognition.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from yarvis.models import WHISPER_REVISIONS, whisper_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from yarvis.config.models import SttModel
    from yarvis.speech.base import AudioArray

# Whisper hallucinates text ("Thank you.") on noise; drop segments it is unsure contain speech.
NO_SPEECH_THRESHOLD: Final = 0.6


class _Segment(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def no_speech_prob(self) -> float: ...


class WhisperModelLike(Protocol):
    """The part of ``faster_whisper.WhisperModel`` that Yarvis uses."""

    def transcribe(
        self,
        audio: AudioArray,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        condition_on_previous_text: bool = True,
        without_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> tuple[Iterable[_Segment], Any]:
        """Transcribe audio."""
        ...


logger = logging.getLogger(__name__)

SttDevice = Literal["cpu", "mps"]

# int8 is the fastest exact-enough type on CPU; on the Apple GPU let CTranslate2 choose.
_COMPUTE_TYPES: Final[dict[SttDevice, str]] = {"cpu": "int8", "mps": "default"}


def _load_faster_whisper(
    name: SttModel, root: Path, *, local_only: bool, device: SttDevice = "cpu"
) -> WhisperModelLike:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import ctranslate2  # noqa: PLC0415 - heavy import, load lazily
    from faster_whisper import WhisperModel  # noqa: PLC0415
    from huggingface_hub.utils import logging as hub_logging  # noqa: PLC0415

    # Both libraries print hints straight to the terminal; keep the conversation clean.
    ctranslate2.set_log_level(logging.ERROR)
    hub_logging.set_verbosity_error()

    model: WhisperModelLike = WhisperModel(
        name,
        device=device,
        compute_type=_COMPUTE_TYPES[device],
        download_root=str(root),
        revision=WHISPER_REVISIONS[name],
        local_files_only=local_only,
    )
    return model


class WhisperSTT:
    """``SpeechToText`` backed by a local Whisper model."""

    def __init__(
        self,
        model: SttModel,
        *,
        cache_dir: Path,
        vocabulary: Sequence[str] = (),
        device: SttDevice = "cpu",
        loader: Callable[..., WhisperModelLike] = _load_faster_whisper,
    ) -> None:
        self.model_name = model
        self.device: SttDevice = device
        """Requested device; after ``load`` it is the device actually in use."""
        # Whisper conditions on the prompt text, which strongly biases it towards these
        # spellings. Phrased as a plain list so it does not leak into transcripts.
        self._prompt = ", ".join(vocabulary) + "." if vocabulary else None
        self._root = whisper_dir(cache_dir)
        self._loader = loader
        self._model: WhisperModelLike | None = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        """Load the model, downloading the pinned revision on first use. Blocking.

        If the Apple GPU (MPS) backend is requested but fails, falls back to the CPU.
        """
        if self._model is not None:
            return
        if self.device != "cpu":
            try:
                self._model = self._load_on(self.device)
            except Exception as exc:  # noqa: BLE001 - experimental backend: any failure
                logger.warning("Whisper could not use %s (%s); using the CPU", self.device, exc)
                self.device = "cpu"
            else:
                return
        self._model = self._load_on("cpu")

    def _load_on(self, device: SttDevice) -> WhisperModelLike:
        try:
            return self._loader(self.model_name, self._root, local_only=True, device=device)
        except Exception:  # noqa: BLE001 - not cached yet (error type varies by hub version)
            return self._loader(self.model_name, self._root, local_only=False, device=device)

    async def transcribe(self, audio: AudioArray) -> str:
        """Transcribe 16 kHz mono audio. Runs in a worker thread."""
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self.load)
            return await asyncio.to_thread(self._transcribe_blocking, audio)

    def _transcribe_blocking(self, audio: AudioArray) -> str:
        if self._model is None:
            msg = "Whisper is not loaded"
            raise RuntimeError(msg)
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self._prompt,
        )
        texts = [
            segment.text.strip()
            for segment in segments
            if segment.no_speech_prob < NO_SPEECH_THRESHOLD
        ]
        return " ".join(text for text in texts if text)
