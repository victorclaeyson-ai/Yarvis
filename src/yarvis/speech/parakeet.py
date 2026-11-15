"""Speech-to-text with NVIDIA Parakeet on the Apple GPU (MLX).

Parakeet TDT 0.6B v2 makes fewer mistakes than any Whisper model: 6.05 % of words wrong on
average on the Open ASR leaderboard, against 7.83 % for Whisper large-v3-turbo and more for
the small Whisper models that are fast enough on a CPU. Unlike Whisper it does not pad
every utterance to 30 seconds, so short commands are also transcribed quickly. It needs
Apple Silicon; elsewhere Yarvis uses Whisper.

The weights are downloaded by Yarvis itself and verified (see ``yarvis.models``); this
module only loads files that are already in place and never touches the network.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from yarvis.models import PARAKEET_FILES, missing, parakeet_dir

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yarvis.speech.base import AudioArray

logger = logging.getLogger(__name__)

MODEL_NAME: Final = "parakeet-tdt-0.6b-v2"


class ParakeetModelMissingError(RuntimeError):
    """The Parakeet files have not been downloaded (or are incomplete)."""


class ParakeetEngine(Protocol):
    """A loaded model that turns 16 kHz mono audio into text."""

    def transcribe(self, audio: AudioArray) -> str:
        """Transcribe (blocking)."""
        ...


def _load_mlx(directory: Path) -> ParakeetEngine:  # pragma: no cover - needs Apple Silicon
    # Imported by name: MLX only exists on Apple Silicon, so type checkers elsewhere
    # cannot resolve it, and the package ships without type hints anyway.
    mx: Any = importlib.import_module("mlx.core")
    trees: Any = importlib.import_module("mlx.utils")
    utils: Any = importlib.import_module("parakeet_mlx.utils")
    audio_module: Any = importlib.import_module("parakeet_mlx.audio")

    # What parakeet_mlx.from_pretrained does, minus its attempt to reach Hugging Face.
    model: Any = utils.from_config(json.loads((directory / "config.json").read_text()))
    model.load_weights(str(directory / "model.safetensors"))  # safetensors: data only
    weights = [(name, w.astype(mx.bfloat16)) for name, w in trees.tree_flatten(model.parameters())]
    model.update(trees.tree_unflatten(weights))
    preprocessing: Any = model.preprocessor_config
    shortest = int(preprocessing.hop_length)

    class _MlxParakeet:
        def transcribe(self, audio: AudioArray) -> str:
            if audio.size < shortest:
                return ""
            # float32, as parakeet_mlx's own loader produces: its spectrogram code reads
            # the complex FFT output as the input's dtype, which only works for float32.
            samples = mx.array(np.ascontiguousarray(audio, dtype=np.float32))
            mel = audio_module.get_logmel(samples, preprocessing)
            result: Any = model.generate(mel)[0]
            return str(result.text).strip()

    return _MlxParakeet()


class ParakeetSTT:
    """``SpeechToText`` backed by Parakeet on the Apple GPU.

    MLX is not built to be driven from several threads (arrays belong to the thread that
    created their stream), so loading and every transcription run on one dedicated worker
    thread, one at a time.
    """

    model_name: Final = MODEL_NAME
    device: Final = "mps"
    """Where the model runs (the Apple GPU), for the benchmark and the status line."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        loader: Callable[[Path], ParakeetEngine] = _load_mlx,
    ) -> None:
        self._directory = parakeet_dir(cache_dir)
        self._loader = loader
        self._engine: ParakeetEngine | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yarvis-parakeet")

    def load(self) -> None:
        """Load the downloaded model onto the GPU. Blocking.

        Raises:
            ParakeetModelMissingError: If the files are not downloaded yet.
        """
        self._worker.submit(self._load_on_worker).result()

    async def transcribe(self, audio: AudioArray) -> str:
        """Transcribe 16 kHz mono audio on the worker thread."""
        return await asyncio.wrap_future(self._worker.submit(self._transcribe_on_worker, audio))

    def _load_on_worker(self) -> ParakeetEngine:
        if self._engine is None:
            if missing(PARAKEET_FILES, self._directory):
                msg = "Parakeet is not downloaded yet. Run: yarvis models download"
                raise ParakeetModelMissingError(msg)
            self._engine = self._loader(self._directory)
            logger.info("Speech recognition: %s on the Apple GPU", MODEL_NAME)
        return self._engine

    def _transcribe_on_worker(self, audio: AudioArray) -> str:
        return self._load_on_worker().transcribe(audio)
