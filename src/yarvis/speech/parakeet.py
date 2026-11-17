"""Speech-to-text with NVIDIA Parakeet: on the Apple GPU (MLX), or on any processor (ONNX).

Parakeet TDT 0.6B v2 makes fewer mistakes than any Whisper model: 6.05 % of words wrong on
average on the Open ASR leaderboard, against 7.83 % for Whisper large-v3-turbo and more for
the small Whisper models that are fast enough on a CPU. Unlike Whisper it does not pad
every utterance to 30 seconds, so short commands are also transcribed quickly: through
ONNX Runtime it runs many times faster than real time on an ordinary desktop processor,
so Windows and Linux get the same model as an Apple Silicon Mac, without a GPU.

The weights are downloaded by Yarvis itself and verified (see ``yarvis.models``); this
module only loads files that are already in place and never touches the network.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

import numpy as np

from yarvis.models import missing, parakeet_dir, parakeet_files

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

    from yarvis.models import ParakeetBuild
    from yarvis.speech.base import AudioArray

logger = logging.getLogger(__name__)

MODEL_NAME: Final = "parakeet-tdt-0.6b-v2"
SAMPLE_RATE: Final[Literal[16000]] = 16_000
_SHORTEST: Final = SAMPLE_RATE // 10  # less than 0.1 s is not a word


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


class Recognizer(Protocol):
    """onnx-asr's loaded model (only what Yarvis uses)."""

    def recognize(self, waveform: npt.NDArray[np.float32], *, sample_rate: Literal[16000]) -> str:
        """Transcribe one waveform."""
        ...


def _load_onnx(directory: Path) -> ParakeetEngine:  # pragma: no cover - a 2.5 GB model
    # Imported when needed (it loads ONNX Runtime), by name: ONNX Runtime has no type hints.
    onnx_asr: Any = importlib.import_module("onnx_asr")
    # The directory exists with the verified files, so onnx-asr loads them offline: it
    # never reaches Hugging Face itself.
    model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v2", directory, providers=["CPUExecutionProvider"]
    )
    return OnnxParakeet(cast("Recognizer", model))


class OnnxParakeet:
    """Parakeet through onnx-asr (ONNX Runtime), on the processor."""

    def __init__(self, model: Recognizer) -> None:
        self._model = model

    def transcribe(self, audio: AudioArray) -> str:
        """Transcribe 16 kHz mono audio (blocking)."""
        if audio.size < _SHORTEST:
            return ""
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        return self._model.recognize(samples, sample_rate=SAMPLE_RATE).strip()


_LOADERS: Final[dict[ParakeetBuild, Callable[[Path], ParakeetEngine]]] = {
    "mlx": _load_mlx,
    "onnx": _load_onnx,
}
_WHERE: Final[dict[ParakeetBuild, str]] = {"mlx": "the Apple GPU", "onnx": "the processor"}


class ParakeetSTT:
    """``SpeechToText`` backed by Parakeet, on the Apple GPU or on the processor.

    MLX is not built to be driven from several threads (arrays belong to the thread that
    created their stream), so loading and every transcription run on one dedicated worker
    thread, one at a time.
    """

    model_name: Final = MODEL_NAME

    def __init__(
        self,
        *,
        cache_dir: Path,
        build: ParakeetBuild = "mlx",
        loader: Callable[[Path], ParakeetEngine] | None = None,
    ) -> None:
        self._build: ParakeetBuild = build
        self.device = "mps" if build == "mlx" else "cpu"
        """Where the model runs, for the benchmark and the status line."""
        self._directory = parakeet_dir(cache_dir, build)
        self._loader = loader or _LOADERS[build]
        self._engine: ParakeetEngine | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yarvis-parakeet")

    def load(self) -> None:
        """Load the downloaded model. Blocking.

        Raises:
            ParakeetModelMissingError: If the files are not downloaded yet.
        """
        self._worker.submit(self._load_on_worker).result()

    async def transcribe(self, audio: AudioArray) -> str:
        """Transcribe 16 kHz mono audio on the worker thread."""
        return await asyncio.wrap_future(self._worker.submit(self._transcribe_on_worker, audio))

    def _load_on_worker(self) -> ParakeetEngine:
        if self._engine is None:
            if missing(parakeet_files(self._build), self._directory):
                msg = "Parakeet is not downloaded yet. Run: yarvis models download"
                raise ParakeetModelMissingError(msg)
            self._engine = self._loader(self._directory)
            logger.info("Speech recognition: %s on %s", MODEL_NAME, _WHERE[self._build])
        return self._engine

    def _transcribe_on_worker(self, audio: AudioArray) -> str:
        return self._load_on_worker().transcribe(audio)
