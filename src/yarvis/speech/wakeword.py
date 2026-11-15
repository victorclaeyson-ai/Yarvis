"""Wake-word detection with openWakeWord models, run directly on ONNX Runtime.

Pipeline, for every 80 ms (1280 samples at 16 kHz) of audio:

    audio ─▶ melspectrogram.onnx ─▶ 76-frame window ─▶ embedding_model.onnx (96 values)
          ─▶ last 16 embeddings ─▶ hey_jarvis.onnx ─▶ score in [0, 1]

This mirrors ``openwakeword.utils.AudioFeatures`` (Apache-2.0) without its dependencies
(scikit-learn, scipy, tflite). Everything runs locally; audio never leaves the machine.

Licensing note: the openWakeWord *code* is Apache-2.0, but its pre-trained *models* are
CC BY-NC-SA 4.0 (non-commercial). Yarvis downloads them at runtime instead of
redistributing them; a commercial build would need its own trained model (see ADR 0006).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
import onnxruntime

from yarvis.models import WAKE_WORD_FILES, ModelFile, wake_word_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from yarvis.speech.base import AudioArray

CHUNK_SAMPLES: Final = 1280  # 80 ms at 16 kHz: the model's step size
_MEL_CONTEXT: Final = 160 * 3  # extra samples the melspectrogram needs on the left
_WINDOW: Final = 76  # mel frames per embedding
_MEL_BINS: Final = 32
_MEL_MAX: Final = 10 * 97  # ~10 s of mel frames
_FEATURES: Final = 16  # embeddings per wake-word prediction
_FEATURES_MAX: Final = 120  # ~10 s of embeddings
_WARMUP_CHUNKS: Final = 5  # ignore scores until the buffers hold real audio
_EMBEDDING: Final = 96

Pcm = np.ndarray[Any, np.dtype[np.int16]]
Floats = np.ndarray[Any, np.dtype[np.float32]]


class InferenceSession(Protocol):
    """The part of ``onnxruntime.InferenceSession`` used here."""

    def run(self, output_names: Sequence[str] | None, inputs: Mapping[str, Any]) -> list[Any]:
        """Run the model."""
        ...


def _session(path: Path) -> InferenceSession:
    ort = cast("Any", onnxruntime)  # onnxruntime ships without type information
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class WakeWordDetector:
    """Streaming detector. Feed it microphone frames of any size; it reports scores.

    Not thread-safe; use one instance per audio stream.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        threshold: float = 0.5,
        files: tuple[ModelFile, ModelFile, ModelFile] = WAKE_WORD_FILES,
        session_factory: Callable[[Path], InferenceSession] = _session,
    ) -> None:
        if not 0 < threshold < 1:
            msg = "threshold must be between 0 and 1"
            raise ValueError(msg)
        self.threshold = threshold
        self._directory = wake_word_dir(cache_dir)
        self._files = files
        self._session_factory = session_factory
        self._models: _Models | None = None
        self._pending = np.zeros(0, dtype=np.int16)
        self._raw: deque[np.int16] = deque(maxlen=CHUNK_SAMPLES + _MEL_CONTEXT)
        self._mel_buffer = np.ones((_WINDOW, _MEL_BINS), dtype=np.float32)
        self._features = np.zeros((0, _EMBEDDING), dtype=np.float32)
        self._background = np.zeros((0, _EMBEDDING), dtype=np.float32)
        self._chunks = 0

    @property
    def loaded(self) -> bool:
        """Whether the models are loaded."""
        return self._models is not None

    def load(self) -> None:
        """Load the three ONNX models (must already be downloaded). Blocking."""
        if self._models is not None:
            return
        melspec, embedding, wake_word = (self._directory / f.name for f in self._files)
        models = _Models(
            melspectrogram=self._session_factory(melspec),
            embedding=self._session_factory(embedding),
            wake_word=self._session_factory(wake_word),
        )
        # Like openWakeWord, start from embeddings of quiet noise rather than zeros, so the
        # model sees "background" until real audio has filled the buffer.
        noise = np.random.default_rng(0).integers(-1000, 1000, 16_000 * 4).astype(np.int16)
        self._background = _embeddings_of(models, noise)
        self._models = models
        self.reset()

    def reset(self) -> None:
        """Forget all audio, e.g. after a detection or when resuming listening."""
        self._pending = np.zeros(0, dtype=np.int16)
        self._raw.clear()
        self._mel_buffer = np.ones((_WINDOW, _MEL_BINS), dtype=np.float32)
        self._features = self._background.copy()
        self._chunks = 0

    def process(self, frame: AudioArray) -> float:
        """Add audio (float32, 16 kHz); return the highest score of any completed chunk."""
        models = self._models
        if models is None:
            msg = "the wake-word models are not loaded"
            raise RuntimeError(msg)
        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        self._pending = np.concatenate((self._pending, pcm))
        best = 0.0
        while self._pending.size >= CHUNK_SAMPLES:
            chunk, self._pending = self._pending[:CHUNK_SAMPLES], self._pending[CHUNK_SAMPLES:]
            best = max(best, self._score_chunk(models, chunk))
        return best

    def detected(self, frame: AudioArray) -> bool:
        """Process ``frame`` and return True if the wake word was just heard."""
        return self.process(frame) >= self.threshold

    def _score_chunk(self, models: _Models, chunk: Pcm) -> float:
        self._raw.extend(chunk.tolist())
        mel = _melspectrogram(models, np.asarray(self._raw, dtype=np.int16))
        self._mel_buffer = np.vstack((self._mel_buffer, mel))[-_MEL_MAX:]
        window = self._mel_buffer[-_WINDOW:][np.newaxis, :, :, np.newaxis]
        embedding = _run(models.embedding, "input_1", window).reshape(1, _EMBEDDING)
        self._features = np.vstack((self._features, embedding))[-_FEATURES_MAX:]
        self._chunks += 1
        if self._chunks <= _WARMUP_CHUNKS:
            return 0.0
        features = self._features[-_FEATURES:][np.newaxis].astype(np.float32)
        return float(_run(models.wake_word, "x.1", features).reshape(-1)[0])


@dataclass(frozen=True, slots=True)
class _Models:
    melspectrogram: InferenceSession
    embedding: InferenceSession
    wake_word: InferenceSession


def _run(session: InferenceSession, name: str, value: Floats) -> Floats:
    return np.asarray(session.run(None, {name: value})[0], dtype=np.float32)


def _melspectrogram(models: _Models, pcm: Pcm) -> Floats:
    audio = pcm.astype(np.float32)[np.newaxis, :]
    spec = np.squeeze(_run(models.melspectrogram, "input", audio))
    return (spec / 10 + 2).astype(np.float32).reshape(-1, _MEL_BINS)


def _embeddings_of(models: _Models, pcm: Pcm) -> Floats:
    spec = _melspectrogram(models, pcm)
    windows = [spec[i : i + _WINDOW] for i in range(0, spec.shape[0] - _WINDOW + 1, 8)]
    batch = np.asarray(windows, dtype=np.float32)[:, :, :, np.newaxis]
    return _run(models.embedding, "input_1", batch).reshape(-1, _EMBEDDING)
