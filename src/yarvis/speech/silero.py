"""Silero VAD (v6) running on ONNX Runtime, one 32 ms frame at a time.

The model file ships inside the ``faster-whisper`` wheel, so it is covered by the hashes
in ``uv.lock`` and needs no separate download.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import onnxruntime

from yarvis.speech.base import FRAME_SAMPLES

if TYPE_CHECKING:
    from pathlib import Path

    from yarvis.speech.base import AudioArray

CONTEXT_SAMPLES: Final = 64
STATE_SHAPE: Final = (1, 1, 128)


def bundled_model_path() -> Path:
    """Path of the Silero VAD model bundled with ``faster-whisper``."""
    resource = resources.files("faster_whisper") / "assets" / "silero_vad_v6.onnx"
    return cast("Path", resource)


class SileroVAD:
    """Streaming voice-activity detection. Not thread-safe; one instance per stream."""

    def __init__(self, model_path: Path | None = None) -> None:
        ort = cast("Any", onnxruntime)  # onnxruntime ships without type information
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session: Any = ort.InferenceSession(
            str(model_path or bundled_model_path()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._h = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def probability(self, frame: AudioArray) -> float:
        """Speech probability in [0, 1] for one 512-sample frame at 16 kHz."""
        if frame.shape != (FRAME_SAMPLES,):
            msg = f"expected a frame of {FRAME_SAMPLES} samples, got shape {frame.shape}"
            raise ValueError(msg)
        model_input = np.concatenate([self._context, frame]).astype(np.float32)[np.newaxis, :]
        outputs = cast(
            "list[np.ndarray[Any, np.dtype[np.float32]]]",
            self._session.run(None, {"input": model_input, "h": self._h, "c": self._c}),
        )
        probabilities, self._h, self._c = outputs
        self._context = frame[-CONTEXT_SAMPLES:].astype(np.float32)
        return float(probabilities.reshape(-1)[0])

    def reset(self) -> None:
        """Clear the recurrent state before a new, unrelated stream of audio."""
        self._h.fill(0.0)
        self._c.fill(0.0)
        self._context.fill(0.0)
