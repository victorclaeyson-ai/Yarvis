from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from test_speech_engines import frames, speech_clip
from yarvis.models import WAKE_WORD_FILES, is_present, wake_word_dir
from yarvis.speech.wakeword import CHUNK_SAMPLES, WakeWordDetector

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class FakeSession:
    """Stands in for one ONNX model; ``fn`` maps the input array to the output array."""

    def __init__(self, fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]) -> None:
        self.fn = fn
        self.calls = 0

    def run(self, output_names: Sequence[str] | None, inputs: Mapping[str, Any]) -> list[Any]:
        assert output_names is None
        self.calls += 1
        (value,) = inputs.values()
        return [self.fn(np.asarray(value))]


def loudness_models() -> dict[str, FakeSession]:
    """Fake models whose score is the peak level of the newest audio chunk.

    melspectrogram: every frame holds the peak of the newest 1280 samples (pre-transform);
    embedding: the newest mel frame; wake word: the newest embedding.
    """

    def melspectrogram(audio: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        count = max(1, (audio.shape[1] - 480) // 160)
        peak = float(np.abs(audio[0, -CHUNK_SAMPLES:]).max()) / 32767
        return np.full((1, 1, count, 32), (peak - 2) * 10, dtype=np.float32)  # undo x/10+2

    def embedding(windows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert windows.shape[1:] == (76, 32, 1)
        return np.repeat(windows[:, -1, 0, 0].reshape(-1, 1, 1, 1), 96, axis=3)

    def wake_word(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert features.shape == (1, 16, 96)
        return np.array([[features[0, -1, 0]]], dtype=np.float32)

    return {
        "melspectrogram.onnx": FakeSession(melspectrogram),
        "embedding_model.onnx": FakeSession(embedding),
        "hey_jarvis_v0.1.onnx": FakeSession(wake_word),
    }


def make_detector(
    tmp_path: Path, threshold: float = 0.5
) -> tuple[WakeWordDetector, dict[str, FakeSession], list[Path]]:
    sessions = loudness_models()
    opened: list[Path] = []

    def factory(path: Path) -> FakeSession:
        opened.append(path)
        return sessions[path.name]

    detector = WakeWordDetector(cache_dir=tmp_path, threshold=threshold, session_factory=factory)
    return detector, sessions, opened


def tone(level: float, samples: int = CHUNK_SAMPLES) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.full(samples, level, dtype=np.float32)


def test_threshold_must_be_a_probability(tmp_path: Path) -> None:
    for threshold in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError, match="threshold"):
            WakeWordDetector(cache_dir=tmp_path, threshold=threshold)


def test_processing_requires_loaded_models(tmp_path: Path) -> None:
    detector, _, _ = make_detector(tmp_path)
    assert not detector.loaded
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.process(tone(0.0))


def test_load_opens_the_pinned_files_once(tmp_path: Path) -> None:
    detector, _, opened = make_detector(tmp_path)
    detector.load()
    detector.load()
    assert detector.loaded
    assert opened == [wake_word_dir(tmp_path) / model.name for model in WAKE_WORD_FILES]


def test_scores_arrive_once_per_80_ms_chunk(tmp_path: Path) -> None:
    detector, sessions, _ = make_detector(tmp_path)
    detector.load()
    model = sessions["hey_jarvis_v0.1.onnx"]
    embedding = sessions["embedding_model.onnx"]
    before = embedding.calls
    assert detector.process(tone(0.0, 512)) == 0.0
    assert detector.process(tone(0.0, 512)) == 0.0
    assert embedding.calls == before  # 1024 samples: not a full chunk yet
    detector.process(tone(0.0, 512))
    assert embedding.calls == before + 1  # 1536 samples: one chunk, 256 carried over
    assert model.calls == 0  # still warming up


def test_the_first_chunks_are_ignored_while_the_buffers_fill(tmp_path: Path) -> None:
    detector, _, _ = make_detector(tmp_path)
    detector.load()
    loud = tone(0.9)
    assert [detector.process(loud) for _ in range(5)] == [0.0] * 5
    assert detector.process(loud) == pytest.approx(0.9, abs=0.01)


def test_detection_threshold_and_reset(tmp_path: Path) -> None:
    detector, _, _ = make_detector(tmp_path, threshold=0.6)
    detector.load()
    for _ in range(5):
        detector.process(tone(0.0))
    assert not detector.detected(tone(0.55))
    assert detector.detected(tone(0.7))
    detector.reset()
    assert not detector.detected(tone(0.7))  # warming up again after a reset


def test_large_frames_report_the_best_chunk(tmp_path: Path) -> None:
    detector, _, _ = make_detector(tmp_path)
    detector.load()
    for _ in range(5):
        detector.process(tone(0.0))
    audio = np.concatenate([tone(0.2), tone(0.8), tone(0.1)])
    assert detector.process(audio) == pytest.approx(0.8, abs=0.01)


def test_loud_input_is_clipped_not_wrapped(tmp_path: Path) -> None:
    detector, _, _ = make_detector(tmp_path)
    detector.load()
    for _ in range(5):
        detector.process(tone(0.0))
    assert detector.process(tone(4.0)) == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- real models

REAL_MODELS = os.environ.get("YARVIS_WAKE_WORD_MODELS")


@pytest.mark.skipif(not REAL_MODELS, reason="set YARVIS_WAKE_WORD_MODELS to the model folder")
def test_real_models_hear_hey_yarvis(tmp_path: Path) -> None:
    source = Path(REAL_MODELS or "")
    target = wake_word_dir(tmp_path)
    target.mkdir(parents=True)
    for model in WAKE_WORD_FILES:
        assert is_present(model, source), model.name
        (target / model.name).symlink_to(source / model.name)
    detector = WakeWordDetector(cache_dir=tmp_path)
    detector.load()

    silence = [np.zeros(512, np.float32)] * 60
    assert not any(detector.detected(frame) for frame in silence)
    # The test clip says "Hey Yarvis, ..." with a voice the model was never trained on.
    assert any(detector.detected(frame) for frame in frames(speech_clip()))
