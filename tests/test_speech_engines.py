"""Speech backends: Silero runs for real (it ships in a wheel); Whisper and Kokoro use fakes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from yarvis.speech.base import FRAME_SAMPLES
from yarvis.speech.endpointing import Endpointer, EndpointerSettings, EndpointEvent
from yarvis.speech.kokoro import SAMPLE_RATE, KokoroTTS, UnknownVoiceError
from yarvis.speech.silero import SileroVAD, bundled_model_path
from yarvis.speech.whisper import WhisperSTT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from yarvis.speech.base import AudioArray

DATA = Path(__file__).parent / "data"


def speech_clip() -> AudioArray:
    """~4.4 s at 16 kHz: silence, "Hey Yarvis, what time is it in Tokyo right now?", silence."""
    pcm = np.load(DATA / "speech_16k.npy")
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def frames(audio: AudioArray) -> list[AudioArray]:
    usable = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
    return list(audio[:usable].reshape(-1, FRAME_SAMPLES))


# --------------------------------------------------------------------------- Silero


def test_silero_model_ships_with_faster_whisper() -> None:
    assert Path(str(bundled_model_path())).is_file()


def test_silero_separates_speech_from_silence() -> None:
    vad = SileroVAD()
    probabilities = [vad.probability(f) for f in frames(speech_clip())]
    leading_silence = probabilities[:10]
    assert max(leading_silence) < 0.2
    assert max(probabilities) > 0.9
    assert sum(p > 0.5 for p in probabilities) > 50  # roughly 3 s of speech


def test_silero_reset_makes_runs_reproducible() -> None:
    vad = SileroVAD()
    clip = frames(speech_clip())[:40]
    first = [vad.probability(f) for f in clip]
    vad.reset()
    second = [vad.probability(f) for f in clip]
    assert first == pytest.approx(second)


def test_silero_rejects_wrong_frame_size() -> None:
    with pytest.raises(ValueError, match="512"):
        SileroVAD().probability(np.zeros(100, np.float32))


def test_default_settings_capture_the_whole_sentence() -> None:
    vad, endpointer = SileroVAD(), Endpointer(EndpointerSettings())
    events = [endpointer.push(f, vad.probability(f)) for f in frames(speech_clip())]
    assert EndpointEvent.SPEECH_STARTED in events
    assert EndpointEvent.END_OF_TURN in events
    assert 2.5 < len(endpointer.utterance()) / 16_000 < 3.6


# --------------------------------------------------------------------------- Whisper


@dataclass
class Segment:
    text: str
    no_speech_prob: float


class FakeWhisper:
    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self.options: dict[str, Any] = {}

    def transcribe(self, audio: AudioArray, **options: Any) -> tuple[Iterable[Segment], None]:
        self.options = options
        return iter(self.segments), None


async def test_whisper_joins_confident_segments(tmp_path: Path) -> None:
    model = FakeWhisper(
        [Segment(" Hello", 0.01), Segment(" Thank you.", 0.9), Segment(" world ", 0.1)]
    )
    calls: list[bool] = []

    def loader(name: str, root: Path, *, local_only: bool, device: str = "cpu") -> FakeWhisper:
        calls.append(local_only)
        assert name == "base.en"
        assert root.is_relative_to(tmp_path)
        return model

    stt = WhisperSTT(
        "base.en", cache_dir=tmp_path, vocabulary=["Yarvis", "Zedelgem"], loader=loader
    )
    assert await stt.transcribe(np.zeros(16_000, np.float32)) == "Hello world"
    assert model.options["initial_prompt"] == "Yarvis, Zedelgem."
    assert await stt.transcribe(np.zeros(16_000, np.float32)) == "Hello world"
    assert calls == [True]  # loaded once, from the local cache
    assert model.options["language"] == "en"
    assert model.options["beam_size"] == 1


def test_whisper_downloads_when_not_cached(tmp_path: Path) -> None:
    calls: list[bool] = []

    def loader(name: str, root: Path, *, local_only: bool, device: str = "cpu") -> FakeWhisper:
        calls.append(local_only)
        if local_only:
            raise FileNotFoundError("not cached")
        return FakeWhisper([])

    stt = WhisperSTT("tiny.en", cache_dir=tmp_path, loader=loader)
    assert stt._prompt is None  # pyright: ignore[reportPrivateUsage]
    stt.load()
    stt.load()
    assert calls == [True, False]


def test_whisper_requires_loading_first(tmp_path: Path) -> None:
    stt = WhisperSTT("tiny.en", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not loaded"):
        stt._transcribe_blocking(np.zeros(1, np.float32))  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- Kokoro


class FakeKokoro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, str]] = []

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        self.calls.append((text, voice, speed, lang))
        return [0.0, 0.5, -0.5], SAMPLE_RATE

    def get_voices(self) -> list[str]:
        return ["af_heart", "bm_george", "jf_alpha"]


async def test_kokoro_synthesizes_with_the_right_language(tmp_path: Path) -> None:
    engine = FakeKokoro()
    loaded: list[tuple[str, str]] = []

    def loader(model: Path, voices: Path) -> FakeKokoro:
        loaded.append((model.name, voices.name))
        return engine

    tts = KokoroTTS(cache_dir=tmp_path, voice="bm_george", speed=1.1, loader=loader)
    audio = await tts.synthesize("Good evening.")
    assert audio.dtype == np.float32
    assert audio.tolist() == [0.0, 0.5, -0.5]
    assert engine.calls == [("Good evening.", "bm_george", 1.1, "en-gb")]
    assert loaded == [("kokoro-v1.0.onnx", "voices-v1.0.bin")]
    assert tts.sample_rate == 24_000


def test_kokoro_rejects_non_english_voice_prefix(tmp_path: Path) -> None:
    with pytest.raises(UnknownVoiceError, match="English"):
        KokoroTTS(cache_dir=tmp_path, voice="jf_alpha")


def test_kokoro_lists_voices_when_unknown(tmp_path: Path) -> None:
    tts = KokoroTTS(cache_dir=tmp_path, voice="am_nobody", loader=lambda m, v: FakeKokoro())
    with pytest.raises(UnknownVoiceError, match="af_heart, bm_george"):
        tts.load()


def test_kokoro_requires_loading_first(tmp_path: Path) -> None:
    tts = KokoroTTS(cache_dir=tmp_path, voice="af_heart")
    with pytest.raises(RuntimeError, match="not loaded"):
        tts._synthesize_blocking("hi")  # pyright: ignore[reportPrivateUsage]


def test_whisper_falls_back_to_cpu_when_the_gpu_fails(tmp_path: Path) -> None:
    attempts: list[tuple[str, bool]] = []

    def loader(name: str, root: Path, *, local_only: bool, device: str = "cpu") -> FakeWhisper:
        attempts.append((device, local_only))
        if device == "mps":
            raise ValueError("unsupported device mps")
        return FakeWhisper([])

    stt = WhisperSTT("base.en", cache_dir=tmp_path, device="mps", loader=loader)
    stt.load()
    assert stt.device == "cpu"
    assert attempts == [("mps", True), ("mps", False), ("cpu", True)]


def test_whisper_uses_the_gpu_when_it_works(tmp_path: Path) -> None:
    devices: list[str] = []

    def loader(name: str, root: Path, *, local_only: bool, device: str = "cpu") -> FakeWhisper:
        devices.append(device)
        return FakeWhisper([])

    stt = WhisperSTT("base.en", cache_dir=tmp_path, device="mps", loader=loader)
    stt.load()
    assert stt.device == "mps"
    assert devices == ["mps"]
