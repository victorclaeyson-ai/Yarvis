"""Choosing the speech recognizer (Parakeet on Apple Silicon, Whisper elsewhere)."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fakes import FakeSTT
from yarvis.config.models import VoiceConfig
from yarvis.models import PARAKEET_FILES, parakeet_dir
from yarvis.speech.choice import (
    FallbackSTT,
    choose_engine,
    describe,
    make_stt,
    parakeet_available,
    parakeet_supported,
)
from yarvis.speech.parakeet import ParakeetModelMissingError, ParakeetSTT
from yarvis.speech.whisper import WhisperSTT

if TYPE_CHECKING:
    from pathlib import Path

    from yarvis.speech.base import AudioArray


@pytest.mark.parametrize(
    ("system", "machine", "installed", "supported"),
    [
        ("Darwin", "arm64", True, True),
        ("Darwin", "arm64", False, False),  # the package is missing
        ("Darwin", "x86_64", True, False),  # an Intel Mac has no MLX
        ("Linux", "aarch64", True, False),
    ],
)
def test_parakeet_needs_apple_silicon(
    system: str, machine: str, installed: bool, supported: bool
) -> None:
    assert parakeet_supported(system, machine, installed=installed) is supported


def test_parakeet_available_checks_this_machine() -> None:
    assert isinstance(parakeet_available(), bool)


def voice(**settings: object) -> VoiceConfig:
    return VoiceConfig.model_validate(settings)


def test_choose_engine(caplog: pytest.LogCaptureFixture) -> None:
    assert choose_engine(voice(), parakeet_ok=True) == "parakeet"
    assert choose_engine(voice(), parakeet_ok=False) == "faster-whisper"
    assert choose_engine(voice(stt="faster-whisper"), parakeet_ok=True) == "faster-whisper"
    assert choose_engine(voice(stt="parakeet"), parakeet_ok=True) == "parakeet"
    with caplog.at_level(logging.WARNING):
        assert choose_engine(voice(stt="parakeet"), parakeet_ok=False) == "faster-whisper"
    assert "Apple Silicon" in caplog.text


def test_describe() -> None:
    assert describe(voice(stt="parakeet")) == "parakeet"
    assert describe(voice(stt="faster-whisper", stt_model="base.en")) == "whisper base.en"


def test_make_stt(tmp_path: Path) -> None:
    whisper = make_stt(voice(stt_model="tiny.en"), tmp_path, parakeet_ok=False)
    assert isinstance(whisper, WhisperSTT)
    assert whisper.model_name == "tiny.en"
    parakeet = make_stt(voice(), tmp_path, parakeet_ok=True)
    assert isinstance(parakeet, FallbackSTT)
    assert isinstance(parakeet.active, ParakeetSTT)


class Engine(FakeSTT):
    def __init__(
        self, text: str, *, device: str, fails: bool = False, breaks: bool = False
    ) -> None:
        super().__init__(text)
        self.device = device
        self.model_name = f"model-{text}"
        self.fails = fails
        self.breaks = breaks
        self.loads = 0

    def load(self) -> None:
        self.loads += 1
        if self.fails:
            msg = "no GPU today"
            raise RuntimeError(msg)

    async def transcribe(self, audio: AudioArray) -> str:
        if self.breaks:
            msg = "[matmul] shapes do not match"
            raise ValueError(msg)
        return await super().transcribe(audio)


async def test_fallback_uses_the_primary_when_it_loads() -> None:
    primary = Engine("primary", device="mps")
    stt = FallbackSTT(primary, lambda: Engine("fallback", device="cpu"))
    assert await stt.transcribe(np.zeros(160, np.float32)) == "primary"  # loads on demand
    stt.load()  # already loaded: nothing happens
    assert primary.loads == 1
    assert (stt.device, stt.model_name) == ("mps", "model-primary")


async def test_fallback_takes_over_when_the_primary_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fallback = Engine("fallback", device="cpu")
    stt = FallbackSTT(Engine("primary", device="mps", fails=True), lambda: fallback)
    with caplog.at_level(logging.WARNING):
        stt.load()
    assert "fell back to Whisper: no GPU today" in caplog.text
    assert await stt.transcribe(np.zeros(160, np.float32)) == "fallback"
    assert stt.active is fallback
    assert (stt.device, stt.model_name, fallback.loads) == ("cpu", "model-fallback", 1)


class Plain(FakeSTT):
    def load(self) -> None:
        pass


async def test_fallback_takes_over_when_the_primary_breaks_mid_use(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model that loads but cannot transcribe must not leave Yarvis deaf (or stopped)."""
    primary = Engine("primary", device="mps", breaks=True)
    made: list[Engine] = []

    def fallback() -> Engine:
        made.append(Engine("whisper", device="cpu"))
        return made[-1]

    stt = FallbackSTT(primary, fallback)
    audio = np.zeros(16_000, np.float32)
    with caplog.at_level(logging.WARNING):
        heard = await asyncio.gather(stt.transcribe(audio), stt.transcribe(audio))
    assert heard == ["whisper", "whisper"]  # the failed requests are heard again
    assert "fell back to Whisper: [matmul] shapes do not match" in caplog.text
    assert len(made) == 1 and made[0].loads == 1  # switched once
    assert await stt.transcribe(audio) == "whisper"
    assert stt.device == "cpu"
    made[0].breaks = True  # Whisper failing too is a real error: reported, not hidden
    with pytest.raises(ValueError, match="matmul"):
        await stt.transcribe(audio)


def test_fallback_describes_engines_without_details() -> None:
    stt = FallbackSTT(Plain(), Plain)  # no device or model name to report
    assert (stt.device, stt.model_name) == ("cpu", "unknown")


# --------------------------------------------------------------------------- Parakeet


def install_parakeet(cache: Path) -> None:
    directory = parakeet_dir(cache)
    directory.mkdir(parents=True)
    for model in PARAKEET_FILES:
        with (directory / model.name).open("wb") as handle:
            handle.truncate(model.size)  # sparse: right size, no disk space


class FakeParakeet:
    def __init__(self, threads: list[str]) -> None:
        self.threads = threads
        self.heard: list[int] = []

    def transcribe(self, audio: AudioArray) -> str:
        self.threads.append(threading.current_thread().name)
        self.heard.append(audio.size)
        return "play We Go On on Spotify"


async def test_parakeet_runs_on_one_worker_thread(tmp_path: Path) -> None:
    install_parakeet(tmp_path)
    threads: list[str] = []
    loaded: list[Path] = []
    engine = FakeParakeet(threads)

    def loader(directory: Path) -> FakeParakeet:
        threads.append(threading.current_thread().name)
        loaded.append(directory)
        return engine

    stt = ParakeetSTT(cache_dir=tmp_path, loader=loader)
    assert (stt.device, stt.model_name) == ("mps", "parakeet-tdt-0.6b-v2")
    stt.load()
    audio = np.zeros(16_000, np.float32)
    assert await stt.transcribe(audio) == "play We Go On on Spotify"
    assert await stt.transcribe(audio) == "play We Go On on Spotify"
    assert loaded == [parakeet_dir(tmp_path)]  # loaded once
    assert engine.heard == [16_000, 16_000]
    # MLX arrays belong to the thread that made them: everything runs on the same one.
    assert len(set(threads)) == 1
    assert threads[0].startswith("yarvis-parakeet")


async def test_parakeet_loads_on_first_use(tmp_path: Path) -> None:
    install_parakeet(tmp_path)
    stt = ParakeetSTT(cache_dir=tmp_path, loader=lambda _: FakeParakeet([]))
    assert await stt.transcribe(np.zeros(16_000, np.float32)) == "play We Go On on Spotify"


async def test_parakeet_refuses_missing_or_partial_files(tmp_path: Path) -> None:
    stt = ParakeetSTT(cache_dir=tmp_path, loader=lambda _: FakeParakeet([]))
    with pytest.raises(ParakeetModelMissingError, match="yarvis models download"):
        stt.load()
    install_parakeet(tmp_path)
    weights = parakeet_dir(tmp_path) / "model.safetensors"
    with weights.open("r+b") as handle:
        handle.truncate(1000)  # an interrupted copy
    with pytest.raises(ParakeetModelMissingError):
        await stt.transcribe(np.zeros(16_000, np.float32))


async def test_a_parakeet_that_cannot_load_falls_back_to_whisper(tmp_path: Path) -> None:
    whisper = Engine("whisper heard you", device="cpu")
    stt = FallbackSTT(ParakeetSTT(cache_dir=tmp_path), lambda: whisper)  # not downloaded
    assert await stt.transcribe(np.zeros(16_000, np.float32)) == "whisper heard you"
