from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fakes import VALID_KEY, FakeKeyring, FakeLLM, Invoke, TickingClock
from test_runtime_and_run import FakeBackends, LoadableSTT, LoadableTTS, install_models
from yarvis.bench import (
    Measurement,
    SttCandidate,
    recommendation,
    resample,
    run_bench,
    timed,
)
from yarvis.security.secrets import KEYRING_SERVICE

if TYPE_CHECKING:
    from yarvis.cli.services import Services
    from yarvis.paths import AppPaths


class DeviceSTT(LoadableSTT):
    def __init__(self, device: str, *, fallback: bool = False) -> None:
        super().__init__("what's the weather like in zedelgem tomorrow")
        self.requested = device
        self.device = "cpu" if fallback else device


def test_resample() -> None:
    audio = np.linspace(-1, 1, 24_000, dtype=np.float32)
    out = resample(audio, 24_000, 16_000)
    assert out.dtype == np.float32
    assert out.size == 16_000
    assert resample(audio, 16_000, 16_000).size == 24_000


async def test_timed_warms_up_then_measures() -> None:
    calls: list[int] = []

    async def action() -> None:
        calls.append(1)

    runs = await timed(action, 3, TickingClock(step=0.5))
    assert len(calls) == 4
    assert runs == pytest.approx((500.0, 500.0, 500.0))


async def test_run_bench_measures_everything() -> None:
    llm = FakeLLM(["ok"], ["ok"], ["ok"], ["ok"])
    progress: list[str] = []
    results = await run_bench(
        tts=LoadableTTS(),
        recognizers=[
            SttCandidate("whisper base.en on cpu", "cpu", lambda: DeviceSTT("cpu")),
            SttCandidate("whisper base.en on mps", "mps", lambda: DeviceSTT("mps", fallback=True)),
        ],
        llm=llm,
        rounds=2,
        on_progress=progress.append,
        clock=TickingClock(step=0.1),
    )
    assert [r.component for r in results] == [
        "Voice (Kokoro)",
        "Speech-to-text",
        "Speech-to-text",
        "Claude, first token",
    ]
    assert all(len(r.runs_ms) == 2 for r in results)
    assert "heard" in results[1].note
    assert results[1].setting == "whisper base.en on cpu"
    assert "mps unavailable" in results[2].note
    assert llm.warm_ups == 1
    assert progress[0] == "Loading the voice…"


async def test_run_bench_without_claude() -> None:
    results = await run_bench(
        tts=LoadableTTS(),
        recognizers=[SttCandidate("parakeet on mps", "mps", lambda: DeviceSTT("mps"))],
        llm=None,
        rounds=1,
    )
    assert [r.component for r in results] == ["Voice (Kokoro)", "Speech-to-text"]


def stt(device: str, ms: float, note: str = "") -> Measurement:
    return Measurement("Speech-to-text", f"whisper base.en on {device}", (ms,), note)


def test_recommendation() -> None:
    gpu_wins = recommendation([stt("cpu", 400), stt("mps", 150)])
    assert gpu_wins is not None
    assert 'stt_device = "mps"' in gpu_wins
    assert "2.7x" in gpu_wins
    cpu_fine = recommendation([stt("cpu", 400), stt("mps", 380)])
    assert cpu_fine is not None
    assert 'keep stt_device = "cpu"' in cpu_fine
    assert recommendation([stt("cpu", 400)]) is None
    assert recommendation([stt("cpu", 400), stt("mps", 100, "mps unavailable, ran on cpu")]) is None
    parakeet = Measurement("Speech-to-text", "parakeet on mps", (90,))
    assert recommendation([stt("cpu", 400), parakeet]) is None  # only compares Whisper


def test_bench_command(
    invoke: Invoke, services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths)
    fakes = FakeBackends(FakeLLM(*[["ok"]] * 6))
    services.backends = fakes.build
    result = invoke("bench", "--rounds", "1")
    assert result.exit_code == 0, result.output
    assert "Yarvis bench" in result.output
    assert "Speech-to-text" in result.output
    assert "Claude, first token" in result.output


def test_bench_command_without_claude(
    invoke: Invoke, services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths)
    fakes = FakeBackends(FakeLLM())
    services.backends = fakes.build
    result = invoke("bench", "--rounds", "1", "--no-claude")
    assert result.exit_code == 0, result.output
    assert "Claude, first token" not in result.output
    assert fakes.llm.calls == []


def test_bench_command_measures_parakeet_on_apple_silicon(
    invoke: Invoke, services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths, parakeet=True)
    services.parakeet_ok = lambda: True
    fakes = FakeBackends(FakeLLM(*[["ok"]] * 6))
    services.backends = fakes.build
    result = invoke("bench", "--rounds", "1", "--no-claude")
    assert result.exit_code == 0, result.output
    assert "parakeet on mps" in result.output
    assert fakes.stt_engines[0] == "parakeet"
    assert "faster-whisper" in fakes.stt_engines  # compared with Whisper


def test_bench_command_measures_parakeet_on_a_pcs_processor(
    invoke: Invoke, services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths, parakeet_onnx=True)
    services.parakeet_ok = lambda: True
    services.parakeet_build = lambda: "onnx"
    services.backends = FakeBackends(FakeLLM(*[["ok"]] * 6)).build
    result = invoke("bench", "--rounds", "1", "--no-claude")
    assert result.exit_code == 0, result.output
    assert "parakeet on cpu" in result.output
