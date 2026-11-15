"""``yarvis bench``: measure every stage on this machine and recommend settings.

The test sentence is synthesized with the real voice and fed to speech recognition, so the
benchmark needs no recordings and also checks that the two models understand each other.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from yarvis.llm.base import Message
from yarvis.speech.base import MIC_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from yarvis.llm.base import LLMProvider
    from yarvis.runtime import LoadableSTT, LoadableTTS
    from yarvis.speech.base import AudioArray

TTS_PHRASE: Final = "Good afternoon, Victor."
STT_PHRASE: Final = "What's the weather like in Zedelgem tomorrow?"
GPU_ADVANTAGE: Final = 0.8  # recommend the GPU only if it is at least 20 % faster


@dataclass(frozen=True, slots=True)
class Measurement:
    """Timings of one component with one setting."""

    component: str
    setting: str
    runs_ms: tuple[float, ...]
    note: str = ""

    @property
    def median_ms(self) -> float:
        """Median of the runs."""
        return statistics.median(self.runs_ms)


@dataclass(frozen=True, slots=True)
class SttCandidate:
    """A speech recognition setting to measure."""

    setting: str
    """Shown in the table, e.g. "whisper base.en on cpu"."""
    device: str
    """Where it should run; if it ends up elsewhere, the table says so."""
    make: Callable[[], LoadableSTT]


def resample(audio: AudioArray, source_rate: int, target_rate: int) -> AudioArray:
    """Linear-interpolation resampling (adequate for benchmarks, not for playback)."""
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)
    count = round(audio.size * target_rate / source_rate)
    positions = np.linspace(0, audio.size, count, endpoint=False)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


async def timed(
    action: Callable[[], Awaitable[object]],
    rounds: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[float, ...]:
    """Run ``action`` once to warm up, then ``rounds`` times; return durations in ms."""
    await action()
    durations: list[float] = []
    for _ in range(rounds):
        start = clock()
        await action()
        durations.append((clock() - start) * 1000)
    return tuple(durations)


async def first_token_ms(llm: LLMProvider, clock: Callable[[], float] = time.perf_counter) -> float:
    """Time until Claude's first token for a tiny prompt."""
    start = clock()
    messages = [Message("user", "Reply with exactly one word: ok")]
    async with aclosing(llm.stream(messages, system="Be brief.", max_tokens=5)) as stream:
        async for _token in stream:
            break
    return (clock() - start) * 1000


async def run_bench(
    *,
    tts: LoadableTTS,
    recognizers: Sequence[SttCandidate],
    llm: LLMProvider | None,
    rounds: int = 3,
    on_progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[Measurement]:
    """Measure the voice, each speech recognition setting, and Claude's first token."""

    def progress(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    results: list[Measurement] = []
    progress("Loading the voice…")
    await asyncio.to_thread(tts.load)
    progress("Measuring the voice…")
    runs = await timed(lambda: tts.synthesize(TTS_PHRASE), rounds, clock)
    results.append(Measurement("Voice (Kokoro)", f'"{TTS_PHRASE}"', runs))

    speech = await tts.synthesize(STT_PHRASE)
    sample = resample(speech, tts.sample_rate, MIC_SAMPLE_RATE)
    for candidate in recognizers:
        progress(f"Measuring speech recognition: {candidate.setting}…")
        stt = candidate.make()
        await asyncio.to_thread(stt.load)
        runs = await timed(lambda stt=stt: stt.transcribe(sample), rounds, clock)
        heard = await stt.transcribe(sample)
        actual = getattr(stt, "device", candidate.device)
        note = (
            f'heard: "{heard}"'
            if actual == candidate.device
            else f"{candidate.device} unavailable, ran on {actual}"
        )
        results.append(Measurement("Speech-to-text", candidate.setting, runs, note))

    if llm is not None:
        progress("Measuring Claude…")
        await llm.warm_up()
        await first_token_ms(llm, clock)  # warm the connection with a real request
        runs = tuple([await first_token_ms(llm, clock) for _ in range(rounds)])
        results.append(Measurement("Claude, first token", "warm connection", runs))
    return results


def recommendation(results: Sequence[Measurement]) -> str | None:
    """Suggest the GPU for Whisper if it is clearly faster there."""
    by_device = {
        m.setting.rsplit(" on ", 1)[-1]: m
        for m in results
        if m.component == "Speech-to-text"
        and m.setting.startswith("whisper ")
        and "unavailable" not in m.note
    }
    cpu, gpu = by_device.get("cpu"), by_device.get("mps")
    if cpu is None or gpu is None:
        return None
    if gpu.median_ms < cpu.median_ms * GPU_ADVANTAGE:
        return (
            f"Speech recognition is {cpu.median_ms / gpu.median_ms:.1f}x faster on the GPU. "
            'Set stt_device = "mps" under [voice] in your config.'
        )
    return 'The CPU is as fast as the GPU for speech recognition here; keep stt_device = "cpu".'
