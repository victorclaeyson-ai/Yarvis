from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Self

import numpy as np
import pytest

from fakes import FakeWakeWord
from test_speech_engines import frames, speech_clip
from yarvis.ears import (
    BARGE_IN_PREFIX_MS,
    FRAME_MS,
    WAKE_PREFIX_MS,
    MicrophoneEars,
    SharedMicrophone,
    barge_in_mode,
)
from yarvis.listen import MicrophoneSilentError, MicrophoneStalledError, PauseHooks
from yarvis.speech.endpointing import EndpointerSettings
from yarvis.speech.silero import SileroVAD

if TYPE_CHECKING:
    from types import TracebackType

    from yarvis.config.models import BargeIn
    from yarvis.ears import BargeInMode, Interruption
    from yarvis.speech.base import AudioArray

QUIET = np.full(512, 0.001, np.float32)
NO_HOOKS = PauseHooks(on_pause=lambda audio, control: None, on_resume=lambda: None)


class QueueMic:
    """A microphone fed by the test. Reading is cancellation-safe, like the real one."""

    def __init__(self, *initial: AudioArray) -> None:
        self.queue: asyncio.Queue[AudioArray | None] = asyncio.Queue()
        self.opened = 0
        self.closed = 0
        self.flushes = 0
        self.feed(*initial)

    def feed(self, *items: AudioArray | None) -> None:
        for item in items:
            self.queue.put_nowait(item)

    def end(self) -> None:
        self.queue.put_nowait(None)

    async def __aenter__(self) -> Self:
        self.opened += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.closed += 1

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> AudioArray:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def flush(self) -> None:
        self.flushes += 1
        while not self.queue.empty():
            self.queue.get_nowait()


class ScriptedVAD:
    """Speech probability = the frame's second sample (tests write it there)."""

    def __init__(self) -> None:
        self.resets = 0

    def probability(self, frame: AudioArray) -> float:
        return float(frame[1])

    def reset(self) -> None:
        self.resets += 1


def voiced(probability: float) -> AudioArray:
    frame = np.full(512, 0.01, np.float32)
    frame[1] = probability
    return frame


def trigger() -> AudioArray:
    frame = np.full(512, 0.01, np.float32)
    frame[0] = FakeWakeWord.TRIGGER
    return frame


def make_ears(
    mic: QueueMic,
    *,
    always_on: bool = True,
    wake_word: FakeWakeWord | None = None,
    barge_in: BargeInMode = "off",
    barge_in_vad: ScriptedVAD | None = None,
) -> tuple[MicrophoneEars, SharedMicrophone]:
    shared = SharedMicrophone(lambda: mic, always_on=always_on)
    ears = MicrophoneEars(
        shared,
        vad=SileroVAD(),
        settings=EndpointerSettings(no_speech_timeout_ms=1_500),
        wake_word=wake_word,
        barge_in=barge_in,
        barge_in_vad=barge_in_vad,
    )
    return ears, shared


def count(milliseconds: int) -> int:
    return -(-milliseconds // FRAME_MS)


# --------------------------------------------------------------------------- shared microphone


async def test_always_on_microphone_stays_open_between_holds() -> None:
    mic = QueueMic()
    async with SharedMicrophone(lambda: mic, always_on=True) as shared:
        assert shared.is_open
        async with shared.hold():
            pass
        assert shared.is_open and mic.closed == 0
    assert not shared.is_open and mic.closed == 1


async def test_on_demand_microphone_opens_for_the_outermost_hold_only() -> None:
    mic = QueueMic()
    async with SharedMicrophone(lambda: mic, always_on=False) as shared:
        assert not shared.is_open
        with pytest.raises(RuntimeError, match="not open"):
            shared.frames()
        shared.flush()  # harmless while closed
        async with shared.hold():
            async with shared.hold():
                assert shared.is_open
            assert shared.is_open  # still held by the outer block
        assert not shared.is_open
        async with shared.hold():
            pass
    assert (mic.opened, mic.closed) == (2, 2)


async def test_frames_survive_a_cancelled_reader() -> None:
    mic = QueueMic()
    async with SharedMicrophone(lambda: mic, always_on=True) as shared:
        reader = asyncio.ensure_future(anext(shared.frames()))
        await asyncio.sleep(0)
        reader.cancel()
        await asyncio.wait({reader})
        mic.feed(QUIET)
        assert await anext(shared.frames()) is QUIET


# --------------------------------------------------------------------------- listening


async def test_a_fresh_listen_drops_stale_audio() -> None:
    mic = QueueMic(*frames(speech_clip()))  # e.g. recorded while Yarvis was talking
    ears, shared = make_ears(mic)
    async with shared:
        task = asyncio.ensure_future(ears.listen(NO_HOOKS))
        await asyncio.sleep(0)
        mic.feed(*[QUIET] * 60)
        assert await task is None
    assert mic.flushes == 1


async def test_listening_continues_from_a_prefix_without_dropping_audio() -> None:
    clip = frames(speech_clip())
    mic = QueueMic(*clip[20:], *[QUIET] * 40)
    ears, shared = make_ears(mic)
    async with shared:
        utterance = await ears.listen(NO_HOOKS, prefix=clip[:20])
    assert utterance is not None
    assert len(utterance.audio) > 2.5 * 16_000
    assert mic.flushes == 0


async def test_listening_can_wait_less_for_speech() -> None:
    mic = QueueMic(*[QUIET] * 5)
    shared = SharedMicrophone(lambda: mic, always_on=False)
    ears = MicrophoneEars(
        shared, vad=SileroVAD(), settings=EndpointerSettings(no_speech_timeout_ms=60_000)
    )
    assert await ears.listen(NO_HOOKS, follow_up=True, prefix=(), timeout_s=0.1) is None
    assert not shared.is_open  # opened for the listen only


# --------------------------------------------------------------------------- wake word


async def test_wake_returns_the_audio_around_the_wake_word() -> None:
    mic = QueueMic()
    detector = FakeWakeWord()
    ears, shared = make_ears(mic, wake_word=detector)
    assert ears.wake_word and not ears.barge_in
    async with shared:
        waiting = asyncio.ensure_future(ears.wake())
        await asyncio.sleep(0)
        mic.feed(*[QUIET] * 100, trigger())
        prefix = await waiting
    assert len(prefix) == count(WAKE_PREFIX_MS)
    assert prefix[-1][0] == FakeWakeWord.TRIGGER
    assert detector.resets == 2  # before listening, and after the detection
    assert mic.flushes == 1


async def test_wake_needs_a_detector() -> None:
    ears, _ = make_ears(QueueMic())
    with pytest.raises(RuntimeError, match="wake word is off"):
        await ears.wake()


async def test_wake_reports_a_microphone_that_stops() -> None:
    mic = QueueMic()
    ears, shared = make_ears(mic, wake_word=FakeWakeWord())
    async with shared:
        waiting = asyncio.ensure_future(ears.wake())
        await asyncio.sleep(0)
        mic.feed(QUIET)
        mic.end()
        with pytest.raises(MicrophoneStalledError):
            await waiting


async def test_wake_reports_a_muted_microphone() -> None:
    mic = QueueMic()
    ears, shared = make_ears(mic, wake_word=FakeWakeWord())
    async with shared:
        waiting = asyncio.ensure_future(ears.wake())
        await asyncio.sleep(0)
        mic.feed(*[np.zeros(512, np.float32)] * 60)
        with pytest.raises(MicrophoneSilentError):
            await waiting


# --------------------------------------------------------------------------- barge-in


async def interrupt(ears: MicrophoneEars, mic: QueueMic, *audio: AudioArray | None) -> Interruption:
    waiting = asyncio.ensure_future(ears.interruption())
    await asyncio.sleep(0)
    mic.feed(*audio)
    return await waiting


async def test_talking_interrupts_with_headphones() -> None:
    mic = QueueMic(voiced(0.99))  # stale: flushed
    vad = ScriptedVAD()
    ears, shared = make_ears(mic, barge_in="voice", barge_in_vad=vad)
    async with shared:
        quiet = [voiced(0.1)] * 30
        talking = [voiced(0.95)] * 6 + [voiced(0.3)] + [voiced(0.95)] * 10  # a short dip
        result = await interrupt(ears, mic, *quiet, *talking)
    assert not result.wake_word
    assert len(result.prefix) == count(BARGE_IN_PREFIX_MS)
    assert vad.resets == 1


async def test_a_cough_does_not_interrupt() -> None:
    mic = QueueMic()
    ears, shared = make_ears(mic, barge_in="voice", barge_in_vad=ScriptedVAD())
    async with shared:
        cough = [voiced(0.95)] * 5 + [voiced(0.1)] * 5
        with pytest.raises(MicrophoneStalledError):
            await interrupt(ears, mic, *cough * 4, None)


async def test_saying_the_wake_word_interrupts_on_speakers() -> None:
    mic = QueueMic()
    ears, shared = make_ears(mic, wake_word=FakeWakeWord(), barge_in="wake_word")
    async with shared:
        result = await interrupt(ears, mic, *[voiced(0.99)] * 20, trigger())
    assert result.wake_word
    assert result.prefix[-1][0] == FakeWakeWord.TRIGGER


async def test_interruption_is_unavailable_when_barge_in_is_off() -> None:
    ears, shared = make_ears(QueueMic())
    async with shared:
        with pytest.raises(RuntimeError, match="barge-in is off"):
            await ears.interruption()


def test_barge_in_needs_its_detectors() -> None:
    shared = SharedMicrophone(QueueMic, always_on=False)
    settings = EndpointerSettings()
    with pytest.raises(ValueError, match="wake-word detector"):
        MicrophoneEars(shared, vad=SileroVAD(), settings=settings, barge_in="wake_word")
    with pytest.raises(ValueError, match="voice activity detector"):
        MicrophoneEars(shared, vad=SileroVAD(), settings=settings, barge_in="voice")


@pytest.mark.parametrize(
    ("setting", "device", "wake_word", "expected"),
    [
        ("auto", "AirPods Pro", True, "voice"),
        ("auto", "MacBook Pro Speakers", True, "wake_word"),
        ("auto", None, True, "wake_word"),
        ("auto", "MacBook Pro Speakers", False, "off"),
        ("voice", "MacBook Pro Speakers", False, "voice"),
        ("wake_word", "AirPods Pro", True, "wake_word"),
        ("wake_word", "AirPods Pro", False, "off"),
        ("off", "AirPods Pro", True, "off"),
    ],
)
def test_barge_in_mode(
    setting: BargeIn, device: str | None, wake_word: bool, expected: BargeInMode
) -> None:
    assert barge_in_mode(setting, output_device=device, wake_word=wake_word) == expected
