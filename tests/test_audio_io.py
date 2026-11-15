from __future__ import annotations

import asyncio
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from yarvis.audio import devices
from yarvis.audio.devices import AudioBackendError, load_sounddevice
from yarvis.audio.microphone import Microphone
from yarvis.audio.speaker import PlaybackBuffer, Speaker


class PortAudioError(Exception):
    pass


class FakeStream:
    device = 3

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice(ModuleType):
    PortAudioError = PortAudioError

    def __init__(self, *, fail: bool = False, names: dict[int, str] | None = None) -> None:
        super().__init__("sounddevice")
        self.fail = fail
        self.names = names or {}
        self.streams: list[FakeStream] = []

    def query_devices(self, device: int) -> dict[str, Any]:
        if device not in self.names:
            raise ValueError(f"no device {device}")
        return {"name": self.names[device]}

    def _open(self, **kwargs: Any) -> FakeStream:
        if self.fail:
            raise PortAudioError("Invalid device")
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream

    def InputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
        return self._open(**kwargs)

    def OutputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
        return self._open(**kwargs)


# --------------------------------------------------------------------------- microphone


async def test_microphone_delivers_frames_from_the_audio_thread() -> None:
    sd = FakeSoundDevice()
    async with Microphone(device="Test Mic", loader=lambda: sd) as mic:
        stream = sd.streams[0]
        assert stream.started
        assert stream.kwargs["samplerate"] == 16_000
        assert stream.kwargs["blocksize"] == 512
        assert stream.kwargs["device"] == "Test Mic"
        block = np.ones((512, 1), dtype=np.float32)
        await asyncio.to_thread(stream.callback, block, 512, None, None)
        frame = await asyncio.wait_for(anext(mic), timeout=1)
        assert frame.shape == (512,)
        block[:] = 0  # the microphone must have copied the data
        assert frame.sum() == 512
    assert stream.closed


async def test_microphone_drops_oldest_frames_when_full() -> None:
    sd = FakeSoundDevice()
    async with Microphone(loader=lambda: sd, max_queue=2) as mic:
        for value in range(4):
            sd.streams[0].callback(np.full((512, 1), value, np.float32), 512, None, None)
        await asyncio.sleep(0)
        assert mic.dropped == 2
        assert [int((await anext(mic))[0]) for _ in range(2)] == [2, 3]


async def test_microphone_ignores_callbacks_after_close() -> None:
    sd = FakeSoundDevice()
    mic = Microphone(loader=lambda: sd)
    mic._callback(np.zeros((512, 1), np.float32), 512, None, None)  # pyright: ignore[reportPrivateUsage]
    async with mic:
        pass


async def test_microphone_open_errors_are_reported() -> None:
    with pytest.raises(AudioBackendError, match="cannot open the microphone"):
        async with Microphone(loader=lambda: FakeSoundDevice(fail=True)):
            pass


def test_missing_portaudio_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> ModuleType:
        raise OSError("PortAudio library not found")

    monkeypatch.setattr(devices, "_import_sounddevice", broken)
    with pytest.raises(AudioBackendError, match="PortAudio"):
        load_sounddevice()


# --------------------------------------------------------------------------- playback buffer


def test_buffer_reads_across_chunks_and_pads_with_silence() -> None:
    buffer = PlaybackBuffer()
    buffer.append(np.array([1, 2, 3], np.float32))
    buffer.append(np.array([4, 5], np.float32))
    buffer.append(np.array([], np.float32))
    out = np.full(4, 9.0, np.float32)
    assert buffer.read_into(out) == 4
    assert out.tolist() == [1, 2, 3, 4]
    assert buffer.pending == 1
    out = np.full(3, 9.0, np.float32)
    assert buffer.read_into(out) == 1
    assert out.tolist() == [5, 0, 0]
    assert buffer.pending == 0


@given(
    chunks=st.lists(st.integers(1, 50), max_size=20),
    reads=st.lists(st.integers(1, 64), min_size=1, max_size=40),
)
def test_buffer_preserves_order_and_count(chunks: list[int], reads: list[int]) -> None:
    buffer = PlaybackBuffer()
    expected: list[float] = []
    counter = 0
    for size in chunks:
        values = np.arange(counter, counter + size, dtype=np.float32) + 1
        counter += size
        expected.extend(values.tolist())
        buffer.append(values)
    played: list[float] = []
    for size in reads:
        out = np.zeros(size, np.float32)
        written = buffer.read_into(out)
        played.extend(out[:written].tolist())
        assert not out[written:].any()
    assert played == expected[: len(played)]
    assert buffer.pending == len(expected) - len(played)


def test_buffer_clear() -> None:
    buffer = PlaybackBuffer()
    buffer.append(np.ones(10, np.float32))
    buffer.clear()
    assert buffer.pending == 0


# --------------------------------------------------------------------------- speaker


async def test_speaker_plays_queued_audio_and_drains() -> None:
    sd = FakeSoundDevice()
    async with Speaker(sample_rate=24_000, loader=lambda: sd) as speaker:
        stream = sd.streams[0]
        assert stream.kwargs["samplerate"] == 24_000
        await speaker.play(np.ones(1000, np.float32))
        drained = asyncio.create_task(speaker.drain())
        await asyncio.sleep(0)
        assert not drained.done()
        out = np.zeros((600, 1), np.float32)
        await asyncio.to_thread(stream.callback, out, 600, None, None)
        assert out.sum() == 600
        await asyncio.to_thread(stream.callback, out, 600, None, None)
        await asyncio.wait_for(drained, timeout=2)
    assert stream.closed


async def test_speaker_stop_discards_audio_and_releases_drain() -> None:
    sd = FakeSoundDevice()
    async with Speaker(sample_rate=24_000, loader=lambda: sd) as speaker:
        await speaker.play(np.ones(10_000, np.float32))
        drained = asyncio.create_task(speaker.drain())
        await asyncio.sleep(0)
        speaker.stop()
        await asyncio.wait_for(drained, timeout=2)


async def test_speaker_drain_with_nothing_queued_returns() -> None:
    sd = FakeSoundDevice()
    async with Speaker(sample_rate=24_000, loader=lambda: sd) as speaker:
        await asyncio.wait_for(speaker.drain(), timeout=1)


async def test_speaker_open_errors_are_reported() -> None:
    with pytest.raises(AudioBackendError, match="cannot open the speaker"):
        async with Speaker(sample_rate=24_000, loader=lambda: FakeSoundDevice(fail=True)):
            pass


async def test_microphone_flush_drops_queued_frames() -> None:
    sd = FakeSoundDevice()
    async with Microphone(loader=lambda: sd) as microphone:
        callback = sd.streams[0].callback
        for value in (0.1, 0.2):
            await asyncio.to_thread(callback, np.full((512, 1), value, np.float32), 512, None, None)
        await asyncio.sleep(0.01)
        microphone.flush()
        await asyncio.to_thread(callback, np.full((512, 1), 0.3, np.float32), 512, None, None)
        frame = await asyncio.wait_for(anext(microphone), 1)
        assert frame[0] == pytest.approx(0.3)


async def test_speaker_reports_the_device_it_plays_on() -> None:
    sd = FakeSoundDevice(names={3: "AirPods Pro"})
    async with Speaker(sample_rate=24_000, loader=lambda: sd) as speaker:
        assert speaker.device_name == "AirPods Pro"


async def test_speaker_device_name_is_optional() -> None:
    sd = FakeSoundDevice()  # PortAudio cannot name the device
    async with Speaker(sample_rate=24_000, loader=lambda: sd) as speaker:
        assert speaker.device_name is None
