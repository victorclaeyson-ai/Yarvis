"""Gap-free audio playback that can be stopped instantly.

Audio is appended to a thread-safe buffer; the PortAudio callback pulls exactly as many
samples as the device asks for. This keeps sentences flowing without gaps and makes
``stop()`` immediate, which barge-in relies on.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from yarvis.audio.devices import AudioBackendError, load_sounddevice

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType, TracebackType

    from yarvis.speech.base import AudioArray

_DRAIN_POLL_SECONDS = 0.25


class PlaybackBuffer:
    """Thread-safe FIFO of audio chunks, read in arbitrary-sized blocks."""

    def __init__(self) -> None:
        self._chunks: deque[AudioArray] = deque()
        self._offset = 0
        self._pending = 0
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        """Samples waiting to be played."""
        with self._lock:
            return self._pending

    def append(self, audio: AudioArray) -> None:
        """Queue ``audio`` (copied, so the caller may reuse its array)."""
        if audio.size == 0:
            return
        with self._lock:
            self._chunks.append(np.array(audio, dtype=np.float32).reshape(-1))
            self._pending += audio.size

    def read_into(self, out: AudioArray) -> int:
        """Fill ``out`` from the queue, zero-padding the rest. Returns samples written."""
        written = 0
        with self._lock:
            while written < out.size and self._chunks:
                chunk = self._chunks[0]
                take = min(out.size - written, chunk.size - self._offset)
                out[written : written + take] = chunk[self._offset : self._offset + take]
                written += take
                self._offset += take
                if self._offset == chunk.size:
                    self._chunks.popleft()
                    self._offset = 0
            self._pending -= written
        out[written:] = 0.0
        return written

    def clear(self) -> None:
        """Drop everything queued."""
        with self._lock:
            self._chunks.clear()
            self._offset = 0
            self._pending = 0


class Speaker:
    """Async context manager that plays mono float32 audio at a fixed sample rate."""

    def __init__(
        self,
        *,
        sample_rate: int,
        device: str | int | None = None,
        loader: Callable[[], ModuleType] = load_sounddevice,
    ) -> None:
        self.sample_rate = sample_rate
        self._device = device
        self._loader = loader
        self._buffer = PlaybackBuffer()
        self._idle = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None
        self.device_name: str | None = None
        """Name of the device actually playing (known once open)."""

    async def __aenter__(self) -> Self:
        sd = self._loader()
        self._loop = asyncio.get_running_loop()
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                latency="low",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            msg = f"cannot open the speaker: {exc}"
            raise AudioBackendError(msg) from exc
        self.device_name = _device_name(sd, self._stream)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    async def play(self, audio: AudioArray) -> None:
        """Queue audio; returns immediately."""
        self._buffer.append(audio)

    async def drain(self) -> None:
        """Wait until all queued audio has been played (or ``stop`` was called)."""
        while self._buffer.pending:
            self._idle.clear()
            if not self._buffer.pending:
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._idle.wait(), timeout=_DRAIN_POLL_SECONDS)

    def stop(self) -> None:
        """Stop playback immediately and discard queued audio."""
        self._buffer.clear()
        self._idle.set()

    def _callback(self, outdata: AudioArray, frames: int, time: object, status: object) -> None:
        """PortAudio thread: fill the device buffer."""
        del frames, time, status
        had_audio = self._buffer.pending > 0
        self._buffer.read_into(outdata[:, 0])
        if had_audio and self._buffer.pending == 0:
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(self._idle.set)


def _device_name(sd: ModuleType, stream: Any) -> str | None:  # noqa: ANN401 - untyped library
    """Name of the device ``stream`` plays on; None if PortAudio cannot tell."""
    try:
        info: Any = sd.query_devices(stream.device)
        return str(info["name"])
    except (sd.PortAudioError, ValueError, TypeError, KeyError, AttributeError):
        return None
