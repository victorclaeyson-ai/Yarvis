"""Microphone capture as an async stream of fixed-size frames.

The PortAudio callback runs on a native thread; it copies each block and hands it to the
event loop with ``call_soon_threadsafe``. If the consumer falls behind, the oldest frames
are dropped (fresh audio matters more than old audio) and counted.

The operating system's microphone indicator is always truthful: the stream is open exactly
while Yarvis can hear (only while listening, or the whole session when the wake word is on).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from yarvis.audio.devices import AudioBackendError, load_sounddevice
from yarvis.speech.base import FRAME_SAMPLES, MIC_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType, TracebackType

    from yarvis.speech.base import AudioArray

DEFAULT_QUEUE_FRAMES = 128  # ~4 s of audio


class Microphone:
    """Async context manager and iterator over 16 kHz mono frames of ``FRAME_SAMPLES``."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        max_queue: int = DEFAULT_QUEUE_FRAMES,
        loader: Callable[[], ModuleType] = load_sounddevice,
    ) -> None:
        self._device = device
        self._loader = loader
        self._queue: asyncio.Queue[AudioArray] = asyncio.Queue(maxsize=max_queue)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None
        self.dropped = 0

    async def __aenter__(self) -> Self:
        sd = self._loader()
        self._loop = asyncio.get_running_loop()
        try:
            self._stream = sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            msg = f"cannot open the microphone: {exc}"
            raise AudioBackendError(msg) from exc
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> AudioArray:
        return await self._queue.get()

    def flush(self) -> None:
        """Drop queued frames, e.g. audio recorded while Yarvis itself was talking."""
        while not self._queue.empty():
            self._queue.get_nowait()

    def _callback(self, indata: AudioArray, frames: int, time: object, status: object) -> None:
        """PortAudio thread: copy the block and pass it to the event loop."""
        del frames, time, status
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        frame = np.array(indata[:, 0], dtype=np.float32)
        loop.call_soon_threadsafe(self._push, frame)

    def _push(self, frame: AudioArray) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(frame)
