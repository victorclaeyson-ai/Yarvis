"""Everything Yarvis hears: one microphone shared by the wake word, listening and barge-in.

Three consumers take turns reading the same stream:

* idle: the wake-word detector waits for "Hey Yarvis";
* listening: the endpointer captures what you say;
* replying: a barge-in monitor waits for you to interrupt.

Hand-overs pass audio along instead of dropping it. After the wake word, the last two
seconds go to the listener: detection lands up to half a second after the phrase, so a
request said in one breath ("Hey Yarvis what time is it") would otherwise lose its first
words. After a voice barge-in, the speech that triggered it is kept. Anything recorded
while nobody was reading (such as Yarvis's own voice) is flushed before a fresh start.

Privacy: audio only lives in memory, a few seconds at most, and is never written to disk.
It is never sent anywhere either, with one exception you opt into: with an AssemblyAI key
stored, the speech of each request is streamed to AssemblyAI while you say it (see
``speech.assemblyai``). The language model only ever receives the transcript.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, Protocol, Self

from yarvis.audio.devices import looks_like_headphones
from yarvis.listen import MicrophoneStalledError, SignalCheck, listen, read_frame
from yarvis.speech.base import FRAME_SAMPLES, MIC_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from yarvis.config.models import BargeIn
    from yarvis.listen import AudioTap, PauseHooks, Utterance
    from yarvis.speech.base import AudioArray, VoiceActivityDetector
    from yarvis.speech.endpointing import EndpointerSettings

logger = logging.getLogger(__name__)

FRAME_MS: Final = FRAME_SAMPLES * 1000 // MIC_SAMPLE_RATE
WAKE_PREFIX_MS: Final = 2_000
"""Audio before the detection handed to the listener (covers the phrase plus latency)."""
BARGE_IN_PREFIX_MS: Final = 800
BARGE_IN_SPEECH_MS: Final = 320
"""Sustained speech needed to interrupt by voice: a cough or "mm-hm" is not enough."""
BARGE_IN_THRESHOLD: Final = 0.7
"""Stricter than the endpointer, because a false interruption cuts Yarvis off."""

BargeInMode = Literal["voice", "wake_word", "off"]


class FrameSource(Protocol):
    """A microphone: used as ``async with``, then iterated for frames.

    Reading must be safe to cancel: a cancelled ``anext`` loses no audio and leaves the
    stream usable, because readers hand the stream to each other by cancelling.
    """

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def __aiter__(self) -> AsyncIterator[AudioArray]:
        """Iterate frames."""
        ...

    def flush(self) -> None:
        """Drop frames that are queued but not read yet."""
        ...


class WakeWord(Protocol):
    """A streaming wake-word detector (see ``speech.wakeword``)."""

    def reset(self) -> None:
        """Forget all audio heard so far."""
        ...

    def detected(self, frame: AudioArray) -> bool:
        """Feed one frame; True when the wake word was just heard."""
        ...


@dataclass(frozen=True, slots=True)
class Interruption:
    """The user interrupted Yarvis while it was replying."""

    prefix: tuple[AudioArray, ...]
    """Audio already captured, to be heard as the start of the next request."""
    wake_word: bool
    """True when interrupted by saying the wake word; False when by simply talking."""


def barge_in_mode(setting: BargeIn, *, output_device: str | None, wake_word: bool) -> BargeInMode:
    """Decide how the user may interrupt, given the setting and the hardware.

    Voice barge-in needs headphones: on speakers the microphone hears Yarvis itself, and
    PortAudio offers no echo cancellation. Saying the wake word works on speakers because
    the detector only reacts to "Hey Yarvis", not to whatever Yarvis is saying.
    """
    if setting == "off":
        return "off"
    if setting == "voice":
        return "voice"
    if setting == "auto" and looks_like_headphones(output_device):
        return "voice"
    if not wake_word:
        if setting == "wake_word":
            logger.warning("barge_in = 'wake_word' needs the wake word; barge-in is off")
        return "off"
    return "wake_word"


class SharedMicrophone:
    """Keeps one input stream open while anyone holds it, or all the time (``always_on``).

    Everything runs on one event loop; the lock only orders opening and closing.
    """

    def __init__(self, open_source: Callable[[], FrameSource], *, always_on: bool) -> None:
        self.always_on = always_on
        self._open_source = open_source
        self._stack: AsyncExitStack | None = None
        self._source: FrameSource | None = None
        self._frames: AsyncIterator[AudioArray] | None = None
        self._holders = 0
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        """Whether the stream is open (the OS shows its microphone indicator)."""
        return self._source is not None

    async def __aenter__(self) -> Self:
        if self.always_on:
            async with self._lock:
                await self._open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        async with self._lock:
            await self._close()

    @asynccontextmanager
    async def hold(self) -> AsyncGenerator[None]:
        """Keep the stream open for the duration of the block (opening it if needed)."""
        async with self._lock:
            if self._source is None:
                await self._open()
            self._holders += 1
        try:
            yield
        finally:
            self._holders -= 1
            if self._holders == 0 and not self.always_on:
                async with self._lock:
                    if self._holders == 0:  # nobody grabbed it while we waited
                        await self._close()

    def frames(self) -> AsyncIterator[AudioArray]:
        """The live frame iterator; each frame is delivered to one reader only."""
        if self._frames is None:
            msg = "the microphone is not open"
            raise RuntimeError(msg)
        return self._frames

    def flush(self) -> None:
        """Drop audio nobody has read yet."""
        if self._source is not None:
            self._source.flush()

    async def _open(self) -> None:
        stack = AsyncExitStack()
        source = await stack.enter_async_context(self._open_source())
        self._stack, self._source, self._frames = stack, source, aiter(source)
        logger.info("Microphone on")

    async def _close(self) -> None:
        stack = self._stack
        self._stack = self._source = self._frames = None
        if stack is not None:
            await stack.aclose()
            logger.info("Microphone off")


class MicrophoneEars:
    """The assistant's ``Ears``, backed by a shared microphone."""

    def __init__(
        self,
        microphone: SharedMicrophone,
        *,
        vad: VoiceActivityDetector,
        settings: EndpointerSettings,
        wake_word: WakeWord | None = None,
        barge_in: BargeInMode = "off",
        barge_in_vad: VoiceActivityDetector | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        tap: AudioTap | None = None,
    ) -> None:
        if barge_in == "wake_word" and wake_word is None:
            msg = "wake-word barge-in needs a wake-word detector"
            raise ValueError(msg)
        if barge_in == "voice" and barge_in_vad is None:
            msg = "voice barge-in needs its own voice activity detector"
            raise ValueError(msg)
        self._microphone = microphone
        self._vad = vad
        self._settings = settings
        self._wake_word = wake_word
        self._barge_in: BargeInMode = barge_in
        self._barge_in_vad = barge_in_vad
        self._on_speech_start = on_speech_start
        self._on_level = on_level
        self._tap = tap

    @property
    def wake_word(self) -> bool:
        """Whether ``wake()`` is available."""
        return self._wake_word is not None

    @property
    def barge_in(self) -> bool:
        """Whether ``interruption()`` is available."""
        return self._barge_in != "off"

    def session(self) -> AbstractAsyncContextManager[None]:
        """Keep the microphone open across several steps (listening, then replying)."""
        return self._microphone.hold()

    async def listen(
        self,
        hooks: PauseHooks,
        *,
        follow_up: bool = False,
        prefix: Sequence[AudioArray] | None = None,
        timeout_s: float | None = None,
    ) -> Utterance | None:
        """Capture one request.

        ``prefix`` is audio that was already captured (after the wake word or a barge-in);
        the stream then continues without dropping anything. Without it, stale audio is
        flushed first. ``timeout_s`` replaces the usual wait for speech to start (a
        follow-up listens only for the time left in the conversation).
        """
        del follow_up  # the assistant decides what silence means; the microphone does not
        settings = self._settings
        if timeout_s is not None:
            settings = replace(settings, no_speech_timeout_ms=max(1, round(timeout_s * 1000)))
        async with self._microphone.hold():
            if prefix is None:
                self._microphone.flush()
            async with aclosing(_chain(prefix or (), self._microphone.frames())) as frames:
                return await listen(
                    frames,
                    self._vad,
                    settings,
                    on_speech_start=self._on_speech_start,
                    on_level=self._on_level,
                    hooks=hooks,
                    tap=self._tap,
                )

    async def wake(self) -> tuple[AudioArray, ...]:
        """Wait for the wake word; returns the audio around it (see ``WAKE_PREFIX_MS``)."""
        detector = self._require_wake_word()
        async with self._microphone.hold():
            self._microphone.flush()
            return await self._until_wake_word(detector)

    async def interruption(self, *, wake_word_only: bool = False) -> Interruption:
        """Wait until the user interrupts (call while Yarvis is replying).

        ``wake_word_only``: ignore plain talking, e.g. while music plays that the microphone
        would take for the user.
        """
        async with self._microphone.hold():
            self._microphone.flush()  # the end of the user's own request
            if self._barge_in == "voice" and not wake_word_only:
                return Interruption(await self._until_voice(), wake_word=False)
            if self._barge_in != "off":
                prefix = await self._until_wake_word(self._require_wake_word())
                return Interruption(prefix, wake_word=True)
        msg = "barge-in is off"
        raise RuntimeError(msg)

    async def _until_wake_word(self, detector: WakeWord) -> tuple[AudioArray, ...]:
        detector.reset()
        recent: deque[AudioArray] = deque(maxlen=_frames(WAKE_PREFIX_MS))
        check_signal = SignalCheck(FRAME_MS)
        iterator = self._microphone.frames()
        while (frame := await read_frame(iterator)) is not None:
            check_signal(frame)
            recent.append(frame)
            if detector.detected(frame):
                logger.info("Wake word detected")
                detector.reset()
                return tuple(recent)
        raise MicrophoneStalledError

    async def _until_voice(self) -> tuple[AudioArray, ...]:
        vad = self._barge_in_vad
        if vad is None:  # pragma: no cover - guaranteed by __init__
            msg = "voice barge-in needs a voice activity detector"
            raise RuntimeError(msg)
        vad.reset()
        recent: deque[AudioArray] = deque(maxlen=_frames(BARGE_IN_PREFIX_MS))
        needed = _frames(BARGE_IN_SPEECH_MS)
        score = 0
        check_signal = SignalCheck(FRAME_MS)
        iterator = self._microphone.frames()
        while (frame := await read_frame(iterator)) is not None:
            check_signal(frame)
            recent.append(frame)
            # Short dips inside a word are fine; a real pause quickly resets the count.
            score = score + 1 if vad.probability(frame) >= BARGE_IN_THRESHOLD else max(0, score - 2)
            if score >= needed:
                logger.info("Barge-in: the user started talking")
                return tuple(recent)
        raise MicrophoneStalledError

    def _require_wake_word(self) -> WakeWord:
        if self._wake_word is None:
            msg = "the wake word is off"
            raise RuntimeError(msg)
        return self._wake_word


def _frames(milliseconds: int) -> int:
    return max(1, -(-milliseconds // FRAME_MS))


async def _chain(
    prefix: Sequence[AudioArray], frames: AsyncIterator[AudioArray]
) -> AsyncGenerator[AudioArray]:
    for frame in prefix:
        yield frame
    async for frame in frames:
        yield frame
