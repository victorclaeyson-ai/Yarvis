"""Capture one utterance: from the first word to the end-of-turn silence."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from yarvis.audio.devices import AudioBackendError
from yarvis.speech.endpointing import Endpointer, EndpointEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from yarvis.speech.base import AudioArray, VoiceActivityDetector
    from yarvis.speech.endpointing import EndpointerSettings

logger = logging.getLogger(__name__)

# macOS delivers pure digital silence (not an error) when microphone access is denied.
SILENT_MIC_MS = 1_500
# If no audio arrives at all for this long, the input device is stuck or gone.
STALLED_MIC_SECONDS = 2.5


class MicrophoneSilentError(AudioBackendError):
    """The microphone delivers nothing but zeros: usually a missing OS permission."""

    def __init__(self) -> None:
        super().__init__(
            "the microphone is completely silent. On macOS, allow your terminal app under "
            "System Settings → Privacy & Security → Microphone, then restart the terminal."
        )


class MicrophoneStalledError(AudioBackendError):
    """The microphone stopped delivering audio."""

    def __init__(self) -> None:
        super().__init__(
            "the microphone is not delivering any audio. Check the input device in "
            "System Settings → Sound, or set voice.input_device in the config."
        )


class PauseControl(Protocol):
    """Lets whoever handles a pause decide how long to keep waiting."""

    def require_silence_ms(self, milliseconds: int) -> None:
        """End the turn only after this much silence (counted from the end of speech)."""
        ...


class _EndpointerControl:
    """``PauseControl`` acting on the live endpointer of a ``listen`` call."""

    def __init__(self, endpointer: Endpointer) -> None:
        self._endpointer = endpointer

    def require_silence_ms(self, milliseconds: int) -> None:
        logger.info("Sentence sounds unfinished: waiting up to %d ms", milliseconds)
        self._endpointer.require_silence_ms(milliseconds)


@runtime_checkable
class AudioTap(Protocol):
    """Receives the user's speech while they talk (streaming speech recognition).

    ``begin``, then ``feed`` for every later frame, then ``end``: once per utterance.
    """

    def begin(self, audio: AudioArray) -> None:
        """Speech started; ``audio`` is what was captured so far (with its pre-roll)."""
        ...

    def feed(self, frame: AudioArray) -> None:
        """The next frame of the same utterance."""
        ...

    def end(self, audio: AudioArray | None) -> None:
        """The utterance is complete (``audio``, as it will be transcribed) or abandoned."""
        ...


@dataclass(frozen=True, slots=True)
class PauseHooks:
    """Callbacks for speculative work during pauses (see ``pipeline.Speculation``)."""

    on_pause: Callable[[AudioArray, PauseControl], None]
    """The user went quiet; receives the speech so far and a way to wait longer."""
    on_resume: Callable[[], None]
    """The user spoke again: whatever ``on_pause`` started is now stale."""


@dataclass(frozen=True, slots=True)
class Utterance:
    """What the user said, plus the timestamps the latency report needs."""

    audio: AudioArray
    speech_end: float
    turn_detected: float


async def listen(
    frames: AsyncIterator[AudioArray],
    vad: VoiceActivityDetector,
    settings: EndpointerSettings,
    *,
    on_speech_start: Callable[[], None] | None = None,
    on_level: Callable[[float], None] | None = None,
    hooks: PauseHooks | None = None,
    tap: AudioTap | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Utterance | None:
    """Consume ``frames`` until the user finishes a sentence.

    ``on_level`` receives the RMS level of every frame (for a live level meter); ``hooks``
    are told about pauses so speculative work can start early; ``tap`` receives the speech
    as it is spoken.

    Returns:
        The utterance, or None if nobody spoke before the timeout.

    Raises:
        MicrophoneSilentError: If the input is digital silence (permission problem).
        MicrophoneStalledError: If the microphone stops delivering audio.
    """
    session = _Session(
        settings,
        vad,
        on_speech_start=on_speech_start,
        on_level=on_level,
        hooks=hooks,
        tap=tap,
        clock=clock,
    )
    iterator = aiter(frames)
    try:
        while (frame := await read_frame(iterator)) is not None:
            finished, utterance = session.push(frame)
            if finished:
                return utterance
    finally:
        session.close()
    return None


async def read_frame(iterator: AsyncIterator[AudioArray]) -> AudioArray | None:
    """Next microphone frame, or None when the stream ends.

    Raises:
        MicrophoneStalledError: If no audio arrives for ``STALLED_MIC_SECONDS``.
    """
    try:
        async with asyncio.timeout(STALLED_MIC_SECONDS):
            return await anext(iterator)
    except StopAsyncIteration:
        return None
    except TimeoutError:
        logger.warning("No audio from the microphone for %.1f s", STALLED_MIC_SECONDS)
        raise MicrophoneStalledError from None


class SignalCheck:
    """Detects a microphone that delivers pure digital silence (a permission problem)."""

    def __init__(self, frame_ms: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = max(1, -(-SILENT_MIC_MS // frame_ms))  # rounded up
        self._silent_frames = 0
        self._heard_signal = False
        self._clock = clock
        self._opened = clock()

    def __call__(self, frame: AudioArray) -> None:
        """Check one frame.

        Raises:
            MicrophoneSilentError: After ``SILENT_MIC_MS`` of exact zeros.
        """
        if self._heard_signal:
            return
        if frame.any():
            self._heard_signal = True
            waited = (self._clock() - self._opened) * 1000
            logger.info("Microphone delivering audio after %d ms", waited)
            return
        self._silent_frames += 1
        if self._silent_frames >= self._limit:
            raise MicrophoneSilentError


class _Session:
    """State of one ``listen`` call: endpointing plus microphone sanity checks."""

    def __init__(
        self,
        settings: EndpointerSettings,
        vad: VoiceActivityDetector,
        *,
        on_speech_start: Callable[[], None] | None,
        on_level: Callable[[float], None] | None,
        hooks: PauseHooks | None,
        tap: AudioTap | None = None,
        clock: Callable[[], float],
    ) -> None:
        self._endpointer = Endpointer(settings)
        self._vad = vad
        self._on_speech_start = on_speech_start
        self._on_level = on_level
        self._hooks = hooks
        self._tap = tap
        self._tapping = False
        self._clock = clock
        self._check_signal = SignalCheck(settings.frame_ms, clock)
        vad.reset()

    def push(self, frame: AudioArray) -> tuple[bool, Utterance | None]:
        """Process one frame. Returns (finished, utterance)."""
        self._check_signal(frame)
        if self._on_level is not None:
            self._on_level(float(np.sqrt(np.mean(np.square(frame)))))
        event = self._endpointer.push(frame, self._vad.probability(frame))
        self._stream(event, frame)
        if event in (EndpointEvent.END_OF_TURN, EndpointEvent.MAX_DURATION):
            now = self._clock()
            logger.info("End of turn (%s)", event.value)
            utterance = Utterance(
                audio=self._endpointer.utterance(),
                speech_end=now - self._endpointer.trailing_silence_ms / 1000,
                turn_detected=now,
            )
            self._end_tap(utterance.audio)
            return True, utterance
        if event is EndpointEvent.NO_SPEECH:
            logger.info("No speech before the timeout")
            return True, None
        self._notify(event)
        return False, None

    def close(self) -> None:
        """Listening stopped: an utterance still streaming was abandoned."""
        self._end_tap(None)

    def _stream(self, event: EndpointEvent, frame: AudioArray) -> None:
        """Pass the speech to the tap: everything from the start of the utterance on."""
        if self._tap is None:
            return
        if event is EndpointEvent.SPEECH_STARTED:
            self._tapping = True
            self._tap.begin(self._endpointer.utterance())
        elif self._tapping:
            self._tap.feed(frame)

    def _end_tap(self, audio: AudioArray | None) -> None:
        if self._tap is not None and self._tapping:
            self._tapping = False
            self._tap.end(audio)

    def _notify(self, event: EndpointEvent) -> None:
        if event is EndpointEvent.SPEECH_STARTED:
            logger.info("Speech started")
            if self._on_speech_start is not None:
                self._on_speech_start()
        elif event is EndpointEvent.PAUSE and self._hooks is not None:
            self._hooks.on_pause(self._endpointer.utterance(), _EndpointerControl(self._endpointer))
        elif event is EndpointEvent.RESUME and self._hooks is not None:
            self._hooks.on_resume()
