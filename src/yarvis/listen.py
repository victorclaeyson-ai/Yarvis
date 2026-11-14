"""Capture one utterance: from the first word to the end-of-turn silence."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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
    clock: Callable[[], float] = time.monotonic,
) -> Utterance | None:
    """Consume ``frames`` until the user finishes a sentence.

    ``on_level`` receives the RMS level of every frame (for a live level meter); ``hooks``
    are told about pauses so speculative work can start early.

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
        clock=clock,
    )
    iterator = aiter(frames)
    while (frame := await _next_frame(iterator)) is not None:
        finished, utterance = session.push(frame)
        if finished:
            return utterance
    return None


async def _next_frame(iterator: AsyncIterator[AudioArray]) -> AudioArray | None:
    try:
        async with asyncio.timeout(STALLED_MIC_SECONDS):
            return await anext(iterator)
    except StopAsyncIteration:
        return None
    except TimeoutError:
        logger.warning("No audio from the microphone for %.1f s", STALLED_MIC_SECONDS)
        raise MicrophoneStalledError from None


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
        clock: Callable[[], float],
    ) -> None:
        self._endpointer = Endpointer(settings)
        self._vad = vad
        self._on_speech_start = on_speech_start
        self._on_level = on_level
        self._hooks = hooks
        self._clock = clock
        self._silent_limit = settings.frames(SILENT_MIC_MS)
        self._silent_frames = 0
        self._heard_signal = False
        self._opened = clock()
        vad.reset()

    def push(self, frame: AudioArray) -> tuple[bool, Utterance | None]:
        """Process one frame. Returns (finished, utterance)."""
        self._check_signal(frame)
        if self._on_level is not None:
            self._on_level(float(np.sqrt(np.mean(np.square(frame)))))
        event = self._endpointer.push(frame, self._vad.probability(frame))
        if event in (EndpointEvent.END_OF_TURN, EndpointEvent.MAX_DURATION):
            now = self._clock()
            logger.info("End of turn (%s)", event.value)
            utterance = Utterance(
                audio=self._endpointer.utterance(),
                speech_end=now - self._endpointer.trailing_silence_ms / 1000,
                turn_detected=now,
            )
            return True, utterance
        if event is EndpointEvent.NO_SPEECH:
            logger.info("No speech before the timeout")
            return True, None
        self._notify(event)
        return False, None

    def _check_signal(self, frame: AudioArray) -> None:
        if self._heard_signal:
            return
        if frame.any():
            self._heard_signal = True
            waited = (self._clock() - self._opened) * 1000
            logger.info("Microphone delivering audio after %d ms", waited)
            return
        self._silent_frames += 1
        if self._silent_frames >= self._silent_limit:
            raise MicrophoneSilentError

    def _notify(self, event: EndpointEvent) -> None:
        if event is EndpointEvent.SPEECH_STARTED:
            logger.info("Speech started")
            if self._on_speech_start is not None:
                self._on_speech_start()
        elif event is EndpointEvent.PAUSE and self._hooks is not None:
            self._hooks.on_pause(self._endpointer.utterance(), _EndpointerControl(self._endpointer))
        elif event is EndpointEvent.RESUME and self._hooks is not None:
            self._hooks.on_resume()
