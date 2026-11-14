"""Capture one utterance: from the first word to the end-of-turn silence."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    clock: Callable[[], float] = time.monotonic,
) -> Utterance | None:
    """Consume ``frames`` until the user finishes a sentence.

    ``on_level`` receives the RMS level of every frame (for a live level meter).

    Returns:
        The utterance, or None if nobody spoke before the timeout.

    Raises:
        MicrophoneSilentError: If the input is digital silence (permission problem).
    """
    endpointer = Endpointer(settings)
    vad.reset()
    silent_limit = settings.frames(SILENT_MIC_MS)
    heard_signal = False
    silent_frames = 0
    iterator = aiter(frames)
    opened = clock()
    while True:
        try:
            async with asyncio.timeout(STALLED_MIC_SECONDS):
                frame = await anext(iterator)
        except StopAsyncIteration:
            return None
        except TimeoutError:
            logger.warning("No audio from the microphone for %.1f s", STALLED_MIC_SECONDS)
            raise MicrophoneStalledError from None
        if not heard_signal:
            if frame.any():
                heard_signal = True
                logger.info("Microphone delivering audio after %d ms", (clock() - opened) * 1000)
            else:
                silent_frames += 1
                if silent_frames >= silent_limit:
                    raise MicrophoneSilentError
        if on_level is not None:
            on_level(float(np.sqrt(np.mean(np.square(frame)))))
        event = endpointer.push(frame, vad.probability(frame))
        if event is EndpointEvent.SPEECH_STARTED:
            logger.info("Speech started")
            if on_speech_start is not None:
                on_speech_start()
        elif event in (EndpointEvent.END_OF_TURN, EndpointEvent.MAX_DURATION):
            now = clock()
            logger.info("End of turn (%s)", event.value)
            return Utterance(
                audio=endpointer.utterance(),
                speech_end=now - endpointer.trailing_silence_ms / 1000,
                turn_detected=now,
            )
        elif event is EndpointEvent.NO_SPEECH:
            logger.info("No speech before the timeout")
            return None
