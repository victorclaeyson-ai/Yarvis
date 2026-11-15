"""Decide when the user has started and finished speaking.

The endpointer is pure logic: it receives audio frames together with the voice-activity
probability for each frame and emits events. It never touches hardware, which makes it
easy to test exhaustively.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from yarvis.speech.base import AudioArray


class EndpointEvent(StrEnum):
    """What happened after pushing a frame."""

    NONE = "none"
    SPEECH_STARTED = "speech_started"
    PAUSE = "pause"
    """The user went quiet; they may be done. Good moment to start work speculatively."""
    RESUME = "resume"
    """The user spoke again after a ``PAUSE``: discard speculative work."""
    END_OF_TURN = "end_of_turn"
    NO_SPEECH = "no_speech"
    MAX_DURATION = "max_duration"


@dataclass(frozen=True, slots=True)
class EndpointerSettings:
    """Timing thresholds. All durations are in milliseconds."""

    frame_ms: int = 32
    threshold: float = 0.5
    negative_threshold: float = 0.35
    min_speech_ms: int = 160
    end_of_turn_ms: int = 700
    pause_ms: int = 224
    """Silence after which a ``PAUSE`` is reported (must be shorter than ``end_of_turn_ms``)."""
    pre_roll_ms: int = 320
    keep_trailing_ms: int = 160
    no_speech_timeout_ms: int = 8_000
    max_utterance_ms: int = 30_000

    def frames(self, milliseconds: int) -> int:
        """Convert a duration to a whole number of frames (at least one)."""
        return max(1, -(-milliseconds // self.frame_ms))


class _State(StrEnum):
    WAITING = "waiting"
    SPEAKING = "speaking"
    DONE = "done"


class Endpointer:
    """Turn per-frame speech probabilities into start / end-of-turn events.

    Uses hysteresis: speech starts above ``threshold`` and only counts as silence below
    ``negative_threshold``, so brief dips in the middle of a word do not end the turn.
    """

    def __init__(self, settings: EndpointerSettings | None = None) -> None:
        self.settings = settings or EndpointerSettings()
        s = self.settings
        self._pre_roll: deque[AudioArray] = deque(
            maxlen=max(s.frames(s.pre_roll_ms), s.frames(s.min_speech_ms))
        )
        self._utterance: list[AudioArray] = []
        self._state = _State.WAITING
        self._waited = 0
        self._candidate = 0
        self._silence = 0
        self._paused = False
        self._required_silence = s.frames(s.end_of_turn_ms)

    @property
    def speaking(self) -> bool:
        """Whether speech has started and not yet ended."""
        return self._state is _State.SPEAKING

    def require_silence_ms(self, milliseconds: int) -> None:
        """Change how much silence ends *this* pause (e.g. wait longer mid-sentence).

        Reset to ``end_of_turn_ms`` whenever the user speaks again.
        """
        self._required_silence = self.settings.frames(milliseconds)

    @property
    def trailing_silence_ms(self) -> int:
        """Silence since the last voiced frame, in milliseconds."""
        return self._silence * self.settings.frame_ms

    def push(self, frame: AudioArray, probability: float) -> EndpointEvent:
        """Feed one frame and its speech probability."""
        if self._state is _State.DONE:
            return EndpointEvent.NONE
        if self._state is _State.WAITING:
            return self._push_waiting(frame, probability)
        return self._push_speaking(frame, probability)

    def utterance(self) -> AudioArray:
        """The captured speech, including a little context before and after."""
        if not self._utterance:
            return np.zeros(0, dtype=np.float32)
        s = self.settings
        drop = max(0, self._silence - s.frames(s.keep_trailing_ms))
        frames = self._utterance[: len(self._utterance) - drop] if drop else self._utterance
        return np.concatenate(frames).astype(np.float32, copy=False)

    def _push_waiting(self, frame: AudioArray, probability: float) -> EndpointEvent:
        s = self.settings
        self._pre_roll.append(frame)
        self._waited += 1
        self._candidate = self._candidate + 1 if probability >= s.threshold else 0
        if self._candidate >= s.frames(s.min_speech_ms):
            self._state = _State.SPEAKING
            self._utterance = list(self._pre_roll)
            self._pre_roll.clear()
            return EndpointEvent.SPEECH_STARTED
        if self._waited >= s.frames(s.no_speech_timeout_ms):
            self._state = _State.DONE
            return EndpointEvent.NO_SPEECH
        return EndpointEvent.NONE

    def _push_speaking(self, frame: AudioArray, probability: float) -> EndpointEvent:
        s = self.settings
        self._utterance.append(frame)
        if probability >= s.negative_threshold:
            self._silence = 0
            self._required_silence = s.frames(s.end_of_turn_ms)
            if self._paused:
                self._paused = False
                return EndpointEvent.RESUME
        else:
            self._silence += 1
        if self._silence >= self._required_silence:
            self._state = _State.DONE
            return EndpointEvent.END_OF_TURN
        if len(self._utterance) >= s.frames(s.max_utterance_ms):
            self._state = _State.DONE
            return EndpointEvent.MAX_DURATION
        if not self._paused and self._silence == s.frames(s.pause_ms):
            self._paused = True
            return EndpointEvent.PAUSE
        return EndpointEvent.NONE
