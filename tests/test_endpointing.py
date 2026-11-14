from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from yarvis.speech.base import FRAME_SAMPLES
from yarvis.speech.endpointing import Endpointer, EndpointerSettings, EndpointEvent

SETTINGS = EndpointerSettings(
    frame_ms=32,
    min_speech_ms=96,  # 3 frames
    end_of_turn_ms=320,  # 10 frames
    pre_roll_ms=160,  # 5 frames
    keep_trailing_ms=64,  # 2 frames
    no_speech_timeout_ms=3_200,  # 100 frames
    max_utterance_ms=6_400,  # 200 frames
)


def frame(value: float = 0.0) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


def run(endpointer: Endpointer, probabilities: list[float]) -> list[EndpointEvent]:
    events: list[EndpointEvent] = []
    for index, probability in enumerate(probabilities):
        event = endpointer.push(frame(index), probability)
        if event is not EndpointEvent.NONE:
            events.append(event)
    return events


def test_full_turn() -> None:
    endpointer = Endpointer(SETTINGS)
    events = run(endpointer, [0.0] * 20 + [0.9] * 30 + [0.0] * 10)
    assert events == [EndpointEvent.SPEECH_STARTED, EndpointEvent.END_OF_TURN]
    assert not endpointer.speaking
    assert endpointer.trailing_silence_ms == 320


def test_utterance_keeps_pre_roll_and_trims_trailing_silence() -> None:
    endpointer = Endpointer(SETTINGS)
    run(endpointer, [0.0] * 20 + [0.9] * 30 + [0.0] * 10)
    audio = endpointer.utterance()
    frames = audio.reshape(-1, FRAME_SAMPLES)[:, 0].astype(int).tolist()
    # Pre-roll: the 5 frames before speech was confirmed (indices 18..22 include the
    # first speech frames), then all speech, then only 2 of the 10 silent frames.
    assert frames[0] == 18
    assert frames[-1] == 51
    assert len(frames) == 5 + 27 + 2


def test_short_blips_do_not_start_speech() -> None:
    endpointer = Endpointer(SETTINGS)
    assert run(endpointer, [0.9, 0.9, 0.0] * 10) == []


def test_hysteresis_dips_do_not_end_the_turn() -> None:
    endpointer = Endpointer(SETTINGS)
    # Probability 0.4 is below the start threshold but above the negative threshold.
    events = run(endpointer, [0.9] * 5 + [0.4] * 50)
    assert events == [EndpointEvent.SPEECH_STARTED]
    assert endpointer.speaking


def test_no_speech_timeout() -> None:
    endpointer = Endpointer(SETTINGS)
    assert run(endpointer, [0.1] * 150) == [EndpointEvent.NO_SPEECH]
    assert endpointer.utterance().size == 0


def test_max_duration() -> None:
    endpointer = Endpointer(SETTINGS)
    events = run(endpointer, [0.9] * 300)
    assert events == [EndpointEvent.SPEECH_STARTED, EndpointEvent.MAX_DURATION]


def test_frames_rounds_up() -> None:
    assert SETTINGS.frames(33) == 2
    assert SETTINGS.frames(0) == 1


@given(st.lists(st.floats(0, 1), max_size=400))
def test_events_are_well_ordered(probabilities: list[float]) -> None:
    events = run(Endpointer(SETTINGS), probabilities)
    assert len(events) <= 2
    if len(events) == 2:
        assert events[0] is EndpointEvent.SPEECH_STARTED
        assert events[1] in {EndpointEvent.END_OF_TURN, EndpointEvent.MAX_DURATION}
    if events == [EndpointEvent.NO_SPEECH]:
        assert EndpointEvent.SPEECH_STARTED not in events


@pytest.mark.parametrize("default", [EndpointerSettings()])
def test_default_settings_are_sane(default: EndpointerSettings) -> None:
    assert default.negative_threshold < default.threshold
    assert default.frames(default.pre_roll_ms) >= default.frames(default.min_speech_ms)
