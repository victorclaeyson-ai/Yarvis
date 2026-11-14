from __future__ import annotations

import pytest

from yarvis.speech.turn_taking import sounds_finished


@pytest.mark.parametrize(
    "text",
    [
        "What's the weather in Zedelgem?",
        "Set a timer for ten minutes.",
        "Thanks!",
        "So what?",
        "hey yarvis what time is it",  # no punctuation: give the benefit of the doubt
        "",
        "   ",
    ],
)
def test_finished(text: str) -> None:
    assert sounds_finished(text)


@pytest.mark.parametrize(
    "text",
    [
        "So I was thinking about the",
        "So I was thinking about the.",  # Whisper sometimes adds a period anyway
        "I want to plan a trip and",
        "Remind me to call, um",
        "Can you check my",
        "Well,",
        "The thing is...",
        "The thing is…",
        "First of all:",
        "I'd like to",
    ],
)
def test_unfinished(text: str) -> None:
    assert not sounds_finished(text)
