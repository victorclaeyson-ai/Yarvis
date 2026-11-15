from __future__ import annotations

import pytest

from yarvis.speech.turn_taking import ends_conversation, sounds_finished, strip_wake_phrase


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


# --------------------------------------------------------------------------- wake phrase


@pytest.mark.parametrize(
    ("text", "anywhere", "expected"),
    [
        ("Hey Yarvis, what time is it?", False, "what time is it?"),
        ("Hey Yarvis what time is it?", False, "what time is it?"),
        ("Hey, Jarvis. What's up?", False, "What's up?"),
        ("Okay Yarvis", False, ""),
        ("Yarvis, what time is it?", False, "what time is it?"),
        ("Yarvis!", False, ""),
        ("  Hey Yarvis!  ", False, ""),
        ("hey jarvis. hey jarvis what's up", False, "what's up"),  # repeated
        # A name inside the sentence is not an address.
        ("Yarvis is a cool name.", False, "Yarvis is a cool name."),
        ("Yarvis is a cool name.", True, "Yarvis is a cool name."),
        ("Who was Jarvis in Iron Man?", True, "Who was Jarvis in Iron Man?"),
        # After the wake word's pre-roll, whatever came before the phrase is dropped...
        ("so anyway. Hey Yarvis, what's the weather?", True, "what's the weather?"),
        ("so anyway. Hey Yarvis, hi", False, "so anyway. Hey Yarvis, hi"),
        # ...but only near the start, not deep inside a long request.
        (
            "What is the weather like in Ghent today and tomorrow? Ask Yarvis.",
            True,
            "What is the weather like in Ghent today and tomorrow? Ask Yarvis.",
        ),
        ("", True, ""),
    ],
)  # fmt: skip
def test_strip_wake_phrase(text: str, anywhere: bool, expected: str) -> None:
    assert strip_wake_phrase(text, anywhere=anywhere) == expected


def test_strip_wake_phrase_with_a_custom_name() -> None:
    assert strip_wake_phrase("Hey Friday, lights on", names=("Friday",)) == "lights on"
    assert strip_wake_phrase("Hey Yarvis, lights on", names=("Friday",)) == "Hey Yarvis, lights on"


@pytest.mark.parametrize(
    "text",
    [
        "Thanks!", "Thank you so much, Yarvis.", "That's all.", "That's all for now, thanks.",
        "Never mind.", "Okay, bye.", "Good night, Jarvis.", "No thanks.", "Stop.", "Cheers",
    ],
)  # fmt: skip
def test_goodbyes_close_the_conversation(text: str) -> None:
    assert ends_conversation(text)


@pytest.mark.parametrize(
    "text",
    [
        "Okay.", "No.", "", "Yarvis", "Thanks, and what's the weather tomorrow?",
        "Thank you for the recipe, can you also give me one for dessert?",
        "Stop the timer and set a new one for ten minutes please",
    ],
)  # fmt: skip
def test_requests_keep_the_conversation_open(text: str) -> None:
    assert not ends_conversation(text)
