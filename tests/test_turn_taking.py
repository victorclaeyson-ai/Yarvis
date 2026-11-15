from __future__ import annotations

import pytest

from yarvis.speech.turn_taking import (
    ends_conversation,
    meant_for_assistant,
    sounds_finished,
    strip_wake_phrase,
)


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


# --------------------------------------------------------------------------- who is it for?


@pytest.mark.parametrize(
    "said",
    [
        "Open Spotify.", "open spotify", "Can you open Brave?", "Open a new tab.",
        "Okay, so can you please play some jazz?", "Play.",
        "Play Feel What You Want by Kristine W.", "Pause.", "Skip.", "Next song.", "Next",
        "Turn it up.", "Turn the music down", "Turn up the volume", "Louder!", "Volume to 30",
        "Set a timer for ten minutes.", "Set a pasta timer", "Remind me at six to call mom.",
        "Cancel the timer.", "Search for trains to Ghent.", "Google the weather in Leuven",
        "Go back.", "Go to YouTube.", "Go to kuleuven.be", "Switch to the Gmail tab.",
        "Close this tab.", "Click on the first video.", "Type hello in Messages",
        "What's playing?", "What song is this?", "I want to hear something by Daft Punk",
        "Stop the music.", "Hey Yarvis, what's the weather?", "What do you think, Yarvis?",
        "Jarvis, how tall is the Eiffel Tower?", "Shuffle.", "Mute.",
    ],
)  # fmt: skip
def test_requests_for_yarvis(said: str) -> None:
    assert meant_for_assistant(said)


@pytest.mark.parametrize(
    "said",
    [
        "Do you want to play with me?", "Play fair!", "Let's play FIFA.", "Go to bed.",
        "I'm going to the shop.", "What time do you have to leave?", "Stop it!",
        "Turn left at the lights.", "Next time we should go.", "Set the table, please.",
        "Can you pass me the salt?", "That was a good movie.", "Thanks man.",
        "Where did you put my keys?", "Close the door.", "Search me.", "",
        "And then we went to Ghent.", "What about tomorrow?", "Open the window.",
        "Stop!", "Play with your brother.",
    ],
)  # fmt: skip
def test_talking_to_someone_else(said: str) -> None:
    assert not meant_for_assistant(said)


def test_right_after_an_answer_a_follow_up_question_counts() -> None:
    assert meant_for_assistant("And what about tomorrow?", just_answered=True)
    assert meant_for_assistant("How about Brussels?", just_answered=True)
    assert not meant_for_assistant("And then we went to Ghent.", just_answered=True)
    assert not meant_for_assistant("Do you want to play with me?", just_answered=True)
