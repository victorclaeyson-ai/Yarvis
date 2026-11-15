from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yarvis.speech.sentences import SentenceSplitter, SpeechFilter, to_speakable


def split(
    text: str,
    *,
    chunk: int = 3,
    min_chars: int = 40,
    max_chars: int = 220,
    first_clause_chars: int | None = 24,
) -> list[str]:
    """Feed ``text`` in small chunks, like a token stream."""
    splitter = SentenceSplitter(
        min_chars=min_chars, max_chars=max_chars, first_clause_chars=first_clause_chars
    )
    out: list[str] = []
    for start in range(0, len(text), chunk):
        out.extend(splitter.feed(text[start : start + chunk]))
    if tail := splitter.flush():
        out.append(tail)
    return out


def test_first_sentence_is_emitted_immediately_even_if_short() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Sure. ") == ["Sure."]


def test_first_clause_is_emitted_at_a_comma_when_long_enough() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("It's grey and drizzly in Zedelgem") == []
    assert splitter.feed(", as usual for November. ") == ["It's grey and drizzly in Zedelgem,"]
    assert splitter.flush() == "as usual for November."


def test_a_short_opening_clause_is_spoken_first() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Right now in Leuven, it's grey") == ["Right now in Leuven,"]
    assert splitter.feed(" and drizzly. ") == []  # later sentences are merged
    assert splitter.flush() == "it's grey and drizzly."


def test_short_first_clauses_wait_for_more_text() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Well, ") == []
    assert splitter.feed("that depends. ") == ["Well, that depends."]


def test_first_clause_split_can_be_disabled() -> None:
    splitter = SentenceSplitter(first_clause_chars=None)
    assert splitter.feed("It's grey and drizzly in Zedelgem, as usual. ") == [
        "It's grey and drizzly in Zedelgem, as usual."
    ]


def test_numbers_with_commas_are_not_split() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("The population is about 24,000 people. ") == [
        "The population is about 24,000 people."
    ]


def test_later_short_sentences_are_merged() -> None:
    text = "Sure. It is sunny. It is warm. Tomorrow will be rainy and quite a bit colder."
    assert split(text, min_chars=20, first_clause_chars=None) == [
        "Sure.",
        "It is sunny. It is warm.",
        "Tomorrow will be rainy and quite a bit colder.",
    ]


def test_waits_for_the_character_after_a_terminator() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Hello there.") == []
    assert splitter.feed(" How") == ["Hello there."]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith will see you now.",
        "Bring fruit, e.g. apples, and water.",
        "J. R. R. Tolkien wrote it.",
        "Pi is roughly 3.14 these days.",
    ],
)
def test_no_false_boundaries(text: str) -> None:
    assert split(text, first_clause_chars=None) == [text]


def test_question_and_exclamation_marks_and_quotes() -> None:
    assert split('Really?! He said "no." Then left.', min_chars=5) == [
        "Really?!",
        'He said "no."',
        "Then left.",
    ]


def test_newlines_end_sentences() -> None:
    assert split("First line\nSecond line that is long enough\n") == [
        "First line",
        "Second line that is long enough",
    ]


def test_long_text_without_punctuation_is_split_at_a_clause() -> None:
    text = ("word " * 30) + "and then, more words " * 10
    pieces = split(text, max_chars=120)
    assert all(len(piece) <= 120 for piece in pieces[:-1])
    assert len(pieces) > 1


def test_flush_on_empty_returns_none() -> None:
    assert SentenceSplitter().flush() is None


def test_invalid_settings() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        SentenceSplitter(min_chars=50, max_chars=40)


@given(text=st.text(max_size=400), chunk=st.integers(1, 20))
def test_no_text_is_lost_or_reordered(text: str, chunk: int) -> None:
    pieces = split(text, chunk=chunk)
    assert "".join("".join(pieces).split()) == "".join(text.split())


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("**Bold** and *italic* and `code`.", "Bold and italic and code."),
        ("See [the docs](https://example.com) now.", "See the docs now."),
        ("Visit https://example.com today.", "Visit a link today."),
        ("## Heading", "Heading"),
        ("- first item", "first item"),
        ("1. step one", "step one"),
        ("```python\nprint(1)\n```", ""),
        ("The answer is no.", "The answer is no."),
        ("snake_case_name stays", "snake_case_name stays"),
    ],
)
def test_to_speakable(raw: str, spoken: str) -> None:
    assert to_speakable(raw) == spoken


def test_speech_filter_skips_code_blocks_across_sentences() -> None:
    speak = SpeechFilter()
    pieces = ["Here you go:", "```python", "print('hi')", "```", "That prints hi."]
    assert [speak(piece) for piece in pieces] == ["Here you go:", "", "", "", "That prints hi."]


def test_speech_filter_handles_inline_fences() -> None:
    speak = SpeechFilter()
    assert speak("```x```") == ""
    assert speak("Normal again.") == "Normal again."
