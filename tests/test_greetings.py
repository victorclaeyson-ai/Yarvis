from __future__ import annotations

from datetime import UTC, datetime

import pytest

from yarvis.agent.greetings import Greeter


def at(hour: int) -> datetime:
    return datetime(2026, 11, 15, hour, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (2, "Still up, Victor?"),
        (7, "Good morning, Victor."),
        (13, "Good afternoon, Victor."),
        (21, "Good evening, Victor."),
    ],
)
def test_the_first_greeting_fits_the_time_of_day(hour: int, expected: str) -> None:
    assert Greeter("Victor", clock=lambda: at(hour)).next() == expected


def test_later_greetings_are_short_and_never_repeat() -> None:
    greeter = Greeter("Victor", clock=lambda: at(21), choose=lambda options: options[0])
    phrases = [greeter.next() for _ in range(4)]
    assert phrases[0] == "Good evening, Victor."
    assert phrases[1:] == ["Yes, Victor?", "I'm listening.", "Yes, Victor?"]
    assert all(phrase in greeter.phrases() for phrase in phrases)


def test_a_new_part_of_the_day_gets_a_new_greeting() -> None:
    now = [at(11)]
    greeter = Greeter(None, clock=lambda: now[0])
    assert greeter.next() == "Good morning."
    now[0] = at(12)
    assert greeter.next() == "Good afternoon."


def test_without_a_name() -> None:
    greeter = Greeter(clock=lambda: at(21))
    assert greeter.phrases() == [
        "Good evening.", "Yes?", "I'm listening.", "At your service.", "How can I help?",
        "Go ahead.",
    ]  # fmt: skip
