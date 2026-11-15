"""What Yarvis says when you only say "Hey Yarvis".

A fixed, pre-synthesized phrase rather than a Claude reply: it plays instantly, costs
nothing and sends nothing anywhere. The first wake-up in a part of the day gets a proper
greeting ("Good evening, Victor."); later ones get something short ("Yes, Victor?").
"""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_PERIODS: Final = (
    (5, "Good morning{name}."),
    (12, "Good afternoon{name}."),
    (18, "Good evening{name}."),
)
_LATE: Final = "Still up{name}?"  # midnight to 5 a.m.
_SHORT: Final = (
    "Yes{name}?",
    "I'm listening.",
    "At your service.",
    "How can I help?",
    "Go ahead{name}.",
)


class Greeter:
    """Chooses the next greeting; never the same phrase twice in a row."""

    def __init__(
        self,
        user_name: str | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        choose: Callable[[Sequence[str]], str] = secrets.choice,
    ) -> None:
        self._suffix = f", {user_name}" if user_name else ""
        self._clock = clock
        self._choose = choose
        self._greeted_period: str | None = None
        self._last: str | None = None

    def next(self) -> str:
        """The greeting to say now."""
        period = self._period()
        if period != self._greeted_period:
            self._greeted_period = period
            phrase = period
        else:
            options = [p for p in self.short_phrases() if p != self._last]
            phrase = self._choose(options)
        self._last = phrase
        return phrase

    def phrases(self) -> list[str]:
        """Every phrase ``next`` may return in this part of the day (to synthesize early)."""
        return [self._period(), *self.short_phrases()]

    def short_phrases(self) -> list[str]:
        """The short acknowledgements."""
        return [phrase.format(name=self._suffix) for phrase in _SHORT]

    def _period(self) -> str:
        hour = self._clock().hour
        template = _LATE
        for start, greeting in _PERIODS:
            if hour >= start:
                template = greeting
        return template.format(name=self._suffix)
