"""Smart model switching: hard questions go to a bigger model, within a daily budget.

Why it exists (ADR 0019): Haiku 5.5 gets the first word out fast, but Opus 5.5 answers
hard questions better. A question reaches the bigger model in two ways:

- the user asks ("think hard: …", "take your time and …"), recognised here by code;
- Haiku hands it over by calling the ``think_harder`` tool (when ``llm.switching`` is
  "auto").

While the bigger model works, code says a short phrase that varies ("One moment."), so
the user knows they were heard.

What it must never do: let the model decide the budget (it is checked in code before every
switch); spend on a guess made at a pause (the conversation waits for a real request);
or change what the permission gate allows (the same toolbox and taint apply to any model).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from yarvis.llm.base import ToolSpec
from yarvis.usage import local_today

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    from yarvis.llm.base import LLMProvider

THINK_HARDER: Final = "think_harder"
THINK_HARDER_SPEC: Final = ToolSpec(
    THINK_HARDER,
    (
        "Hand the user's question to a bigger, slower model that reasons more carefully. "
        "Call it, and say nothing else, when the question needs careful reasoning: "
        "calculations with several steps, planning, comparing options, explanations in "
        "several steps. Don't call it for simple questions, small talk, or actions your "
        "other tools can do."
    ),
    {"type": "object", "properties": {}, "additionalProperties": False},
)
FILLERS: Final = (
    "One moment.",
    "Let me see.",
    "Hmm, let me work that out.",
    "Give me a second.",
    "Good question, one moment.",
)
BIG_MAX_TOKENS: Final = 4096
"""The bigger model thinks before it answers: the voice's 1,024 tokens could cut it off."""
BUDGET_NOTE: Final = "I've used today's budget for thinking harder, so here's my quick answer."
DECLINED: Final = (
    "The bigger model isn't available now: today's budget for it is used up. Answer the "
    "question yourself, briefly."
)

ANSWER_YOURSELF: Final = "You are the bigger model: answer the question yourself."
NOT_ALONE: Final = (
    "Not handed over: call think_harder on its own, without other actions in the same reply."
)

_THINK_HARD: Final = re.compile(
    r"^(?:(?:hey\s+\w+[,!.]?|\w+[,!.])\s+)?(?:please\s+)?"  # "Yarvis, …" / "Hey Yarvis …"
    r"(?:think\s+(?:hard|carefully|it\s+through)|take\s+your\s+time)\b",
    re.IGNORECASE,
)


def is_think_hard(text: str) -> bool:
    """Whether the user asked for the bigger model ("think hard: …", "take your time …").

    Only at the start of their own request: a sentence that merely contains the words
    ("I think hard work pays off") is an ordinary question.
    """
    return bool(_THINK_HARD.match(" ".join(text.split())))


class Fillers:
    """The short phrase said while the bigger model works: varied, never twice in a row."""

    def __init__(self, rng: random.Random | None = None, phrases: Sequence[str] = FILLERS) -> None:
        self._rng = rng or random.Random()  # noqa: S311 - picking a phrase, not a secret
        self._phrases = tuple(phrases)
        self._last: str | None = None

    def next(self) -> str:
        """A phrase other than the last one."""
        choices = [p for p in self._phrases if p != self._last] or list(self._phrases)
        self._last = self._rng.choice(choices)
        return self._last


@dataclass(slots=True)
class Switching:
    """How a conversation reaches the bigger model."""

    big: LLMProvider
    """The bigger model (``llm.big_model``)."""
    auto: bool
    """Haiku may hand questions over itself (``llm.switching = "auto"``)."""
    within_budget: Callable[[], bool]
    """Whether today's spend on the bigger model is still under ``llm.big_model_daily_usd``."""
    fillers: Fillers = field(default_factory=Fillers)
    today: Callable[[], date] = local_today
    max_tokens: int = BIG_MAX_TOKENS
    _noted_on: date | None = None

    def budget_note(self) -> str | None:
        """What to say when the user asked but the budget is used up: once a day."""
        today = self.today()
        if self._noted_on == today:
            return None
        self._noted_on = today
        return BUDGET_NOTE
