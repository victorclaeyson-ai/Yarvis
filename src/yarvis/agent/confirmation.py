"""Did the user say yes? Decided by code from their own words, never by the model.

Only a short, clear answer counts. "Yes, but make it Chrome instead" is not a yes: it is a
new request, and the pending action is dropped.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Literal

from yarvis.speech.turn_taking import DEFAULT_WAKE_NAMES

if TYPE_CHECKING:
    from collections.abc import Iterable

Answer = Literal["yes", "no", "other"]

_MAX_WORDS: Final = 5
_YES: Final = frozenset(
    {
        "yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "confirm", "confirmed",
        "correct", "affirmative", "absolutely", "definitely", "please", "do", "it", "go",
        "ahead", "of", "course", "right", "that's", "fine", "alright", "sounds", "good",
    }
)  # fmt: skip
_YES_ANCHORS: Final = frozenset(
    {"yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "confirm", "confirmed",
     "correct", "affirmative", "absolutely", "definitely", "ahead", "course", "alright", "do"}
)  # fmt: skip
_NO: Final = frozenset(
    {"no", "nope", "nah", "don't", "dont", "cancel", "stop", "wait", "negative", "never",
     "mind", "not"}
)  # fmt: skip
_WORD: Final = re.compile(r"[a-z']+")


def classify_answer(text: str, *, names: Iterable[str] = DEFAULT_WAKE_NAMES) -> Answer:
    """Is ``text`` a clear yes, a clear no, or something else (a new request)?"""
    ignored = {name.lower() for name in names} | {"hey", "thanks", "thank", "you"}
    words = [word for word in _WORD.findall(text.lower()) if word not in ignored]
    if not words or len(words) > _MAX_WORDS:
        return "other"
    if any(word in _NO for word in words):
        return "no"  # "no", "don't", "yes... no wait": any hesitation is a no
    if all(word in _YES for word in words) and any(word in _YES_ANCHORS for word in words):
        return "yes"
    return "other"
