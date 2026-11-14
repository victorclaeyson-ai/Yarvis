"""Does a sentence sound finished? Used to decide how long to wait in a pause.

Silence alone cannot tell "What's the weather tomorrow?" (done) from "So I was thinking
about the…" (still thinking). By the time a pause is a few hundred milliseconds old, the
speculative transcript is already available, so we can look at *how* the sentence ends and
wait longer when it clearly is not finished.
"""

from __future__ import annotations

import re
from typing import Final

# Words that almost never end a finished English sentence: conjunctions, articles,
# prepositions, possessives, auxiliaries and fillers.
_UNFINISHED_ENDINGS: Final = frozenset(
    {
        "and", "but", "or", "so", "because", "although", "though", "while", "if", "unless",
        "then", "than", "that", "which", "who", "whose", "where", "when",
        "the", "a", "an", "this", "these", "those", "my", "your", "his", "her", "our", "their",
        "its", "some", "any", "every", "each",
        "to", "of", "for", "with", "about", "from", "in", "on", "at", "by", "into", "onto",
        "like", "as", "between", "without", "after", "before", "until",
        "is", "are", "was", "were", "be", "been", "am", "will", "would", "could", "should",
        "can", "do", "does", "did", "have", "has", "had", "i", "i'm", "it's", "let's",
        "um", "uh", "uhm", "erm", "er", "hmm", "mm", "well", "also", "maybe", "just",
    }
)  # fmt: skip

_TRAILING_ELLIPSIS: Final = re.compile(r"(\.\.\.|…)\s*$")
_LAST_WORD: Final = re.compile(r"([A-Za-z']+)[^A-Za-z']*$")


def sounds_finished(text: str) -> bool:
    """Heuristic: False only when there is a clear sign the user is mid-sentence.

    Conservative on purpose: when in doubt we answer, because waiting too long on every
    turn would make the assistant feel slow. Clear signs are a trailing "…", a trailing
    comma or similar, or a last word that almost never ends a sentence ("and", "the", "um").
    """
    text = text.strip()
    if not text:
        return True  # nothing intelligible: no reason to wait longer
    if _TRAILING_ELLIPSIS.search(text) or text[-1] in ",;:-–—":  # noqa: RUF001
        return False
    if text.endswith("?"):
        return True  # questions often end in a function word ("So what?", "With who?")
    match = _LAST_WORD.search(text)
    return not (match and match.group(1).lower() in _UNFINISHED_ENDINGS)
