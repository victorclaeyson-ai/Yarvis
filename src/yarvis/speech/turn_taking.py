"""Reading the transcript: does a sentence sound finished, and was the wake word in it?

Does it sound finished? Used to decide how long to wait in a pause.

Silence alone cannot tell "What's the weather tomorrow?" (done) from "So I was thinking
about the…" (still thinking). By the time a pause is a few hundred milliseconds old, the
speculative transcript is already available, so we can look at *how* the sentence ends and
wait longer when it clearly is not finished.
"""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

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


# ----------------------------------------------------------------------------- wake word

WAKE_PHRASE: Final = "Hey Yarvis"
"""What to say (the bundled "hey jarvis" model hears "hey yarvis" just as well)."""
DEFAULT_WAKE_NAMES: Final = ("Yarvis", "Jarvis")
_GREETINGS: Final = r"hey|hi|hello|okay|ok|oh|yo"
# The wake phrase may only be cut from the start of a woken utterance, not from deep inside
# a request that merely mentions the name ("who was Jarvis in Iron Man?").
_WAKE_SEARCH_CHARS: Final = 40
_MAX_REPEATS: Final = 3


@cache
def _wake_patterns(names: tuple[str, ...]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    alternatives = "|".join(re.escape(name) for name in names)
    greeting = rf"\b(?:{_GREETINGS})\b[\s,.!]*\b(?:{alternatives})\b[\s,.!?;:]*"
    # A bare name only counts as an address when punctuation follows it ("Yarvis, ...");
    # otherwise it is part of the sentence ("Yarvis is a cool name").
    address = rf"\b(?:{alternatives})(?:[,.!?;:]+\s*|\s*$)"
    flags = re.IGNORECASE
    return re.compile(rf"(?:{greeting}|{address})", flags), re.compile(
        rf"^\s*(?:{greeting}|{address})", flags
    )


def strip_wake_phrase(
    text: str, *, anywhere: bool = False, names: Iterable[str] = DEFAULT_WAKE_NAMES
) -> str:
    """Remove "Hey Yarvis" (or "Yarvis,") so only the request itself reaches Claude.

    With ``anywhere`` (an utterance that started with the wake word's pre-roll), everything
    up to and including the first wake phrase near the start is dropped: the audio before it
    may hold the end of another conversation, or Yarvis's own voice on speakers.
    """
    search, leading = _wake_patterns(tuple(sorted({name.strip() for name in names if name})))
    text = text.strip()
    if anywhere and (match := search.search(text)) and match.start() <= _WAKE_SEARCH_CHARS:
        text = text[match.end() :]
    # "Hey Yarvis. Hey Yarvis, what's up?": people repeat themselves when unsure it heard.
    for _ in range(_MAX_REPEATS):
        match = leading.match(text)
        if match is None:
            break
        text = text[match.end() :]
    return text.strip()


# ----------------------------------------------------------------------------- goodbyes

# A short request made only of these words closes the conversation ("Thanks, Yarvis",
# "That's all", "Never mind", "Good night"); at least one of _CLOSERS must be present, so
# a lone "okay" or "no" does not.
_CLOSING_WORDS: Final = frozenset(
    {
        "thanks", "thank", "you", "so", "much", "very", "a", "lot", "cheers", "ta",
        "that's", "thats", "that", "that'll", "will", "is", "be", "all", "it", "for", "now",
        "never", "mind", "nevermind", "nothing", "else", "no", "ok", "okay", "alright",
        "great", "perfect", "cool", "good", "night", "bye", "goodbye", "see", "later",
        "stop", "done", "i'm", "im", "we're", "were", "oh", "and",
    }
)  # fmt: skip
_CLOSERS: Final = frozenset(
    {"thanks", "thank", "cheers", "all", "mind", "nevermind", "nothing", "bye", "goodbye",
     "night", "later", "stop", "done"}
)  # fmt: skip
_MAX_CLOSING_WORDS: Final = 8
_WORD: Final = re.compile(r"[a-z']+")


def ends_conversation(text: str, *, names: Iterable[str] = DEFAULT_WAKE_NAMES) -> bool:
    """Whether the user is wrapping up ("Thanks!", "That's all", "Bye, Yarvis").

    Yarvis still answers, but then stops listening until the wake word.
    """
    ignored = {name.lower() for name in names}
    words = [word for word in _WORD.findall(text.lower()) if word not in ignored]
    return (
        0 < len(words) <= _MAX_CLOSING_WORDS
        and all(word in _CLOSING_WORDS for word in words)
        and any(word in _CLOSERS for word in words)
    )


# ----------------------------------------------------------------------------- who is it for?

# While a conversation stays open, people also talk to each other. Only sentences that are
# clearly for Yarvis become requests: ones that use its name, or that are phrased like the
# commands it understands. Everything else is dropped on the spot (never sent anywhere).
# The patterns are deliberately narrow: "play some jazz" is for Yarvis, "play with me" or
# "go to bed" are not.
_LEAD_IN: Final = re.compile(
    r"^(?:(?:please|pls|hey|hi|ok|okay|oh|and|also|now|so|just|um|uh|right|alright|then|"
    r"can you|could you|would you|will you|can u|could u)\s+)+"
)
_SITES: Final = (
    r"youtube|spotify|gmail|google|netflix|github|facebook|instagram|whatsapp|twitter|"
    r"reddit|wikipedia|amazon|linkedin|outlook|maps|calendar|chatgpt|claude|brave|safari"
)
_COMMANDS: Final = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(open|launch|close|quit) (?!(the|a|an|your|my|his|her|it|up|this|that)\b)\S",
        r"^open (a )?new (tab|window)\b",
        r"^play$|^play (?!(fair|nice|with|outside|along|together|ball|games?|cards|"
        r"football|soccer|tennis|fifa|minecraft)\b)\S",
        r"^(pause|resume|unpause|skip|shuffle|unshuffle|mute|unmute)\b",
        r"^(next|previous|last)( (song|track|one|video|tab))?$",
        r"^stop (the )?(music|song|video|playing|timer|alarm)\b",
        r"^(louder|quieter|softer)\b",
        r"^(volume|sound) (up|down|to|at)\b",
        r"^turn (it |the (music|volume|sound|song|video) )?(up|down|on|off)\b",
        r"^turn (up|down|on|off) the (music|volume|sound|song|video)\b",
        r"^set (a |an |the |my )?\S*\s*(timer|alarm|reminder|volume)\b",
        r"^(remind me|cancel (the |my |that )?(timer|alarm|reminder))\b",
        r"^(search|google|look up) (?!me\b)\S",
        r"^go (back|forward)\b",
        rf"^go to (the )?(\S+\.(com|be|org|net|io|eu)|{_SITES})\b",
        r"^(switch|change) to\b",
        r"^(close|reload|refresh) (this|the|that) (tab|page|window)\b",
        r"^(type|press|click|scroll)\b",
        r"^(what'?s|what is) (playing|this song|on)\b",
        r"^(what|which) song\b",
        r"^(i want to|i'd like to|let me) (hear|listen to)\b",
    )
)
# Right after Yarvis answered, "what about tomorrow?" continues the conversation.
_CONTINUATION: Final = re.compile(
    r"^(and )?(what|how) about\b|^and (what|how|where|when|who|why)\b"
)
_NOT_WORDS: Final = re.compile(r"[^a-z0-9'.\s]+")
_LOOSE_DOT: Final = re.compile(r"\.(?!\w)")  # a full stop, not the one in "kuleuven.be"


def meant_for_assistant(
    text: str, *, names: Iterable[str] = DEFAULT_WAKE_NAMES, just_answered: bool = False
) -> bool:
    """Whether an overheard sentence is a request for Yarvis.

    True if it uses Yarvis's name, or is phrased like a command it understands ("open
    Spotify", "can you play some jazz", "turn it up"). With ``just_answered`` (Yarvis spoke
    a moment ago), "and what about tomorrow?" counts as well.
    """
    lowered = text.lower()
    if any(re.search(rf"\b{re.escape(name.lower())}\b", lowered) for name in names if name):
        return True
    words = " ".join(_LOOSE_DOT.sub(" ", _NOT_WORDS.sub(" ", lowered)).split())
    words = _LEAD_IN.sub("", words)
    if any(pattern.search(words) for pattern in _COMMANDS):
        return True
    return just_answered and bool(_CONTINUATION.search(words))
