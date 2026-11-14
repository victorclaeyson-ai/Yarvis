"""Turn a stream of LLM text into sentences that can be spoken one at a time.

Speaking can start as soon as the first sentence is complete, while the model is still
writing the rest. That is the single biggest latency win in the pipeline.

Rules:

* The **first** sentence is emitted as soon as it ends, however short, so audio starts fast.
  Better still, its first *clause* is emitted at a comma, colon or dash once it is long
  enough to sound natural on its own ("Grey skies and drizzle, | as usual for Leuven.").
* Later sentences are merged until they reach ``min_chars``: very short TTS chunks sound
  choppy, and by then the voice is already busy speaking the first one.
* A newline always ends a sentence (lists, paragraphs).
* Abbreviations ("Dr.", "e.g.") and initials ("J. R.") do not end a sentence.
* Text with no punctuation at all is split at a clause or word boundary after ``max_chars``.
"""

from __future__ import annotations

import re
from typing import Final

_TERMINATORS: Final = ".!?…"
_CLOSERS: Final = "\"')]”’"  # noqa: RUF001 - typographic quotes are intended
_ABBREVIATIONS: Final = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "approx",
        "e.g", "i.e", "fig", "inc", "ltd", "co", "mt", "ft", "dept", "est",
    }
)  # fmt: skip


class SentenceSplitter:
    """Incrementally split streamed text into sentences."""

    def __init__(
        self,
        *,
        min_chars: int = 40,
        max_chars: int = 220,
        first_clause_chars: int | None = 24,
    ) -> None:
        if not 0 < min_chars < max_chars:
            msg = "require 0 < min_chars < max_chars"
            raise ValueError(msg)
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._first_clause_chars = first_clause_chars
        self._buffer = ""
        self._pending = ""  # complete but too-short sentences waiting to be merged
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        """Add streamed text; return every sentence that is now ready to speak."""
        self._buffer += delta
        ready: list[str] = []
        while (cut := self._next_cut()) is not None:
            piece = self._buffer[:cut]
            self._buffer = self._buffer[cut:]
            if sentence := self._accept(piece):
                ready.append(sentence)
        return ready

    def flush(self) -> str | None:
        """Return whatever text is left at the end of the stream."""
        rest = _join(self._pending, self._buffer)
        self._pending = ""
        self._buffer = ""
        if not rest:
            return None
        self._emitted += 1
        return rest

    def _accept(self, piece: str) -> str | None:
        candidate = _join(self._pending, piece)
        if not candidate:
            return None
        if self._emitted == 0 or len(candidate) >= self._min_chars:
            self._pending = ""
            self._emitted += 1
            return candidate
        self._pending = candidate
        return None

    def _next_cut(self) -> int | None:
        text = self._buffer
        sentence = self._sentence_cut(text)
        if self._emitted == 0 and self._first_clause_chars is not None:
            clause = _clause_break(text, self._first_clause_chars)
            if clause is not None and (sentence is None or clause < sentence):
                return clause
        if sentence is not None:
            return sentence
        if len(text) > self._max_chars:
            return _soft_break(text, self._max_chars)
        return None

    def _sentence_cut(self, text: str) -> int | None:
        for index, char in enumerate(text):
            if char == "\n":
                return index + 1
            if char in _TERMINATORS:
                end = index + 1
                while end < len(text) and (text[end] in _TERMINATORS or text[end] in _CLOSERS):
                    end += 1
                if end >= len(text):
                    return None  # need the next character to decide
                if text[end].isspace() and not self._is_abbreviation(text, index):
                    return end
        return None

    @staticmethod
    def _is_abbreviation(text: str, dot_index: int) -> bool:
        if text[dot_index] != ".":
            return False
        start = dot_index
        while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
            start -= 1
        word = text[start:dot_index]
        if len(word) == 1 and word.isupper():
            return True  # an initial, as in "J. R. Tolkien"
        return word.lower() in _ABBREVIATIONS


_CLAUSE_BREAK: Final = re.compile(r"(?:,|;|:| -| —| –)\s")  # noqa: RUF001 - dashes are intended


def _clause_break(text: str, min_chars: int) -> int | None:
    """End of the first clause that is at least ``min_chars`` long, if any."""
    for match in _CLAUSE_BREAK.finditer(text):
        if match.start() >= min_chars and not text[: match.start()].rstrip()[-1:].isdigit():
            return match.end()
    return None


def _soft_break(text: str, limit: int) -> int:
    window = text[:limit]
    for separator in (", ", "; ", ": ", " - ", " "):
        position = window.rfind(separator)
        if position > limit // 3:
            return position + len(separator)
    return limit


def _join(first: str, second: str) -> str:
    return " ".join(part for part in (first.strip(), second.strip()) if part)


_MARKDOWN_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"```.*?```", re.DOTALL), " "),  # code blocks are not speakable
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"https?://\S+"), "a link"),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),
    (re.compile(r"(?<!\w)[*_](.+?)[*_](?!\w)"), r"\1"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),
    (re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE), ""),
    (re.compile(r"\s+"), " "),
)


def to_speakable(text: str) -> str:
    """Strip markdown and other things a voice should not read out literally."""
    for pattern, replacement in _MARKDOWN_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


class SpeechFilter:
    """Turns successive sentences into speakable text, skipping fenced code blocks.

    Code blocks span several sentences (the splitter breaks on newlines), so whether a
    sentence is inside one depends on the sentences before it.
    """

    def __init__(self) -> None:
        self._in_code = False

    def __call__(self, sentence: str) -> str:
        """Return what should be spoken for ``sentence`` (possibly nothing)."""
        fences = sentence.count("```")
        if self._in_code or fences:
            if fences % 2:
                self._in_code = not self._in_code
            return ""
        return to_speakable(sentence)
