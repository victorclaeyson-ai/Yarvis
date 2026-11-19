"""Memory: facts Yarvis learned about the user, kept until the user forgets them (ADR 0017).

Why it exists: the profile (ADR 0014) is what the user wrote; memory is what Yarvis picked
up in conversation ("my sister is called Lotte") or was told to keep. Each fact is one
short line in ``memory.json`` in the private data folder, shown and changed by voice and
with ``yarvis memory``.

What it must never do: keep transcripts or anything but these facts (CLAUDE.md rule 5);
keep a secret (rule 3); lose the file halfway through a save; or let two processes (the
app and the CLI) overwrite each other or hand out the same number. Facts are data: the
permission gate decides in code, and nothing written here can change a tier.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from yarvis.paths import LOCK_TIMEOUT_S, private_lock, replace_private_file
from yarvis.security.redact import Redactor, default_redactor

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from yarvis.paths import AppPaths

MEMORY_FILE: Final = "memory.json"
MAX_FACT_CHARS: Final = 200
MAX_FACTS = 500
"""Room for years of facts; the prompt shows the newest, ``memory_search`` finds the rest."""
MAX_FILE_BYTES = 1_000_000

Source = Literal["voice", "cli"]

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+")
_MIN_WORD = 2
_STOPWORDS: Final = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "who",
        "with",
        "you",
        "your",
    ]
)
_HIDDEN: Final = frozenset({"Cc", "Cf", "Co", "Cn", "Cs", "Zl", "Zp"})
"""Control, format (zero-width, bidi, tag), private-use and unassigned characters: unseen
when a fact is read back or listed, yet read by the model."""
_SECRET_WORDS: Final = re.compile(
    r"\b(password|passwd|passcode|pin|pin code|pincode|iban|card number|cvv|cvc|"
    r"security code|wachtwoord|recovery code|seed phrase)\b",
    re.IGNORECASE,
)
_LONG_NUMBER: Final = re.compile(r"\d(?:[ -]?\d){11,}")
"""Twelve or more digits (a card or account number), spaces and dashes allowed."""


class MemoryBookError(Exception):
    """A memory change did not happen. The message says why, for the user."""


class Fact(BaseModel):
    """One thing Yarvis knows about the user."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(ge=1)
    """Never reused, so "forget fact 3" cannot hit a newer fact."""
    text: str = Field(min_length=1, max_length=MAX_FACT_CHARS, pattern=r"^[^\r\n]+$")
    added: AwareDatetime
    source: Source

    @field_validator("text")
    @classmethod
    def _visible(cls, value: str) -> str:
        """Also for a file edited by hand: what is kept is exactly what can be seen."""
        if _has_hidden(value):
            msg = "contains invisible characters"
            raise ValueError(msg)
        return value


class _File(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = 1
    next_id: int = Field(default=1, ge=1)
    facts: tuple[Fact, ...] = ()


class MemoryBook:
    """The facts in ``memory.json``, read again only after the file changed."""

    def __init__(
        self,
        path: Path,
        *,
        redactor: Redactor = default_redactor,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        lock_timeout_s: float = LOCK_TIMEOUT_S,
    ) -> None:
        self.path = path
        self._redactor = redactor
        self._now = now
        self._lock_timeout_s = lock_timeout_s
        self._seen: tuple[int, int, int] | None = None
        self._cached: tuple[Fact, ...] = ()

    @classmethod
    def of(cls, paths: AppPaths) -> MemoryBook:
        """Where this installation keeps it."""
        return cls(paths.data_dir / MEMORY_FILE)

    # ------------------------------------------------------------------ reading

    def facts(self) -> tuple[Fact, ...]:
        """Every fact, oldest first; empty (with a warning) if the file cannot be used."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._seen, self._cached = None, ()
            return ()
        except OSError as exc:
            logger.warning("Memory is not used: cannot read it (%s)", type(exc).__name__)
            self._seen, self._cached = None, ()
            return ()
        # A save replaces the file: its inode changes even when size and time look the same.
        seen = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if seen != self._seen:
            self._seen = seen
            try:
                self._cached = self.load()
            except MemoryBookError as exc:
                logger.warning("Memory is not used: %s", exc)
                self._cached = ()
        return self._cached

    def load(self) -> tuple[Fact, ...]:
        """Every fact, read now.

        Raises:
            MemoryBookError: If the file is damaged, too large or unreadable.
        """
        return self._read().facts

    def get(self, fact_id: int) -> Fact | None:
        """The fact with this number, if there is one."""
        return next((fact for fact in self.facts() if fact.id == fact_id), None)

    def search(self, query: str, *, limit: int = 10) -> list[Fact]:
        """Facts sharing words with ``query``: most shared words first, then newest."""
        wanted = _words(query) - _STOPWORDS
        scored = [(len(wanted & _words(fact.text)), fact) for fact in self.facts()]
        found = [(score, fact) for score, fact in scored if score]
        found.sort(key=lambda pair: (pair[0], pair[1].id), reverse=True)
        return [fact for _, fact in found[:limit]]

    def export(self) -> str:
        """Every fact as JSON (``yarvis memory export``)."""
        return self._read().model_dump_json(indent=2)

    # ------------------------------------------------------------------ changing

    def add(self, text: str, *, source: Source) -> Fact:
        """Keep a new fact and return it.

        Raises:
            MemoryBookError: If it is empty, too long, a secret or already known, if memory
                is full, or if it cannot be saved.
        """
        multiline = "\n" in text or "\r" in text
        text = " ".join(unicodedata.normalize("NFKC", text).split())
        if not text or len(text) > MAX_FACT_CHARS or multiline:
            msg = f"A fact is one line of at most {MAX_FACT_CHARS} characters."
            raise MemoryBookError(msg)
        if _has_hidden(text):
            msg = "That fact contains invisible characters, so it isn't kept."
            raise MemoryBookError(msg)
        if (
            self._redactor.contains_secret(text)
            or _SECRET_WORDS.search(text)
            or _LONG_NUMBER.search(text)
        ):
            msg = (
                "That looks like a password or key: secrets belong in the keychain, not in memory."
            )
            raise MemoryBookError(msg)
        with self._locked():
            record = self._read()
            key = _key(text)
            if same := next((f for f in record.facts if _key(f.text) == key), None):
                msg = f"Already remembered as fact {same.id}."
                raise MemoryBookError(msg)
            if len(record.facts) >= MAX_FACTS:
                msg = f"Memory is full ({MAX_FACTS} facts): forget some first."
                raise MemoryBookError(msg)
            fact = Fact(id=record.next_id, text=text, added=self._now(), source=source)
            self._write(_File(next_id=fact.id + 1, facts=(*record.facts, fact)))
        return fact

    def forget(self, fact_id: int) -> Fact:
        """Forget one fact and return it.

        Raises:
            MemoryBookError: If there is no such fact, or it cannot be saved.
        """
        with self._locked():
            record = self._read()
            fact = next((f for f in record.facts if f.id == fact_id), None)
            if fact is None:
                msg = f"There is no fact {fact_id}."
                raise MemoryBookError(msg)
            kept = tuple(f for f in record.facts if f.id != fact_id)
            self._write(_File(next_id=record.next_id, facts=kept))
        return fact

    def clear(self) -> int:
        """Forget every fact; returns how many there were.

        The numbering goes on where it was, so an old "forget fact 3" never hits a new fact.
        A damaged file is removed: clearing is how the user starts again.

        Raises:
            MemoryBookError: If it cannot be saved or removed.
        """
        with self._locked():
            try:
                record = self._read()
            except MemoryBookError:
                self._remove()
                count = 0
            else:
                count = len(record.facts)
                if self.path.exists():
                    self._write(_File(next_id=record.next_id))
        self._seen, self._cached = None, ()
        return count

    # ------------------------------------------------------------------ the file

    def _read(self) -> _File:
        try:
            if self.path.stat().st_size > MAX_FILE_BYTES:
                raise MemoryBookError(self._damaged("it is too large to be Yarvis's"))
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _File()
        except (OSError, UnicodeDecodeError) as exc:
            raise MemoryBookError(self._damaged(f"unreadable ({type(exc).__name__})")) from exc
        try:
            return _File.model_validate_json(raw)
        except ValidationError as exc:
            raise MemoryBookError(self._damaged("not valid memory")) from exc

    def _damaged(self, why: str) -> str:
        return (
            f"{self.path.name} is damaged: {why}. Fix it, or start again with: yarvis memory clear"
        )

    def _write(self, record: _File) -> None:
        try:
            replace_private_file(self.path, record.model_dump_json(indent=2) + "\n")
        except OSError as exc:
            logger.warning("Could not save memory: %s", type(exc).__name__)
            msg = "Couldn't save the memory: try again."
            raise MemoryBookError(msg) from exc

    def _remove(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:  # on Windows: another program has it open
            logger.warning("Could not remove memory: %s", type(exc).__name__)
            msg = "Couldn't clear the memory (is memory.json open somewhere?): try again."
            raise MemoryBookError(msg) from exc

    @contextmanager
    def _locked(self) -> Generator[None]:
        """Hold the lock for one read-change-write, so two processes never interleave."""
        lock = self.path.with_name(self.path.name + ".lock")
        with ExitStack() as stack:
            try:
                stack.enter_context(private_lock(lock, timeout_s=self._lock_timeout_s))
            except TimeoutError as exc:
                msg = "Memory is busy (another Yarvis is saving): try again."
                raise MemoryBookError(msg) from exc
            except OSError as exc:
                logger.warning("Could not lock memory: %s", type(exc).__name__)
                msg = "Couldn't save the memory: try again."
                raise MemoryBookError(msg) from exc
            yield


def _key(text: str) -> str:
    """What makes two facts the same: their words, whatever the case or full stop."""
    return " ".join(text.casefold().split()).rstrip(".!")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if len(w) >= _MIN_WORD}


def _has_hidden(text: str) -> bool:
    return any(unicodedata.category(char) in _HIDDEN for char in text)
