"""Memory tools: remember, list, search and forget facts about the user (ADR 0017).

Why it exists: so "what do you know about me?" and "forget that" work by voice, and so
Yarvis can keep a lasting fact the user mentions in passing (``memory.auto``).

What it must never do: change memory without the user hearing it. Remembering and
forgetting are tier 1 and shown on the dashboard; after the turn read outside content they
need a spoken "yes" (``asks_when_tainted``, decided by the gate in code, DESIGN.md T7).
Results are the user's own facts, not outside content.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final

from pydantic import Field

from yarvis.memory import MAX_FACT_CHARS, MemoryBookError
from yarvis.tools.base import Arguments, Tier, Tool, ToolError

if TYPE_CHECKING:
    from yarvis.memory import Fact, MemoryBook

ANNOUNCE: Final = "I'll remember that."

_REMEMBER_AUTO: Final = (
    "Keep a lasting fact about the user. Use it when they ask you to remember something, "
    "and also without being asked when they state a durable fact about themselves: names "
    "of the people in their life, preferences, routines, important dates. Never moods, "
    "passing states or one-off details. Whenever you keep something they did not ask you "
    f'to, say so in a few words ("{ANNOUNCE}"). Write one short sentence about the user '
    '("Has a sister called Lotte.").'
)
_REMEMBER_ASKED: Final = (
    "Keep a lasting fact about the user, only when the user asks you to remember it. Write "
    'one short sentence about the user ("Has a sister called Lotte.").'
)


def memory_ability(*, auto: bool) -> str:
    """What memory lets Yarvis do, in plain words for the system prompt."""
    if auto:
        return (
            "remember facts about the user (also without being asked, when they state a "
            f'lasting fact about themselves; then say "{ANNOUNCE}"), list, search and '
            "forget them"
        )
    return "remember facts about the user when the user asks, list, search and forget them"


class RememberArgs(Arguments):
    """Arguments of ``memory_remember``."""

    text: str = Field(min_length=1, max_length=MAX_FACT_CHARS, description="The fact.")


class ForgetArgs(Arguments):
    """Arguments of ``memory_forget``."""

    id: int = Field(ge=1, description="The fact's number, from the memory list.")


class SearchArgs(Arguments):
    """Arguments of ``memory_search``."""

    query: str = Field(min_length=1, max_length=100, description="Words to look for.")


class NoArgs(Arguments):
    """No arguments."""


async def _load(book: MemoryBook) -> tuple[Fact, ...]:
    """Every fact, read off the event loop (like saves, which may wait for a lock).

    Raises:
        ToolError: If the memory file is damaged (the message says how to fix it).
    """
    try:
        return await asyncio.to_thread(book.load)
    except MemoryBookError as exc:
        raise ToolError(str(exc)) from exc


def _line(fact: Fact) -> str:
    return f"{fact.id}: {fact.text}"


class MemoryList(Tool[NoArgs]):
    """Everything Yarvis remembers ("what do you know about me?")."""

    name = "memory_list"
    description = "List every fact you remember about the user, with its number."
    tier = Tier.READ
    Args = NoArgs

    def __init__(self, book: MemoryBook) -> None:
        self._book = book

    async def run(self, args: NoArgs) -> str:
        """List them."""
        facts = await _load(self._book)
        if not facts:
            return "Nothing remembered yet."
        return "\n".join(_line(fact) for fact in facts)

    def describe(self, args: NoArgs) -> str:
        """'check what you remember'."""
        return "check what you remember"


class MemorySearch(Tool[SearchArgs]):
    """Find facts that did not fit in the prompt."""

    name = "memory_search"
    description = (
        "Search the facts you remember about the user, including older ones not shown in "
        "your instructions."
    )
    tier = Tier.READ
    Args = SearchArgs

    def __init__(self, book: MemoryBook) -> None:
        self._book = book

    async def run(self, args: SearchArgs) -> str:
        """Search."""
        await _load(self._book)  # a damaged memory is reported, not taken for empty
        found = self._book.search(args.query)
        if not found:
            return f"Nothing remembered matches {args.query!r}."
        return "\n".join(_line(fact) for fact in found)

    def describe(self, args: SearchArgs) -> str:
        """'search your memory for "Lotte"'."""
        return f'search your memory for "{args.query}"'


class MemoryRemember(Tool[RememberArgs]):
    """Keep a fact."""

    name = "memory_remember"
    tier = Tier.REVERSIBLE
    Args = RememberArgs
    asks_when_tainted = True

    def __init__(self, book: MemoryBook, *, auto: bool) -> None:
        self._book = book
        self.description = _REMEMBER_AUTO if auto else _REMEMBER_ASKED

    async def run(self, args: RememberArgs) -> str:
        """Keep it."""
        try:
            fact = await asyncio.to_thread(self._book.add, args.text, source="voice")
        except MemoryBookError as exc:
            raise ToolError(str(exc)) from exc
        return f"Remembered as fact {fact.id}."

    def describe(self, args: RememberArgs) -> str:
        """'remember that "Has a cat."' (read back before a confirmation)."""
        return f'remember that "{args.text}"'


class MemoryForget(Tool[ForgetArgs]):
    """Forget a fact."""

    name = "memory_forget"
    description = 'Forget one fact about the user, by its number ("forget that").'
    tier = Tier.REVERSIBLE
    Args = ForgetArgs
    asks_when_tainted = True

    def __init__(self, book: MemoryBook) -> None:
        self._book = book

    async def prepare(self, args: ForgetArgs) -> ForgetArgs:
        """Check the fact exists, so the question names it."""
        if self._book.get(args.id) is None:
            msg = f"There is no fact {args.id}."
            raise ToolError(msg)
        return args

    async def run(self, args: ForgetArgs) -> str:
        """Forget it."""
        try:
            fact = await asyncio.to_thread(self._book.forget, args.id)
        except MemoryBookError as exc:
            raise ToolError(str(exc)) from exc
        return f"Forgot fact {fact.id}."

    def describe(self, args: ForgetArgs) -> str:
        """'forget "Prefers tea."' (read back before a confirmation)."""
        fact = self._book.get(args.id)
        return f'forget "{fact.text}"' if fact is not None else f"forget fact {args.id}"


def memory_tools(book: MemoryBook, *, auto: bool) -> list[Tool[Any]]:
    """Every memory tool, sharing ``book``; ``auto`` lets Yarvis keep facts unasked."""
    return [
        MemoryList(book),
        MemorySearch(book),
        MemoryRemember(book, auto=auto),
        MemoryForget(book),
    ]
