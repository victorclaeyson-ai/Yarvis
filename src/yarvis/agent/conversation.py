"""Short-term conversation memory around an LLM provider.

Replies are produced as ``Draft`` objects. A draft starts generating immediately, but it
only becomes part of the conversation once someone consumes it. That lets the pipeline
start asking Claude *speculatively* — while the user may still be talking — and throw the
draft away if they continue, without the abandoned exchange ever entering the history.

Every exchange that is remembered passes through ``Conversation.remember``: the single
place where long-term memory will hook in later.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from typing import TYPE_CHECKING

from yarvis.llm.base import Message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from yarvis.llm.base import LLMProvider

DEFAULT_MAX_MESSAGES = 20
_EXCHANGE = 2  # a user message plus the assistant's reply


class Draft:
    """A reply that is being generated, possibly before we know it will be used."""

    def __init__(self, conversation: Conversation, user_text: str) -> None:
        self.user_text = user_text
        self.first_token_at: float | None = None
        self._conversation = conversation
        self._queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        self._consumed = False
        self._task = asyncio.create_task(self._generate(conversation.messages_for(user_text)))

    async def _generate(self, messages: list[Message]) -> None:
        try:
            async for delta in self._conversation.llm.stream(
                messages,
                system=self._conversation.system_prompt(),
                max_tokens=self._conversation.max_tokens,
            ):
                if self.first_token_at is None:
                    self.first_token_at = self._conversation.clock()
                self._queue.put_nowait(delta)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - handed to the consumer, who re-raises it
            self._queue.put_nowait(exc)
        finally:
            self._queue.put_nowait(None)

    async def deltas(self) -> AsyncGenerator[str]:
        """Consume the reply. From this moment the exchange will be remembered.

        Whatever was consumed is remembered when the stream ends, including a partial reply
        if it is interrupted, so the model knows what the user actually heard.
        """
        if self._consumed:
            msg = "a draft can only be consumed once"
            raise RuntimeError(msg)
        self._consumed = True
        heard: list[str] = []
        try:
            while (item := await self._queue.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                heard.append(item)
                yield item
        finally:
            if not self._task.done():
                self._task.cancel()
            if reply := "".join(heard).strip():
                self._conversation.remember(self.user_text, reply)

    def discard(self) -> None:
        """Abandon the draft: stop generating and remember nothing."""
        if not self._task.done():
            self._task.cancel()


class Conversation:
    """Keeps the recent exchange and creates reply drafts.

    History lives only in memory and is trimmed to ``max_messages`` (whole exchanges), so
    the context stays small and fast. Nothing is written to disk.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        system_prompt: Callable[[], str],
        max_tokens: int,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_messages < _EXCHANGE or max_messages % _EXCHANGE:
            msg = "max_messages must be a positive even number"
            raise ValueError(msg)
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.clock = clock
        self._max_messages = max_messages
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        """The remembered messages, oldest first."""
        return tuple(self._history)

    def reset(self) -> None:
        """Forget the conversation."""
        self._history.clear()

    def messages_for(self, user_text: str) -> list[Message]:
        """The messages to send for a new user message."""
        return [*self._history, Message("user", user_text)]

    def draft(self, user_text: str) -> Draft:
        """Start generating a reply now; it is remembered only once consumed."""
        return Draft(self, user_text)

    async def respond(self, user_text: str) -> AsyncGenerator[str]:
        """Generate and consume a reply in one go."""
        async with aclosing(self.draft(user_text).deltas()) as deltas:
            async for delta in deltas:
                yield delta

    def retract(self, user_text: str) -> bool:
        """Forget the last exchange if it answered ``user_text``; True if one was removed.

        Used when a reply was cut off before the user heard any of it: the question is then
        asked again together with what the user added, and must not appear twice.
        """
        if len(self._history) >= _EXCHANGE and self._history[-_EXCHANGE].text == user_text:
            del self._history[-_EXCHANGE:]
            return True
        return False

    def remember(self, user_text: str, reply: str) -> None:
        """Record a finished exchange (the hook point for long-term memory)."""
        self._history.extend((Message("user", user_text), Message("assistant", reply)))
        overflow = len(self._history) - self._max_messages
        if overflow > 0:
            del self._history[:overflow]
