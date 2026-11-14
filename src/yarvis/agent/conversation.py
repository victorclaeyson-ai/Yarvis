"""Short-term conversation memory around an LLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from yarvis.llm.base import Message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from yarvis.llm.base import LLMProvider

DEFAULT_MAX_MESSAGES = 20
_EXCHANGE = 2  # a user message plus the assistant's reply


class Conversation:
    """Keeps the recent exchange and streams replies.

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
    ) -> None:
        if max_messages < _EXCHANGE or max_messages % _EXCHANGE:
            msg = "max_messages must be a positive even number"
            raise ValueError(msg)
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._max_messages = max_messages
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        """The remembered messages, oldest first."""
        return tuple(self._history)

    def reset(self) -> None:
        """Forget the conversation."""
        self._history.clear()

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """Stream the reply to ``user_text``.

        The exchange is remembered once the reply ends, including a partial reply if the
        stream is interrupted, so the model knows what the user actually heard.
        """
        messages = [*self._history, Message("user", user_text)]
        parts: list[str] = []
        try:
            async for delta in self._llm.stream(
                messages,
                system=self._system_prompt(),
                max_tokens=self._max_tokens,
            ):
                parts.append(delta)
                yield delta
        finally:
            if reply := "".join(parts).strip():
                self._remember(Message("user", user_text), Message("assistant", reply))

    def _remember(self, question: Message, answer: Message) -> None:
        self._history.extend((question, answer))
        overflow = len(self._history) - self._max_messages
        if overflow > 0:
            del self._history[:overflow]
