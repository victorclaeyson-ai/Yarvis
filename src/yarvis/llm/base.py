"""Provider-independent types for language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One message in a conversation."""

    role: Role
    text: str


class LLMError(Exception):
    """A language-model request failed. The message is safe to show to the user."""


class LLMProvider(Protocol):
    """Streams a reply from a language model."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield the reply as text fragments, as soon as they are generated.

        Raises:
            LLMError: On any failure, with a user-presentable message.
        """
        ...
