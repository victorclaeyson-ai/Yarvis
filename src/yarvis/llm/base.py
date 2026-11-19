"""Provider-independent types for language models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool as the model sees it: name, description and JSON schema of its input."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asks to run a tool. ``arguments`` is unvalidated model output."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class WebSearch:
    """The model searched the web on the provider's side (not through the toolbox).

    Its results are now part of the model's context: outside content, like a web page.
    """

    query: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool call produced, sent back to the model."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """One message in a conversation.

    An assistant message may carry tool calls; the user message after it carries their
    results (that is how the model API expects them).
    """

    role: Role
    text: str
    tool_calls: tuple[ToolCall, ...] = field(default=())
    tool_results: tuple[ToolResult, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class SystemPrompt:
    """The system prompt in two parts, so a provider can cache the first.

    ``stable`` changes only when the configuration, the tools or the profile do. Anything
    that changes often (the clock) goes in ``volatile``: inside a cached part it would
    make every request miss the cache (ADR 0016).
    """

    stable: str
    volatile: str = ""

    @property
    def text(self) -> str:
        """The whole prompt, for a provider that does not cache."""
        return self.stable + self.volatile


class LLMError(Exception):
    """A language-model request failed. The message is safe to show to the user."""


class LLMProvider(Protocol):
    """Streams a reply from a language model."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: SystemPrompt,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        allow_tool_calls: bool = True,
    ) -> AsyncGenerator[str | ToolCall | WebSearch]:
        """Yield the reply as text fragments as soon as they are generated.

        If the model wants to use tools, the ``ToolCall`` objects follow at the end of
        the response; the caller runs them and asks again with the results. A provider
        that searches the web itself reports each search as a ``WebSearch``.

        With ``allow_tool_calls`` false the model must answer in text: ``tools`` are
        still listed (earlier tool calls in ``messages`` refer to them) but none may be
        called, not even the provider's own web search.

        Raises:
            LLMError: On any failure, with a user-presentable message.
        """
        ...

    async def warm_up(self) -> None:
        """Open the connection ahead of time so the next request starts faster.

        Must never raise: a failed warm-up only means the next request is slower.
        """
        ...
