"""Claude via the official Anthropic SDK, streamed.

Web search runs on Anthropic's side (the ``web_search`` server tool): Claude decides when
to search, and the results stay inside the request. Each search is reported to the caller
as a ``WebSearch`` so the turn can be treated as holding outside content.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Final

import anthropic
import httpx2

from yarvis.llm.base import LLMError, ToolCall, WebSearch

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from anthropic.types import (
        ContentBlock,
        ContentBlockParam,
        MessageParam,
        TextBlockParam,
        ToolParam,
        ToolResultBlockParam,
        ToolUnionParam,
        ToolUseBlockParam,
        WebSearchTool20250305Param,
    )
    from pydantic import SecretStr

    from yarvis.llm.base import Message, ToolSpec

DEFAULT_MODEL: Final = "claude-haiku-5-5"
"""Anthropic's lowest-latency model: the right default for a voice assistant."""

REQUEST_TIMEOUT_SECONDS: Final = 30.0

KEEPALIVE_SECONDS: Final = 300.0
"""Keep the HTTPS connection open between questions. The HTTP client's default (5 s) meant
almost every question paid for a fresh TLS handshake (~0.2-0.5 s)."""

WEB_SEARCHES_PER_REQUEST: Final = 3
"""At most this many searches per answer (each costs about a cent)."""
_MAX_CONTINUATIONS: Final = 3  # a long search can pause the turn; it is resumed this often
_KEPT_TURNS: Final = 16

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """``LLMProvider`` for Claude."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        web_search: bool = False,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.web_search = web_search
        """Offer Claude's web search whenever tools are offered."""
        # Turns that searched and then called our tools: the API needs them back exactly
        # as sent (search results are encrypted), keyed by their first tool call.
        self._turns: OrderedDict[str, list[ContentBlock]] = OrderedDict()
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            max_retries=1,  # a voice user would rather hear an error than wait
            timeout=REQUEST_TIMEOUT_SECONDS,
            http_client=anthropic.DefaultAsyncHttpxClient(
                limits=httpx2.Limits(
                    max_connections=10,
                    max_keepalive_connections=4,
                    keepalive_expiry=KEEPALIVE_SECONDS,
                ),
            ),
        )

    async def warm_up(self) -> None:
        """Open (or refresh) the HTTPS connection with a free, tiny request."""
        try:
            await self._client.models.list(limit=1)
        except anthropic.APIError as exc:
            logger.debug("Warm-up request failed: %s", type(exc).__name__)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncGenerator[str | ToolCall | WebSearch]:
        """Yield reply text as it is generated, web searches as they happen, then tool calls."""
        params = [self._to_param(message) for message in messages]
        tool_params: list[ToolUnionParam] = [
            _tool_param(t.name, t.description, dict(t.input_schema)) for t in tools
        ]
        if tools and self.web_search:
            tool_params.append(_WEB_SEARCH)
        turn: list[ContentBlock] = []
        try:
            for _ in range(_MAX_CONTINUATIONS + 1):
                async with self._client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=params,
                    tools=tool_params or anthropic.omit,
                ) as stream:
                    async for event in stream:
                        if event.type == "text":
                            yield event.text
                        elif (
                            event.type == "content_block_stop"
                            and event.content_block.type == "server_tool_use"
                            and event.content_block.name == "web_search"
                        ):
                            logger.info("Claude searched the web")
                            yield WebSearch(str(event.content_block.input.get("query", "")))
                    final = await stream.get_final_message()
                turn += final.content
                if final.stop_reason != "pause_turn":
                    break
                # A long search paused the turn: send it back unchanged to let it go on.
                paused: MessageParam = {"role": "assistant", "content": list(turn)}
                params = [*params, paused]
        except anthropic.APIError as exc:
            raise LLMError(_explain(exc, self.model)) from exc
        calls = [block for block in turn if block.type == "tool_use"]
        if calls and any(block.type not in {"text", "tool_use"} for block in turn):
            self._keep(calls[0].id, turn)
        for block in calls:
            yield ToolCall(id=block.id, name=block.name, arguments=dict(block.input))

    def _keep(self, call_id: str, turn: list[ContentBlock]) -> None:
        self._turns[call_id] = turn
        while len(self._turns) > _KEPT_TURNS:
            self._turns.popitem(last=False)

    def _to_param(self, message: Message) -> MessageParam:
        if message.role == "assistant" and message.tool_calls:
            kept = self._turns.get(message.tool_calls[0].id)
            if kept is not None:  # it searched first: send it back exactly as received
                return {"role": "assistant", "content": kept}
        return _to_param(message)


def _tool_param(name: str, description: str, schema: dict[str, object]) -> ToolParam:
    return {"name": name, "description": description, "input_schema": schema}


_WEB_SEARCH: Final[WebSearchTool20250305Param] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": WEB_SEARCHES_PER_REQUEST,
}


def _to_param(message: Message) -> MessageParam:
    if not message.tool_calls and not message.tool_results:
        if message.role == "user":
            return {"role": "user", "content": message.text}
        return {"role": "assistant", "content": message.text}
    blocks: list[ContentBlockParam] = []
    if message.role == "user":
        for result in message.tool_results:  # results must come first in the message
            result_block: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            blocks.append(result_block)
        if message.text or not blocks:
            text_block: TextBlockParam = {"type": "text", "text": message.text}
            blocks.append(text_block)
        return {"role": "user", "content": blocks}
    if message.text or not message.tool_calls:
        blocks.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        use_block: ToolUseBlockParam = {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": dict(call.arguments),
        }
        blocks.append(use_block)
    return {"role": "assistant", "content": blocks}


def _explain(exc: anthropic.APIError, model: str) -> str:  # noqa: PLR0911 - a flat mapping reads best
    """Turn an SDK error into a short message that tells the user what to do."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "Claude rejected the API key. Store a valid one with: yarvis key set anthropic"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This API key is not allowed to use Claude. Check it at console.anthropic.com."
    if isinstance(exc, anthropic.NotFoundError):
        return f"Model {model!r} was not found. Check llm.model in your config."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude is rate-limiting this key. Wait a moment and try again."
    if isinstance(exc, anthropic.BadRequestError) and "credit" in str(exc).lower():
        return "Your Anthropic account is out of credits. Add some at console.anthropic.com."
    if isinstance(exc, anthropic.APITimeoutError):
        return "Claude took too long to answer. Try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach Claude. Check your internet connection."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude returned an error (HTTP {exc.status_code}). Try again shortly."
    return "Something went wrong talking to Claude. Try again."
