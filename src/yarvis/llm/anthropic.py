"""Claude via the official Anthropic SDK, streamed."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import anthropic
import httpx2

from yarvis.llm.base import LLMError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from anthropic.types import MessageParam
    from pydantic import SecretStr

    from yarvis.llm.base import Message

DEFAULT_MODEL: Final = "claude-haiku-5-5"
"""Anthropic's lowest-latency model: the right default for a voice assistant."""

REQUEST_TIMEOUT_SECONDS: Final = 30.0

KEEPALIVE_SECONDS: Final = 300.0
"""Keep the HTTPS connection open between questions. The HTTP client's default (5 s) meant
almost every question paid for a fresh TLS handshake (~0.2-0.5 s)."""

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """``LLMProvider`` for Claude."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
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
    ) -> AsyncGenerator[str]:
        """Yield reply text as it is generated."""
        params = [_to_param(message) for message in messages]
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=params,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIError as exc:
            raise LLMError(_explain(exc, self.model)) from exc


def _to_param(message: Message) -> MessageParam:
    if message.role == "user":
        return {"role": "user", "content": message.text}
    return {"role": "assistant", "content": message.text}


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
