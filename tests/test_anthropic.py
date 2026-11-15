from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from yarvis.llm.anthropic import (
    DEFAULT_MODEL,
    AnthropicProvider,
    _to_param,  # pyright: ignore[reportPrivateUsage]
)
from yarvis.llm.anthropic import _explain as explain  # pyright: ignore[reportPrivateUsage]
from yarvis.llm.base import LLMError, Message, ToolCall, ToolResult, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def text_stream(*chunks: str) -> bytes:
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": DEFAULT_MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    events += [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": c}}
        for c in chunks
    ]
    events += [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    return sse(*events)


def provider(handler: Callable[[httpx2.Request], httpx2.Response]) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(SecretStr("unused"), client=client)


async def collect(p: AnthropicProvider, messages: list[Message]) -> str:
    events = [e async for e in p.stream(messages, system="Be brief.", max_tokens=50)]
    return "".join(e for e in events if isinstance(e, str))


async def test_streams_text_and_sends_the_right_request() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            content=text_stream("Hello", ", Victor."),
            headers={"content-type": "text/event-stream"},
        )

    messages = [Message("user", "Hi"), Message("assistant", "Hey"), Message("user", "Again")]
    assert await collect(provider(handler), messages) == "Hello, Victor."
    body = seen[0]
    assert body["model"] == DEFAULT_MODEL
    assert body["system"] == "Be brief."
    assert body["max_tokens"] == 50
    assert body["stream"] is True
    assert body["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hey"},
        {"role": "user", "content": "Again"},
    ]


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            401,
            {"type": "authentication_error", "message": "invalid x-api-key"},
            "rejected the API key",
        ),
        (403, {"type": "permission_error", "message": "no"}, "not allowed"),
        (404, {"type": "not_found_error", "message": "model"}, "was not found"),
        (429, {"type": "rate_limit_error", "message": "slow down"}, "rate-limiting"),
        (
            400,
            {"type": "invalid_request_error", "message": "Your credit balance is too low"},
            "credits",
        ),
        (400, {"type": "invalid_request_error", "message": "bad"}, "HTTP 400"),
        (529, {"type": "overloaded_error", "message": "busy"}, "HTTP 529"),
    ],
)
async def test_errors_become_friendly_messages(
    status: int, body: dict[str, str], expected: str
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"type": "error", "error": body})

    with pytest.raises(LLMError, match=expected):
        await collect(provider(handler), [Message("user", "Hi")])


async def test_connection_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline", request=request)

    with pytest.raises(LLMError, match="internet connection"):
        await collect(provider(handler), [Message("user", "Hi")])


async def test_timeouts() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError, match="too long"):
        await collect(provider(handler), [Message("user", "Hi")])


def test_default_client_uses_the_key() -> None:
    p = AnthropicProvider(
        SecretStr("sk-ant-key-for-test-only"), model="claude-sonnet-5-5"
    )  # gitleaks:allow
    assert p.model == "claude-sonnet-5-5"


def test_unknown_api_errors_get_a_generic_message() -> None:
    error = anthropic.APIError("weird", request=httpx2.Request("GET", "https://x"), body=None)
    assert "Something went wrong" in explain(error, DEFAULT_MODEL)


async def test_warm_up_lists_models_and_never_raises() -> None:
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        return httpx2.Response(
            200, json={"data": [], "has_more": False, "first_id": None, "last_id": None}
        )

    await provider(handler).warm_up()
    assert requests == ["/v1/models"]

    def failing(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "x"}}
        )

    await provider(failing).warm_up()  # swallowed

    def offline(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline", request=request)

    await provider(offline).warm_up()  # swallowed


def tool_stream() -> bytes:
    start: dict[str, Any] = {
        "type": "message_start",
        "message": {
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "model": DEFAULT_MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    }
    return sse(
        start,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "On it."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "set_timer",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"minutes": 10}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    )


async def test_tools_are_offered_and_tool_calls_come_back() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200, content=tool_stream(), headers={"content-type": "text/event-stream"}
        )

    spec = ToolSpec("set_timer", "Start a timer.", {"type": "object", "properties": {}})
    events = [
        e
        async for e in provider(handler).stream(
            [Message("user", "timer")], system="S", max_tokens=50, tools=[spec]
        )
    ]
    assert events == ["On it.", ToolCall("toolu_1", "set_timer", {"minutes": 10})]
    assert seen[0]["tools"] == [
        {"name": "set_timer", "description": "Start a timer.", "input_schema": spec.input_schema}
    ]


async def test_tool_messages_are_sent_as_blocks() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200, content=text_stream("Done."), headers={"content-type": "text/event-stream"}
        )

    messages = [
        Message("user", "set a timer"),
        Message("assistant", "", tool_calls=(ToolCall("t1", "set_timer", {"minutes": 1}),)),
        Message("user", "", tool_results=(ToolResult("t1", "Timer 1 set.", is_error=False),)),
    ]
    assert await collect(provider(handler), messages) == "Done."
    assert "tools" not in seen[0]
    assert seen[0]["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "set_timer", "input": {"minutes": 1}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "Timer 1 set.",
                    "is_error": False,
                }
            ],
        },
    ]
    mixed = Message("user", "and also", tool_results=(ToolResult("t2", "Nope.", is_error=True),))
    assert list(_to_param(mixed)["content"])[-1] == {"type": "text", "text": "and also"}
    said = Message("assistant", "Sure.", tool_calls=(ToolCall("t3", "x", {}),))
    assert next(iter(_to_param(said)["content"])) == {"type": "text", "text": "Sure."}
