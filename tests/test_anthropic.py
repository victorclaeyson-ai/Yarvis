from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from fakes import RecordingReporter
from yarvis.llm.anthropic import (
    DEFAULT_MODEL,
    AnthropicProvider,
    _to_param,  # pyright: ignore[reportPrivateUsage]
)
from yarvis.llm.anthropic import _explain as explain  # pyright: ignore[reportPrivateUsage]
from yarvis.llm.base import LLMError, Message, ToolCall, ToolResult, ToolSpec, WebSearch
from yarvis.usage import ClaudeUsage

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


# --------------------------------------------------------------------------- web search


def message_start() -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_search",
            "type": "message",
            "role": "assistant",
            "model": DEFAULT_MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    }


def text_block(index: int, text: str) -> list[dict[str, Any]]:
    block = {"type": "text", "text": ""}
    delta = {"type": "text_delta", "text": text}
    return [
        {"type": "content_block_start", "index": index, "content_block": block},
        {"type": "content_block_delta", "index": index, "delta": delta},
        {"type": "content_block_stop", "index": index},
    ]


def search_blocks(index: int, query: str, *, with_result: bool = True) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps({"query": query})},
        },
        {"type": "content_block_stop", "index": index},
    ]  # fmt: skip
    if with_result:
        result = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com/match",
                    "title": "Club Brugge 2-1 Anderlecht",
                    "encrypted_content": "ENCRYPTED",
                    "page_age": None,
                }
            ],
        }
        events += [
            {"type": "content_block_start", "index": index + 1, "content_block": result},
            {"type": "content_block_stop", "index": index + 1},
        ]
    return events


def end(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]


def streaming(*bodies: bytes) -> tuple[Callable[[httpx2.Request], httpx2.Response], list[Any]]:
    seen: list[dict[str, Any]] = []
    queue = list(bodies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200, content=queue.pop(0), headers={"content-type": "text/event-stream"}
        )

    return handler, seen


TIMER = ToolSpec("set_timer", "Start a timer.", {"type": "object", "properties": {}})


def searching(handler: Callable[[httpx2.Request], httpx2.Response]) -> AnthropicProvider:
    p = provider(handler)
    p.web_search = True
    return p


async def test_claude_searches_the_web_and_says_so() -> None:
    body = sse(
        message_start(),
        *text_block(0, "Let me check. "),
        *search_blocks(1, "Club Brugge score"),
        *text_block(3, "Club Brugge won two one."),
        *end("end_turn"),
    )
    handler, seen = streaming(body)
    events = [
        e
        async for e in searching(handler).stream(
            [Message("user", "how did Brugge do")], system="S", max_tokens=50, tools=[TIMER]
        )
    ]
    assert events == ["Let me check. ", WebSearch("Club Brugge score"), "Club Brugge won two one."]
    assert seen[0]["tools"][-1] == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }


async def test_no_web_search_without_tools_or_when_off() -> None:
    handler, seen = streaming(text_stream("Hi."), text_stream("Hi."))
    p = searching(handler)
    assert [e async for e in p.stream([Message("user", "x")], system="S", max_tokens=5)] == ["Hi."]
    assert "tools" not in seen[0]  # the last round, without tools: no search either
    off = provider(handler)
    events = [
        e async for e in off.stream([Message("user", "x")], system="S", max_tokens=5, tools=[TIMER])
    ]
    assert events == ["Hi."]
    assert [tool["name"] for tool in seen[1]["tools"]] == ["set_timer"]


async def test_a_paused_search_is_resumed() -> None:
    first = sse(
        message_start(), *search_blocks(0, "news Leuven", with_result=False), *end("pause_turn")
    )
    second = sse(message_start(), *text_block(0, "Quiet day in Leuven."), *end("end_turn"))
    handler, seen = streaming(first, second)
    events = [
        e
        async for e in searching(handler).stream(
            [Message("user", "news?")], system="S", max_tokens=50, tools=[TIMER]
        )
    ]
    assert events == [WebSearch("news Leuven"), "Quiet day in Leuven."]
    resumed = seen[1]["messages"][-1]
    assert resumed["role"] == "assistant"  # the paused turn goes back unchanged
    assert resumed["content"][0]["type"] == "server_tool_use"
    assert resumed["content"][0]["input"] == {"query": "news Leuven"}


async def test_a_search_before_a_tool_call_is_sent_back_exactly() -> None:
    """Search results are encrypted: the API needs the turn back as it was."""
    use: dict[str, object] = {"type": "tool_use", "id": "toolu_9", "name": "set_timer", "input": {}}
    tool: list[dict[str, object]] = [
        {"type": "content_block_start", "index": 2, "content_block": use},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"minutes": 90}'},
        },
        {"type": "content_block_stop", "index": 2},
    ]  # fmt: skip
    first = sse(message_start(), *search_blocks(0, "match length"), *tool, *end("tool_use"))
    handler, seen = streaming(first, text_stream("Done."))
    p = searching(handler)
    events = [
        e async for e in p.stream([Message("user", "x")], system="S", max_tokens=50, tools=[TIMER])
    ]
    call = ToolCall("toolu_9", "set_timer", {"minutes": 90})
    assert events == [WebSearch("match length"), call]
    follow_up = [
        Message("user", "x"),
        Message("assistant", "", tool_calls=(call,)),
        Message("user", "", tool_results=(ToolResult("toolu_9", "Set."),)),
    ]
    await collect(p, follow_up)
    sent = seen[1]["messages"][1]["content"]
    assert [block["type"] for block in sent] == [
        "server_tool_use", "web_search_tool_result", "tool_use",
    ]  # fmt: skip
    assert sent[1]["content"][0]["encrypted_content"] == "ENCRYPTED"
    for n in range(20):  # only recent turns are kept
        p._keep(f"id{n}", [])  # pyright: ignore[reportPrivateUsage]
    assert "toolu_9" not in p._turns  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- usage reports


def reporting(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> tuple[AnthropicProvider, RecordingReporter]:
    p = provider(handler)
    reporter = RecordingReporter()
    p.report_to(reporter)
    return p, reporter


async def test_each_request_reports_its_usage_and_that_claude_answers() -> None:
    handler, _ = streaming(text_stream("Hello."))
    p, reporter = reporting(handler)
    assert await collect(p, [Message("user", "Hi")]) == "Hello."
    assert reporter.claude_calls == [ClaudeUsage(DEFAULT_MODEL, input_tokens=12, output_tokens=5)]
    assert reporter.health_calls == [("claude", "ok", DEFAULT_MODEL.removeprefix("claude-"))]


async def test_searches_and_cached_tokens_are_counted() -> None:
    start = message_start()
    start["message"]["usage"] |= {"cache_creation_input_tokens": 100, "cache_read_input_tokens": 50}
    finish = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 30, "server_tool_use": {"web_search_requests": 1}},
    }
    body = sse(
        start,
        *search_blocks(0, "Club Brugge score"),
        *text_block(2, "They won."),
        finish,
        {"type": "message_stop"},
    )
    handler, _ = streaming(body)
    p, reporter = reporting(handler)
    p.web_search = True
    _ = [
        e async for e in p.stream([Message("user", "x")], system="S", max_tokens=50, tools=[TIMER])
    ]
    (usage,) = reporter.claude_calls
    assert (usage.cache_write_tokens, usage.cache_read_tokens) == (100, 50)
    assert usage.output_tokens == 30
    assert usage.web_searches == 1


async def test_a_reply_cut_off_is_still_counted() -> None:
    """A draft dropped mid-stream was billed for what it read and wrote so far."""
    body = sse(
        message_start(),
        *search_blocks(0, "news", with_result=False),
        *text_block(1, "A long answer that the user never hears in full"),
        *text_block(2, "and more."),
        *end("end_turn"),
    )
    handler, _ = streaming(body)
    p, reporter = reporting(handler)
    p.web_search = True
    stream = p.stream([Message("user", "x")], system="S", max_tokens=50, tools=[TIMER])
    assert isinstance(await anext(stream), WebSearch)
    assert isinstance(await anext(stream), str)
    await stream.aclose()
    (usage,) = reporter.claude_calls
    assert usage.input_tokens == 12
    assert usage.output_tokens == 12  # 47 characters received, about 4 per token
    assert usage.web_searches == 1  # seen, though the final counts never came
    assert reporter.health_calls == []


async def test_a_failure_marks_claude_as_down_and_counts_nothing() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "no"}}
        return httpx2.Response(401, json=body)

    p, reporter = reporting(handler)
    with pytest.raises(LLMError):
        await collect(p, [Message("user", "Hi")])
    assert reporter.claude_calls == []
    assert reporter.health_calls == [
        (
            "claude",
            "error",
            "Claude rejected the API key. Store a valid one with: yarvis key set anthropic",
        )
    ]


async def test_without_a_reporter_nothing_is_counted() -> None:
    handler, _ = streaming(text_stream("Hello."))
    assert await collect(provider(handler), [Message("user", "Hi")]) == "Hello."


async def test_a_stream_that_fails_before_it_starts_counts_nothing() -> None:
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    handler, _ = streaming(sse(overloaded))
    p, reporter = reporting(handler)
    with pytest.raises(LLMError):
        await collect(p, [Message("user", "Hi")])
    assert reporter.claude_calls == []
    assert reporter.health_calls[0][:2] == ("claude", "error")
