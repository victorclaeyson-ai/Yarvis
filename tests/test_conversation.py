from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fakes import FakeLLM
from yarvis.agent.conversation import Conversation
from yarvis.agent.prompts import system_prompt
from yarvis.llm.base import LLMError, Message


def make(llm: FakeLLM, max_messages: int = 4) -> Conversation:
    return Conversation(
        llm, system_prompt=lambda: "SYSTEM", max_tokens=99, max_messages=max_messages
    )


async def collect(conversation: Conversation, text: str) -> str:
    return "".join([delta async for delta in conversation.respond(text)])


async def test_reply_is_streamed_and_remembered() -> None:
    llm = FakeLLM(["Hel", "lo!"])
    conversation = make(llm)
    assert await collect(conversation, "Hi") == "Hello!"
    assert conversation.history == (Message("user", "Hi"), Message("assistant", "Hello!"))
    messages, system, max_tokens = llm.calls[0]
    assert messages == [Message("user", "Hi")]
    assert (system, max_tokens) == ("SYSTEM", 99)


async def test_history_is_sent_and_trimmed_by_whole_exchanges() -> None:
    llm = FakeLLM(["one"], ["two"], ["three"])
    conversation = make(llm, max_messages=4)
    for text in ("a", "b", "c"):
        await collect(conversation, text)
    assert [m.text for m in conversation.history] == ["b", "two", "c", "three"]
    assert [m.text for m in llm.calls[2][0]] == ["a", "one", "b", "two", "c"]


async def test_failed_reply_without_text_is_not_remembered() -> None:
    conversation = make(FakeLLM([], error=LLMError("down")))
    with pytest.raises(LLMError):
        await collect(conversation, "Hi")
    assert conversation.history == ()


async def test_partial_reply_is_remembered() -> None:
    conversation = make(FakeLLM(["Part of it"], error=LLMError("cut")))
    with pytest.raises(LLMError):
        await collect(conversation, "Hi")
    assert conversation.history[-1] == Message("assistant", "Part of it")


async def test_reset() -> None:
    conversation = make(FakeLLM(["x"]))
    await collect(conversation, "Hi")
    conversation.reset()
    assert conversation.history == ()


def test_max_messages_must_be_even() -> None:
    with pytest.raises(ValueError, match="even"):
        make(FakeLLM(), max_messages=3)


def test_system_prompt_mentions_name_time_and_voice_rules() -> None:
    prompt = system_prompt("Yarvis", datetime(2026, 11, 14, 13, 5, tzinfo=UTC))
    assert "You are Yarvis" in prompt
    assert "Saturday 14 November 2026, 13:05 (UTC)" in prompt
    assert "no markdown" in prompt
    assert "Never pretend you did something" in prompt


def test_system_prompt_defaults_to_now() -> None:
    assert str(datetime.now().year) in system_prompt("Yarvis")  # noqa: DTZ005
