"""Short-term conversation memory and the agent loop around an LLM provider.

Replies are produced as ``Draft`` objects. A draft starts generating immediately, but it
only becomes part of the conversation once someone consumes it. That lets the pipeline
start asking Claude *speculatively* — while the user may still be talking — and throw the
draft away if they continue, without the abandoned exchange ever entering the history.

With tools, a draft is an agent loop: stream the reply, run any tool calls through the
``Toolbox`` (and so through the permission gate), send the results back, repeat. A draft
never runs a tool while it is still speculative: it waits until it is consumed.

Every exchange that is remembered passes through ``Conversation.remember``: the single
place where long-term memory will hook in later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import aclosing
from typing import TYPE_CHECKING, Final

from yarvis.agent.confirmation import classify_answer
from yarvis.llm.base import LLMError, Message, ToolCall, WebSearch

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from yarvis.agent.toolbox import PendingAction, Toolbox
    from yarvis.llm.base import LLMProvider, SystemPrompt

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES = 20
_EXCHANGE = 2  # a user message plus the assistant's reply
MAX_TOOL_ROUNDS: Final = 4
"""Model calls that may use tools in one turn; then it must answer in words."""
TURN_TIMEOUT_S: Final = 60.0


class Draft:
    """A reply that is being generated, possibly before we know it will be used."""

    def __init__(self, conversation: Conversation, user_text: str) -> None:
        self.user_text = user_text
        self.first_token_at: float | None = None
        self._conversation = conversation
        self._queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        self._consumed = False
        self._live = asyncio.Event()
        self._last_char = ""  # of the text emitted so far
        self.done: list[str] = []
        """What this reply has actually done so far ("play … on Spotify")."""
        self.searches: list[str] = []
        """Web searches the model made (on the provider's side) for this reply."""
        self.read_outside = False
        """Whether a tool this reply ran returned outside content (inherited taint aside)."""
        self._task = asyncio.create_task(self._generate())

    async def _generate(self) -> None:
        try:
            async with asyncio.timeout(TURN_TIMEOUT_S):
                await self._agent_loop()
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._queue.put_nowait(LLMError("That took too long, so I stopped."))
        except Exception as exc:  # noqa: BLE001 - handed to the consumer, who re-raises it
            self._queue.put_nowait(exc)
        finally:
            self._queue.put_nowait(None)

    async def _agent_loop(self) -> None:
        conversation = self._conversation
        toolbox = conversation.toolbox
        messages = conversation.messages_for(self.user_text)
        if (pending := conversation.pending) is not None:
            messages[-1] = Message("user", await self._resolve(pending))
        # Outside content read earlier lives on in replies still in the history (ADR 0018).
        tainted = conversation.holds_outside_content or self.read_outside
        for round_number in range(MAX_TOOL_ROUNDS + 1):
            # The last round must answer in words: tools stay listed, but none may be called.
            may_call = round_number < MAX_TOOL_ROUNDS
            text, calls = await self._stream(messages, allow_tool_calls=may_call)
            if not calls or toolbox is None or not may_call:
                return
            await self._live.wait()  # never act on a reply the user may not want
            # Search results read on the provider's side are outside content too.
            outcome = await toolbox.run(calls, tainted=tainted or bool(self.searches))
            self.done += outcome.done
            tainted = outcome.tainted
            self.read_outside = self.read_outside or outcome.read_outside
            messages += [
                Message("assistant", text, tool_calls=tuple(calls)),
                Message("user", "", tool_results=outcome.results),
            ]
            if outcome.pending is not None:
                conversation.pending = outcome.pending
                self._say(outcome.pending.question)  # asked by code, not by the model
                return

    async def _stream(
        self, messages: list[Message], *, allow_tool_calls: bool
    ) -> tuple[str, list[ToolCall]]:
        conversation = self._conversation
        toolbox = conversation.toolbox
        specs = toolbox.specs() if toolbox is not None else ()
        parts: list[str] = []
        calls: list[ToolCall] = []
        async for event in conversation.llm.stream(
            messages,
            system=conversation.system_prompt(),
            max_tokens=conversation.max_tokens,
            tools=specs,
            allow_tool_calls=allow_tool_calls,
        ):
            if isinstance(event, ToolCall):
                calls.append(event)
                continue
            if isinstance(event, WebSearch):
                self._searched(event.query)
                continue
            if self.first_token_at is None:
                self.first_token_at = conversation.clock()
            parts.append(event)
            self._emit(event, new_round=not parts[:-1])
        return "".join(parts), calls

    async def _resolve(self, pending: PendingAction) -> str:
        """The user answered a confirmation question: act on it, tell the model."""
        await self._live.wait()  # a speculative draft must not decide anything
        conversation = self._conversation
        if conversation.pending is pending:
            conversation.pending = None
        answer = classify_answer(self.user_text)
        toolbox = conversation.toolbox
        # The toolbox set the deadline with its own clock, so only it can check it.
        if toolbox is None or toolbox.expired(pending):
            note = f"The question about whether to {pending.description} expired; nothing was done."
        elif answer == "yes":
            logger.info("Confirmed by the user: %s", pending.tool.name)
            confirmed = await toolbox.confirm(pending)
            note = confirmed.note
            if confirmed.done:
                self.done.append(pending.description)
            # Its result is outside content like any other: this turn is tainted from here.
            self.read_outside = self.read_outside or confirmed.read_outside
        elif answer == "no":
            note = f"The user declined, so you did not {pending.description}."
        else:
            note = f"The user said something else, so you did not {pending.description}."
        return f"{self.user_text}\n\n[Yarvis system note: {note}]"

    def _searched(self, query: str) -> None:
        self.searches.append(query)
        toolbox = self._conversation.toolbox
        if toolbox is not None and self._live.is_set():  # a guess at a pause shows nothing
            toolbox.report(_search_activity(query))

    def _say(self, text: str) -> None:
        self._emit(text, new_round=True)

    def _emit(self, text: str, *, new_round: bool) -> None:
        """Queue text; text from a new model round is separated from the previous one."""
        if not text:
            return
        if (
            new_round
            and self._last_char
            and not self._last_char.isspace()
            and not text[0].isspace()
        ):
            text = f" {text}"  # "Sure." + "Done." must not become "Sure.Done."
        self._last_char = text[-1]
        self._queue.put_nowait(text)

    async def deltas(self) -> AsyncGenerator[str]:
        """Consume the reply. From this moment the exchange will be remembered.

        Whatever was consumed is remembered when the stream ends, including a partial reply
        if it is interrupted, so the model knows what the user actually heard.
        """
        if self._consumed:
            msg = "a draft can only be consumed once"
            raise RuntimeError(msg)
        self._consumed = True
        self._live.set()
        if (toolbox := self._conversation.toolbox) is not None:
            for query in self.searches:  # made while it was still a guess
                toolbox.report(_search_activity(query))
        heard: list[str] = []
        finished = False
        try:
            while (item := await self._queue.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                heard.append(item)
                yield item
            finished = True
        finally:
            if not self._task.done():
                self._task.cancel()
            reply = "".join(heard).strip()
            # Tool calls and results are not kept in the history (they are large, and outside
            # content must not outlive its turn), so what was done is noted, by code. Without
            # it, the model reads its own "Playing ..." as a claim it never backed up, and
            # "corrects" it next time, or does it all again ("skip" must not skip two songs).
            done = [*self.done, *(_search_activity(query) for query in self.searches)]
            if done and not finished:
                reply = f"{reply} [Interrupted. Already done: {'; '.join(done)}.]".strip()
            elif done:
                reply = f"{reply} [Done: {'; '.join(done)}.]".strip()
            if reply:
                # Outside content can live on in this reply: while it is in the history, later
                # turns count as tainted too (ADR 0018). Inherited taint does not count here,
                # or it would never leave.
                outside = self.read_outside or bool(self.searches)
                self._conversation.remember(self.user_text, reply, outside=outside)

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
        system_prompt: Callable[[], SystemPrompt],
        max_tokens: int,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        clock: Callable[[], float] = time.monotonic,
        toolbox: Toolbox | None = None,
    ) -> None:
        if max_messages < _EXCHANGE or max_messages % _EXCHANGE:
            msg = "max_messages must be a positive even number"
            raise ValueError(msg)
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.clock = clock
        self.toolbox = toolbox
        self.pending: PendingAction | None = None
        """An action waiting for the user's "yes" (asked at the end of the last reply)."""
        self._max_messages = max_messages
        self._history: list[Message] = []
        self._outside: list[bool] = []
        """Per exchange in the history: whether its turn read outside content."""

    @property
    def history(self) -> tuple[Message, ...]:
        """The remembered messages, oldest first."""
        return tuple(self._history)

    def reset(self) -> None:
        """Forget the conversation."""
        self._history.clear()
        self._outside.clear()
        self.pending = None

    @property
    def holds_outside_content(self) -> bool:
        """Whether a reply in the history was made after reading outside content.

        Such a reply can repeat what a page or email said, so every turn counts as tainted
        while it is there (ADR 0018), not only the turn that read it.
        """
        return any(self._outside)

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
            del self._outside[-1:]
            return True
        return False

    def remember(self, user_text: str, reply: str, *, outside: bool = False) -> None:
        """Record a finished exchange; ``outside``: its turn read outside content."""
        self._history.extend((Message("user", user_text), Message("assistant", reply)))
        self._outside.append(outside)
        overflow = len(self._history) - self._max_messages
        if overflow > 0:
            del self._history[:overflow]
            del self._outside[: overflow // _EXCHANGE]


def _search_activity(query: str) -> str:
    """What the user sees while Claude searches ("search the web for “trains to Ghent”")."""
    return f"search the web for “{query[:80]}”" if query else "search the web"
