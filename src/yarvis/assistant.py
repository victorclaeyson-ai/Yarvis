"""The interactive loop: wait for input, listen, reply, repeat.

Input arrives three ways: a typed line, an empty line or the talk button ("listen now"), or
the wake word. While Yarvis replies, a barge-in monitor can interrupt it; the next request
is then heard straight away, keeping the audio that interrupted it.

Every command runs as its own task so it can be stopped from another thread (the app's
stop button) without ending the loop. Unexpected errors are reported and the loop
continues: a voice assistant should never silently die mid-conversation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from yarvis.audio.devices import AudioBackendError
from yarvis.listen import PauseHooks
from yarvis.llm.base import LLMError
from yarvis.pipeline import TurnObserver
from yarvis.speech.turn_taking import ends_conversation, meant_for_assistant, sounds_finished

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
    from contextlib import AbstractAsyncContextManager

    from yarvis.agent.greetings import Greeter
    from yarvis.ears import Interruption
    from yarvis.listen import PauseControl, Utterance
    from yarvis.pipeline import Speculation, Turn, TurnResult
    from yarvis.speech.base import AudioArray
    from yarvis.tools.ducking import Ducker

logger = logging.getLogger(__name__)

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})
GREETING_WAIT_S = 8.0
LISTEN_STRETCH_S = 20.0  # an open conversation listens in stretches this long
FULL_STRETCH = 0.9
"""After a greeting, how long to wait for the request (at least)."""


class AssistantUi(TurnObserver, Protocol):
    """Everything the user sees."""

    def ready(self, *, voice: bool, wake_word: bool) -> None:
        """Waiting for input (``wake_word``: listening for "Hey Yarvis" too)."""
        ...

    def woke(self) -> None:
        """The wake word was heard."""
        ...

    def listening(self) -> None:
        """Capturing a request."""
        ...

    def speech_started(self) -> None:
        """The user started talking."""
        ...

    def level(self, value: float) -> None:
        """Microphone input level (RMS) while listening."""
        ...

    def nothing_heard(self) -> None:
        """No speech, or nothing intelligible."""
        ...

    def reply_finished(self, result: TurnResult) -> None:
        """The reply has been fully spoken."""
        ...

    def said(self, text: str) -> None:
        """Yarvis says a fixed phrase, e.g. a greeting."""
        ...

    def alert(self, text: str) -> None:
        """A timer or reminder is due (Yarvis is about to say ``text``)."""
        ...

    def activity(self, text: str) -> None:
        """Yarvis is doing something ("set a 10-minute timer")."""
        ...

    def interrupted(self) -> None:
        """The user talked over the reply; Yarvis stopped and is listening."""
        ...

    def error(self, message: str) -> None:
        """Something went wrong; the loop continues."""
        ...

    def stopped(self) -> None:
        """The current command was stopped by the user."""
        ...


class Ears(Protocol):
    """The microphone side: listening, the wake word and barge-in (see ``yarvis.ears``)."""

    @property
    def wake_word(self) -> bool:
        """Whether ``wake`` is available."""
        ...

    @property
    def barge_in(self) -> bool:
        """Whether ``interruption`` is available."""
        ...

    def session(self) -> AbstractAsyncContextManager[None]:
        """Keep the microphone open across listening and replying."""
        ...

    def listen(
        self,
        hooks: PauseHooks,
        *,
        follow_up: bool = False,
        prefix: Sequence[AudioArray] | None = None,
        timeout_s: float | None = None,
    ) -> Awaitable[Utterance | None]:
        """Capture one request (``prefix``: audio already captured; see ``yarvis.ears``)."""
        ...

    def wake(self) -> Awaitable[Sequence[AudioArray]]:
        """Wait for the wake word; returns the audio around it."""
        ...

    def interruption(self, *, wake_word_only: bool = False) -> Awaitable[Interruption]:
        """Wait until the user interrupts a reply (``wake_word_only``: only "Hey Yarvis")."""
        ...


@dataclass(frozen=True, slots=True)
class TurnTaking:
    """How the assistant decides when you are done and whether to keep listening."""

    speculate: bool = True
    """Start transcribing and drafting a reply during pauses."""
    thinking_pause_ms: int | None = 2_000
    """Silence allowed when the sentence sounds unfinished (None: no adaptive waiting)."""
    follow_up_s: float = 15.0
    """After a reply, anything said within this long counts as the answer (0: off). With
    ``stay_open_s``, only after Yarvis asked something ("Should I send it?")."""
    stay_open_s: float = 0.0
    """After a reply, keep listening this long for requests meant for Yarvis: its name or
    a command ("open Spotify"). Other speech is ignored and goes nowhere. 0: off."""


@dataclass(frozen=True, slots=True)
class _Listen:
    """A request to listen (the next step of the loop)."""

    follow_up: bool = False
    """Part of an open conversation: silence quietly ends it."""
    until: float | None = None
    """When the open conversation ends (``clock`` time)."""
    prefix: Sequence[AudioArray] | None = None
    """Audio already captured; None starts fresh."""
    woken: bool = False
    """The audio starts with the wake word."""
    preface: str = ""
    """A request that was interrupted before it was answered; the new words extend it."""
    anything_until: float | None = None
    """Until then, whatever is said is taken as meant for Yarvis (an answer)."""
    follow_ups_until: float | None = None
    """Until then, a follow-up question ("and what about tomorrow?") counts too."""
    quiet: bool = False
    """Keep listening without announcing it again (nothing happened since)."""


@dataclass(frozen=True, slots=True)
class _Woken:
    prefix: Sequence[AudioArray]


@dataclass(frozen=True, slots=True)
class _Announcement:
    text: str


class Assistant:
    """Reads input lines; an empty line means "listen", any text is sent as typed input."""

    def __init__(
        self,
        *,
        turn: Turn,
        lines: AsyncIterator[str],
        ui: AssistantUi,
        ears: Ears | None,
        turn_taking: TurnTaking | None = None,
        greeter: Greeter | None = None,
        announcements: asyncio.Queue[str] | None = None,
        ducker: Ducker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._turn = turn
        self._ducker = ducker
        self._ducking: asyncio.Task[None] | None = None
        self._playback = asyncio.Event()
        """Set when a tool started music or a video during the current reply."""
        self._announcements = announcements
        self._greeter = greeter
        self._clock = clock
        self._lines = lines
        self._ui = ui
        self._ears = ears
        self._turn_taking = turn_taking or TurnTaking()
        self._wake_word = ears is not None and ears.wake_word
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current: asyncio.Task[_Listen | None] | None = None
        self._next_line: asyncio.Future[str | None] | None = None

    async def run(self) -> None:
        """Run until the user quits or input ends."""
        self._loop = asyncio.get_running_loop()
        step: _Listen | None = None
        try:
            while True:
                # A due timer waits for a quiet moment: not while the user is mid-request.
                if (step is None or step.prefix is None) and (text := self._due()) is not None:
                    step = await self._run_command(self._announce(text))
                    continue
                if step is not None:
                    step = await self._run_command(self._listen_turn(step))
                    continue
                await self._music_back_up()  # the conversation is over
                self._ui.ready(voice=self._ears is not None, wake_word=self._wake_word)
                event = await self._next_input()
                if isinstance(event, _Announcement):
                    step = await self._run_command(self._announce(event.text))
                    continue
                if isinstance(event, _Woken):
                    self._music_down()
                    self._ui.woke()
                    await self._turn.chime()
                    step = _Listen(prefix=event.prefix, woken=True)
                    continue
                if event is None or event.strip().lower() in QUIT_COMMANDS:
                    return
                if text := event.strip():
                    step = await self._run_command(self._text_turn(text))
                else:
                    self._music_down()
                    step = _Listen()
        finally:
            if self._next_line is not None:
                self._next_line.cancel()
            await self._music_back_up()

    def _music_down(self) -> None:
        """Turn music down for the conversation, without delaying the listening."""
        if self._ducker is None or (self._ducking is not None and not self._ducking.done()):
            return
        self._ducking = asyncio.create_task(self._ducker.duck())

    def media_started(self) -> None:
        """A tool started music or a video: keep it out of the microphone.

        For the rest of the reply only "Hey Yarvis" can interrupt (the song would otherwise
        "talk over" Yarvis), the music is turned down while Yarvis speaks, and the
        conversation ends after the reply instead of transcribing the music.
        """
        self._playback.set()
        self._music_down()

    async def _music_back_up(self) -> None:
        if self._ducker is None or self._ducking is None:
            return
        ducking, self._ducking = self._ducking, None
        with suppress(Exception):
            await ducking  # it may still be on its way down
        await self._ducker.restore()

    def stop(self) -> None:
        """Stop whatever is happening (listening, thinking, speaking). Thread-safe."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._cancel_current)

    def _cancel_current(self) -> None:
        if self._current is not None and not self._current.done():
            logger.info("Stopping the current command")
            self._current.cancel()

    # ------------------------------------------------------------------------- input

    async def _next_input(self) -> str | _Woken | _Announcement | None:
        """Whatever comes first: a typed line or button press, a due timer, the wake word."""
        if self._next_line is None:
            # Kept across calls: cancelling ``anext`` would close the input generator.
            self._next_line = asyncio.ensure_future(anext(self._lines, None))
        ears = self._ears
        hearing = ears.wake() if ears is not None and self._wake_word else None
        waking = asyncio.ensure_future(hearing) if hearing is not None else None
        queue = self._announcements
        due = asyncio.ensure_future(queue.get()) if queue is not None else None
        others = {task for task in (waking, due) if task is not None}
        try:
            await asyncio.wait({self._next_line, *others}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in others:
                task.cancel()
            if others:
                await asyncio.wait(others)  # one reader at a time on the microphone
        announcement = due.result() if due is not None and not due.cancelled() else None
        if self._next_line.done():
            if announcement is not None and queue is not None:
                queue.put_nowait(announcement)  # not lost: said after this command
            line, self._next_line = self._next_line, None
            return line.result()
        if announcement is not None:
            return _Announcement(announcement)
        if waking is None:  # pragma: no cover - something above finished first
            return await self._next_input()
        error = waking.exception()
        if error is None:
            return _Woken(waking.result())
        self._wake_word = False  # do not retry in a tight loop; the buttons still work
        self._report(error, context="The wake word stopped working")
        return await self._next_input()

    def _due(self) -> str | None:
        """A timer or reminder that is waiting to be said."""
        queue = self._announcements
        if queue is None or queue.empty():
            return None
        return queue.get_nowait()

    async def _announce(self, text: str) -> _Listen | None:
        """Say a due timer or reminder, then give the user a moment to respond."""
        logger.info("Announcing a timer or reminder")
        self._ui.alert(text)
        await self._turn.chime()
        outcome = await self._interruptible(self._turn.say(text))
        if isinstance(outcome, _Interrupted):
            return await self._listen_after(outcome.interruption)
        return self._open_conversation(asked=True)  # "thanks", "five more minutes"

    # ------------------------------------------------------------------------- commands

    async def _run_command(self, command: Coroutine[Any, Any, _Listen | None]) -> _Listen | None:
        """Run one command; returns what to listen for next, if anything."""
        task = asyncio.create_task(command)
        self._current = task
        try:
            await asyncio.wait({task})
        finally:
            self._current = None
            if not task.done():  # ``run`` itself was cancelled
                task.cancel()
        if task.cancelled():
            self._ui.stopped()
            return None
        error = task.exception()
        if error is None:
            return task.result()
        self._report(error)
        return None

    async def _text_turn(self, text: str) -> _Listen | None:
        logger.info("Typed message (%d characters)", len(text))
        outcome = await self._interruptible(self._turn.from_text(text))
        if isinstance(outcome, _Interrupted):
            return await self._listen_after(outcome.interruption)
        self._ui.reply_finished(outcome)
        return None

    async def _listen_turn(self, step: _Listen) -> _Listen | None:
        ears = self._ears
        if ears is None:
            return None
        # With barge-in the microphone stays open from listening through the reply.
        async with ears.session() if ears.barge_in else nullcontext():
            return await self._listen_and_reply(ears, step)

    async def _listen_and_reply(self, ears: Ears, step: _Listen) -> _Listen | None:
        timeout_s = None
        more_after = False  # a long conversation listens in stretches (timers go in between)
        if step.until is not None:
            timeout_s = step.until - self._clock()
            if timeout_s <= 0:
                logger.info("The conversation timed out")
                return None
            more_after = step.follow_up and timeout_s > LISTEN_STRETCH_S
            timeout_s = min(timeout_s, LISTEN_STRETCH_S) if more_after else timeout_s
        if not step.quiet:
            logger.info("Listening for a follow-up" if step.follow_up else "Listening")
            self._ui.listening()
        accept = self._for_yarvis(step)
        speculation: Speculation | None = None
        thinking_pause_ms = self._turn_taking.thinking_pause_ms

        def on_pause(audio: AudioArray, control: PauseControl) -> None:
            nonlocal speculation
            if speculation is not None:
                speculation.discard()
            speculation = None
            if not self._turn_taking.speculate:
                return
            speculation = self._turn.speculate(
                audio, woken=step.woken, preface=step.preface, accept=accept
            )
            if thinking_pause_ms is not None:
                patience = thinking_pause_ms

                def adapt(transcript: str) -> None:
                    if not sounds_finished(transcript):  # mid-sentence: give them time
                        control.require_silence_ms(patience)

                speculation.when_transcribed(adapt)

        def on_resume() -> None:
            nonlocal speculation
            if speculation is not None:
                speculation.discard()
                speculation = None

        try:
            hooks = PauseHooks(on_pause=on_pause, on_resume=on_resume)
            started = self._clock() if more_after else 0.0
            utterance = await ears.listen(
                hooks, follow_up=step.follow_up, prefix=step.prefix, timeout_s=timeout_s
            )
            if utterance is None:
                if not step.follow_up:  # silence after a reply simply ends the conversation
                    self._ui.nothing_heard()
                # Listen again only after a full stretch of silence: anything else ending
                # the listening early must not turn into a busy loop.
                full = more_after and self._clock() - started >= LISTEN_STRETCH_S * FULL_STRETCH
                return replace(step, quiet=True) if full else None
            outcome = await self._interruptible(
                self._turn.from_audio(
                    utterance.audio,
                    speech_end=utterance.speech_end,
                    turn_detected=utterance.turn_detected,
                    speculation=speculation,
                    woken=step.woken,
                    preface=step.preface,
                    accept=accept,
                )
            )
        finally:
            if speculation is not None:
                speculation.discard()  # no-op once used; stops it if we were interrupted
        return await self._after_reply(step, outcome)

    async def _after_reply(
        self, step: _Listen, outcome: TurnResult | _Interrupted | None
    ) -> _Listen | None:
        """Decide what comes next: listen again, greet, or go back to waiting."""
        if isinstance(outcome, _Interrupted):
            return await self._listen_after(outcome.interruption)
        if outcome is None:  # nothing intelligible was said
            if step.woken or self._turn.only_wake_phrase:
                return await self._answer_the_wake_word(step)
            return self._nothing_for_yarvis(step)
        self._ui.reply_finished(outcome)
        if ends_conversation(outcome.transcript):
            logger.info("Conversation closed by the user")
        elif self._playback.is_set():
            logger.info("Music or a video started: say 'Hey Yarvis' to talk again")
        else:
            asked = outcome.reply.rstrip().endswith("?") or self._turn.awaiting_confirmation
            return self._open_conversation(asked=asked)
        return None

    def _nothing_for_yarvis(self, step: _Listen) -> _Listen | None:
        """Noise, or people talking to each other: keep an open conversation going."""
        if not step.follow_up:
            self._ui.nothing_heard()
            return None
        if self._turn.overheard:
            logger.info("Heard talking that was not for Yarvis: ignored, nothing sent")
            if self._turn.dismissed:  # "thanks" or "that's all"
                logger.info("Conversation closed by the user")
                return None
        return replace(step, quiet=True) if step.until is not None else None

    def _open_conversation(self, *, asked: bool, at_least: float = 0.0) -> _Listen | None:
        """Keep listening without the wake word after Yarvis spoke.

        For ``follow_up_s`` anything counts if Yarvis ``asked`` something (or if staying
        open is off); then, for ``stay_open_s``, only what is meant for Yarvis.
        """
        rules = self._turn_taking
        lenient = max(rules.follow_up_s, at_least) if asked or rules.stay_open_s <= 0 else 0.0
        total = max(lenient, rules.stay_open_s)
        if total <= 0:
            return None
        now = self._clock()
        return _Listen(
            follow_up=True,
            until=now + total,
            anything_until=now + lenient,
            follow_ups_until=now + rules.follow_up_s,
        )

    def _for_yarvis(self, step: _Listen) -> Callable[[str], bool] | None:
        """How to tell whether speech in an open conversation is meant for Yarvis.

        None (everything is) when the user woke Yarvis or is still making a request.
        """
        if not step.follow_up or step.woken or step.preface or step.anything_until == step.until:
            return None
        names = self._turn.names

        def accept(text: str) -> bool:
            now = self._clock()
            if step.anything_until is not None and now < step.anything_until:
                return True
            recent = step.follow_ups_until is not None and now < step.follow_ups_until
            return meant_for_assistant(text, names=names, just_answered=recent)

        return accept

    async def _answer_the_wake_word(self, step: _Listen) -> _Listen | None:
        """Only "Hey Yarvis" was said: greet back, then listen for the request."""
        if self._greeter is None:
            logger.info("Heard only the wake word; listening for the request")
            return _Listen(prefix=(), preface=step.preface)  # keep what was said since
        greeting = self._greeter.next()
        logger.info("Heard only the wake word; greeting")
        self._ui.said(greeting)
        outcome = await self._interruptible(self._turn.say(greeting))
        if isinstance(outcome, _Interrupted):
            return await self._listen_after(outcome.interruption)
        listen = self._open_conversation(asked=True, at_least=GREETING_WAIT_S)
        return None if listen is None else replace(listen, preface=step.preface)

    async def _interruptible[T](self, reply: Coroutine[Any, Any, T]) -> T | _Interrupted:
        """Run ``reply`` while watching for barge-in; stop it if the user interrupts."""
        ears = self._ears
        if ears is None or not ears.barge_in:
            return await reply
        self._playback.clear()
        async with ears.session():
            replying = asyncio.create_task(reply)
            watching = asyncio.ensure_future(ears.interruption())
            playback = asyncio.ensure_future(self._playback.wait())
            try:
                while True:
                    await asyncio.wait(
                        {replying, watching, playback}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if replying.done() or watching.done() or not playback.done():
                        break
                    watching.cancel()  # music started: from now on, only "Hey Yarvis"
                    await asyncio.wait({watching})
                    watching = asyncio.ensure_future(
                        ears.interruption(wake_word_only=True)
                        if ears.wake_word
                        else asyncio.get_running_loop().create_future()  # nothing can
                    )
                    playback = asyncio.get_running_loop().create_future()  # handled
                if not replying.done() and watching.exception() is None:
                    interruption = watching.result()
                    logger.info(
                        "Interrupted by %s", "the wake word" if interruption.wake_word else "voice"
                    )
                    replying.cancel()
                    await asyncio.wait({replying})
                    return _Interrupted(interruption)
                if watching.done() and (error := watching.exception()) is not None:
                    # A broken microphone must not cut the reply short.
                    logger.warning("Barge-in stopped working: %s", error)
                return await replying
            finally:
                replying.cancel()
                watching.cancel()
                playback.cancel()
                await asyncio.wait({replying, watching, playback})

    async def _listen_after(self, interruption: Interruption) -> _Listen:
        self._ui.interrupted()
        # Cut off before saying anything (e.g. the user paused, then went on): the request
        # is still open, so the next words extend it instead of replacing it.
        preface = self._turn.unanswered or ""
        if preface:
            logger.info("Interrupted before answering; the request continues")
            self._turn.retract(preface)
        if interruption.wake_word:
            await self._turn.chime()  # the cancelled reply has already silenced the speaker
        return _Listen(prefix=interruption.prefix, woken=interruption.wake_word, preface=preface)

    def _report(self, error: BaseException, *, context: str | None = None) -> None:
        if isinstance(error, LLMError | AudioBackendError):
            logger.warning("%s: %s", type(error).__name__, error)
            message = str(error)
        else:
            logger.error("Unexpected error", exc_info=error)
            message = f"Something went wrong ({type(error).__name__}). Details are in the log."
        self._ui.error(f"{context}: {message}" if context else message)


@dataclass(frozen=True, slots=True)
class _Interrupted:
    interruption: Interruption


class AssistantControls:
    """A thread-safe handle on a running ``Assistant``, e.g. for a GUI's stop button."""

    def __init__(self) -> None:
        self._assistant: Assistant | None = None
        self._lock = threading.Lock()

    def attach(self, assistant: Assistant) -> None:
        """Connect to the assistant that is about to run."""
        with self._lock:
            self._assistant = assistant

    def stop(self) -> None:
        """Stop the current command, if any."""
        with self._lock:
            assistant = self._assistant
        if assistant is not None:
            assistant.stop()
