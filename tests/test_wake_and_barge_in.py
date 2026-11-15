"""The assistant loop with the wake word and barge-in (microphone side faked)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Never

import numpy as np
import pytest

from fakes import FakeEars, FakeLLM, FakeTTS, TickingClock
from test_listen_and_assistant import RecordingUi, utterance
from yarvis.agent.conversation import Conversation
from yarvis.agent.greetings import Greeter
from yarvis.assistant import Assistant, AssistantControls, TurnTaking
from yarvis.ears import Interruption
from yarvis.listen import MicrophoneSilentError, MicrophoneStalledError
from yarvis.pipeline import Turn

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence

    from fakes import ListenFn
    from yarvis.listen import PauseHooks, Utterance
    from yarvis.llm.base import Message, ToolCall, WebSearch
    from yarvis.pipeline import TurnResult
    from yarvis.speech.base import AudioArray

PREFIX = (np.full(512, 0.1, np.float32), np.full(512, 0.2, np.float32))
CHIME = np.ones(7, np.float32)
NO_SPECULATION = TurnTaking(speculate=False)


class ScriptedSTT:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    async def transcribe(self, audio: AudioArray) -> str:
        return self.texts.pop(0)


class PlayingSpeaker:
    """Audio "plays" until ``stop`` is called or ``finish`` is set."""

    device_name = None

    def __init__(self) -> None:
        self.played: list[int] = []
        self.log: list[str] = []
        self.playing = asyncio.Event()
        self.finish = asyncio.Event()

    async def play(self, audio: AudioArray) -> None:
        self.played.append(len(audio))
        self.log.append(f"play:{len(audio)}")
        self.playing.set()

    async def drain(self) -> None:
        await self.finish.wait()

    def stop(self) -> None:
        self.log.append("stop")
        self.finish.set()


class EventUi(RecordingUi):
    """Records events and lets tests wait for them."""

    def __init__(self) -> None:
        super().__init__()
        self.replied = asyncio.Event()
        self.failed = asyncio.Event()
        self.transcribed = asyncio.Event()

    def on_transcript(self, text: str) -> None:
        super().on_transcript(text)
        self.transcribed.set()

    def reply_finished(self, result: TurnResult) -> None:
        super().reply_finished(result)
        self.replied.set()

    def error(self, message: str) -> None:
        super().error(message)
        self.failed.set()


def make_turn(llm: FakeLLM, stt: ScriptedSTT, ui: EventUi) -> tuple[Turn, PlayingSpeaker]:
    speaker = PlayingSpeaker()
    conversation = Conversation(llm, system_prompt=lambda: "S", max_tokens=32, clock=TickingClock())
    turn = Turn(
        conversation,
        stt=stt,
        tts=FakeTTS(),
        speaker=speaker,
        observer=ui,
        chime=CHIME,
    )
    return turn, speaker


def history(turn: Turn) -> list[str]:
    return [m.text for m in turn._conversation.history]  # pyright: ignore[reportPrivateUsage]


async def after(event: asyncio.Event, *items: str) -> AsyncIterator[str]:
    await event.wait()
    for item in items:
        yield item


async def talk_then_quit(replied: asyncio.Event) -> AsyncIterator[str]:
    """Press "talk" once; quit after the reply."""
    yield ""
    await replied.wait()
    yield "q"


async def forever() -> Never:
    await asyncio.Event().wait()
    raise AssertionError  # pragma: no cover


def requests(*answers: Utterance | None) -> ListenFn:
    """A listen function: answers fresh/prefixed listens in order; follow-ups hear nothing."""
    pending = list(answers)

    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        if follow_up:
            return None
        return pending.pop(0)

    return listen


# --------------------------------------------------------------------------- wake word


async def test_the_wake_word_starts_a_request() -> None:
    ui = EventUi()
    wakes = 0

    async def wake() -> Sequence[AudioArray]:
        nonlocal wakes
        wakes += 1
        return PREFIX if wakes == 1 else await forever()

    ears = FakeEars(requests(utterance()), wake=wake)
    turn, speaker = make_turn(
        FakeLLM(["Noon."]), ScriptedSTT("so anyway. Hey Yarvis, what time is it?"), ui
    )
    speaker.finish.set()
    await Assistant(
        turn=turn, lines=after(ui.replied, "q"), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert ui.events == [
        "ready:True:wake",
        "woke",
        "listening",
        "you:what time is it?",  # everything up to the wake phrase is dropped
        "audio",
        "reply:Noon.",
        "listening",  # follow-up (silence)
        "ready:True:wake",
    ]
    assert ears.prefixes == [PREFIX, None]
    assert speaker.played[0] == len(CHIME)
    assert history(turn) == ["what time is it?", "Noon."]


async def test_typed_input_wins_over_the_wake_word() -> None:
    ui = EventUi()
    cancelled = 0

    async def wake() -> Sequence[AudioArray]:
        nonlocal cancelled
        try:
            return await forever()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    async def typed() -> AsyncIterator[str]:
        yield "hello"

    turn, speaker = make_turn(FakeLLM(["Hi."]), ScriptedSTT(), ui)
    speaker.finish.set()
    await Assistant(turn=turn, lines=typed(), ui=ui, ears=FakeEars(wake=wake)).run()
    assert ui.events == ["ready:True:wake", "you:hello", "audio", "reply:Hi.", "ready:True:wake"]
    assert cancelled == 2  # the wake word stopped listening while the typed text was handled


async def test_only_the_wake_word_keeps_listening_for_the_request() -> None:
    ui = EventUi()
    wakes = 0

    async def wake() -> Sequence[AudioArray]:
        nonlocal wakes
        wakes += 1
        return PREFIX if wakes == 1 else await forever()

    ears = FakeEars(requests(utterance(), utterance()), wake=wake)
    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT("Hey Yarvis.", "what time is it"), ui)
    speaker.finish.set()
    await Assistant(
        turn=turn, lines=after(ui.replied, "q"), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert ears.prefixes == [PREFIX, (), None]  # the second listen keeps the buffered audio
    assert "nothing" not in ui.events
    assert "you:what time is it" in ui.events


async def test_a_bare_wake_word_does_not_hold_the_turn_open() -> None:
    controls: list[int] = []

    class Control:
        def require_silence_ms(self, milliseconds: int) -> None:
            controls.append(milliseconds)

    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        if follow_up:
            return None
        hooks.on_pause(np.zeros(8_000, np.float32), Control())
        await asyncio.sleep(0.02)  # the speculative transcript ("Hey Yarvis.") arrives
        hooks.on_resume()  # ...and the user goes on with the request
        return utterance()

    ui = EventUi()
    wakes = 0

    async def wake() -> Sequence[AudioArray]:
        nonlocal wakes
        wakes += 1
        return PREFIX if wakes == 1 else await forever()

    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT("Hey Yarvis.", "what time is it"), ui)
    speaker.finish.set()
    ears = FakeEars(listen, wake=wake)
    await Assistant(turn=turn, lines=after(ui.replied, "q"), ui=ui, ears=ears).run()
    assert controls == []  # "Hey Yarvis" alone is answered with a greeting, not waited on
    assert "you:what time is it" in ui.events


async def test_a_broken_wake_word_falls_back_to_the_buttons() -> None:
    ui = EventUi()

    async def wake() -> Sequence[AudioArray]:
        raise MicrophoneSilentError

    turn, speaker = make_turn(FakeLLM(["Hi."]), ScriptedSTT(), ui)
    speaker.finish.set()
    await Assistant(
        turn=turn, lines=after(ui.failed, "hello"), ui=ui, ears=FakeEars(wake=wake)
    ).run()
    assert ui.events[0] == "ready:True:wake"
    assert ui.events[1].startswith("error:The wake word stopped working: the microphone is")
    assert ui.events[-1] == "ready:True"  # no longer offering the wake word


# --------------------------------------------------------------------------- barge-in


def interrupt_once(
    speaker: PlayingSpeaker, *, wake_word: bool, prefix: Sequence[AudioArray]
) -> Callable[[], Awaitable[Interruption]]:
    calls = 0

    async def interruption() -> Interruption:
        nonlocal calls
        calls += 1
        if calls > 1:
            return await forever()
        await speaker.playing.wait()  # interrupt while Yarvis is talking
        speaker.playing.clear()
        return Interruption(tuple(prefix), wake_word=wake_word)

    return interruption


async def test_talking_over_the_reply_interrupts_it() -> None:
    ui = EventUi()
    turn, speaker = make_turn(
        FakeLLM(["Once upon a time, ", "a long story."], ["Noon."]),
        ScriptedSTT("tell me a story", "actually, what time is it"),
        ui,
    )
    ears = FakeEars(
        requests(utterance(), utterance()),
        interruption=interrupt_once(speaker, wake_word=False, prefix=PREFIX[:1]),
    )
    await Assistant(
        turn=turn, lines=talk_then_quit(ui.replied), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert ui.events[:5] == [
        "ready:True",
        "listening",
        "you:tell me a story",
        "audio",
        "interrupted",
    ]
    assert "reply:Noon." in ui.events
    assert ears.prefixes[:2] == [None, PREFIX[:1]]  # the interrupting words are kept
    assert "stop" in speaker.log
    assert len(CHIME) not in speaker.played  # no chime: the user is already talking
    said = history(turn)
    assert said[0] == "tell me a story"
    assert said[1].startswith("Once upon a time")  # what was said before the interruption
    assert said[2:] == ["actually, what time is it", "Noon."]


async def test_saying_the_wake_word_interrupts_and_chimes() -> None:
    ui = EventUi()
    turn, speaker = make_turn(
        FakeLLM(["A long answer."], ["Noon."]),
        ScriptedSTT("tell me something", "blah blah. Hey Yarvis, what time is it?"),
        ui,
    )
    ears = FakeEars(
        requests(utterance(), utterance()),
        interruption=interrupt_once(speaker, wake_word=True, prefix=PREFIX),
    )
    await Assistant(
        turn=turn, lines=talk_then_quit(ui.replied), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert ears.prefixes[:2] == [None, PREFIX]
    stop = speaker.log.index("stop")
    assert speaker.log[stop + 1] == f"play:{len(CHIME)}"  # the chime follows the silence
    assert "you:what time is it?" in ui.events  # the wake phrase and the echo are cut


async def test_a_reply_that_finishes_is_not_interrupted() -> None:
    ui = EventUi()
    cancelled = 0

    async def interruption() -> Interruption:
        nonlocal cancelled
        try:
            return await forever()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT("what time is it"), ui)
    speaker.finish.set()
    ears = FakeEars(requests(utterance()), interruption=interruption)
    await Assistant(
        turn=turn,
        lines=talk_then_quit(ui.replied),
        ui=ui,
        ears=ears,
        turn_taking=TurnTaking(speculate=False, follow_up_s=0),
    ).run()
    assert "reply:Noon." in ui.events
    assert "interrupted" not in ui.events
    assert cancelled == 1
    assert ears.sessions == 2  # the whole turn, and the reply within it


async def test_a_broken_barge_in_does_not_cut_the_reply(caplog: pytest.LogCaptureFixture) -> None:
    ui = EventUi()

    async def interruption() -> Interruption:
        raise MicrophoneStalledError

    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT(), ui)
    ears = FakeEars(interruption=interruption)

    async def finish_soon() -> None:
        await speaker.playing.wait()
        await asyncio.sleep(0.01)
        speaker.finish.set()

    helper = asyncio.create_task(finish_soon())
    await Assistant(turn=turn, lines=_typed("what time is it"), ui=ui, ears=ears).run()
    await helper
    assert "reply:Noon." in ui.events
    assert "Barge-in stopped working" in caplog.text


async def test_typed_messages_can_be_interrupted_too() -> None:
    ui = EventUi()
    turn, speaker = make_turn(FakeLLM(["A long answer."], ["Sure."]), ScriptedSTT("never mind"), ui)
    ears = FakeEars(
        requests(utterance()),
        interruption=interrupt_once(speaker, wake_word=False, prefix=PREFIX),
    )
    await Assistant(
        turn=turn, lines=_typed("tell me something"), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert ui.events[:4] == ["ready:True", "you:tell me something", "audio", "interrupted"]
    assert ears.prefixes[0] == PREFIX
    assert "reply:Sure." in ui.events


async def test_stop_during_a_reply_with_barge_in() -> None:
    ui = EventUi()
    cancelled = 0

    async def interruption() -> Interruption:
        nonlocal cancelled
        try:
            return await forever()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    turn, speaker = make_turn(FakeLLM(["A long answer."]), ScriptedSTT(), ui)
    ears = FakeEars(interruption=interruption)
    assistant = Assistant(turn=turn, lines=_typed("tell me something"), ui=ui, ears=ears)
    controls = AssistantControls()
    controls.attach(assistant)

    async def press_stop() -> None:
        await speaker.playing.wait()
        controls.stop()

    helper = asyncio.create_task(press_stop())
    await assistant.run()
    await helper
    assert "stopped" in ui.events
    assert cancelled == 1
    assert "stop" in speaker.log


async def _typed(*items: str) -> AsyncIterator[str]:
    for item in items:
        yield item


async def test_continuing_a_paused_question_extends_it() -> None:
    """The user paused, Yarvis started thinking, the user went on: one question, not two."""
    ui = EventUi()
    turn, speaker = make_turn(
        FakeLLM(["never heard"], ["Rain, as usual."]),
        ScriptedSTT("What's the weather tomorrow", "in Ghent?"),
        ui,
    )
    speaker.finish.set()
    calls = 0

    async def interruption() -> Interruption:
        nonlocal calls
        calls += 1
        if calls > 1:
            return await forever()
        await ui.transcribed.wait()  # Yarvis has the question but has not said a word
        return Interruption(PREFIX, wake_word=False)

    llm = turn._conversation.llm  # pyright: ignore[reportPrivateUsage]
    assert isinstance(llm, FakeLLM)
    ears = FakeEars(requests(utterance(), utterance()), interruption=interruption)
    await Assistant(
        turn=turn, lines=talk_then_quit(ui.replied), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert "you:What's the weather tomorrow in Ghent?" in ui.events
    assert [m.text for m in llm.calls[-1][0]] == ["What's the weather tomorrow in Ghent?"]
    assert history(turn) == ["What's the weather tomorrow in Ghent?", "Rain, as usual."]


async def test_an_answer_already_heard_is_not_repeated() -> None:
    ui = EventUi()
    turn, _ = make_turn(
        FakeLLM(["Once upon a time, ", "a long story."], ["Noon."]),
        ScriptedSTT("tell me a story", "what time is it"),
        ui,
    )
    ears = FakeEars(
        requests(utterance(), utterance()),
        interruption=interrupt_once(turn_speaker(turn), wake_word=False, prefix=PREFIX),
    )
    await Assistant(
        turn=turn, lines=talk_then_quit(ui.replied), ui=ui, ears=ears, turn_taking=NO_SPECULATION
    ).run()
    assert "you:what time is it" in ui.events  # not "tell me a story what time is it"


def turn_speaker(turn: Turn) -> PlayingSpeaker:
    speaker = turn._speaker  # pyright: ignore[reportPrivateUsage]
    assert isinstance(speaker, PlayingSpeaker)
    return speaker


# --------------------------------------------------------------------------- greetings


def in_order(*answers: Utterance | None) -> ListenFn:
    """Answers every listen in order (follow-ups included), then hears nothing."""
    pending = list(answers)

    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        return pending.pop(0) if pending else None

    return listen


def evening_greeter() -> Greeter:
    return Greeter("Victor", clock=lambda: datetime(2026, 11, 15, 20, 0, tzinfo=UTC))


def wake_once() -> Callable[[], Awaitable[Sequence[AudioArray]]]:
    wakes = 0

    async def wake() -> Sequence[AudioArray]:
        nonlocal wakes
        wakes += 1
        return PREFIX if wakes == 1 else await forever()

    return wake


async def test_a_bare_wake_word_is_greeted_back() -> None:
    ui = EventUi()
    turn, speaker = make_turn(
        FakeLLM(["Eight o'clock."]), ScriptedSTT("Hey Yarvis.", "what time is it"), ui
    )
    speaker.finish.set()
    ears = FakeEars(in_order(utterance(), utterance()), wake=wake_once())
    await Assistant(
        turn=turn,
        lines=after(ui.replied, "q"),
        ui=ui,
        ears=ears,
        turn_taking=NO_SPECULATION,
        greeter=evening_greeter(),
    ).run()
    assert ui.events[:5] == [
        "ready:True:wake",
        "woke",
        "listening",
        "said:Good evening, Victor.",
        "listening",
    ]
    assert "you:what time is it" in ui.events
    tts = turn._tts  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tts, FakeTTS)
    assert tts.spoken[0] == "Good evening, Victor."
    # After the greeting: start fresh (drop its echo) and wait long enough for the request.
    assert ears.prefixes[:2] == [PREFIX, None]
    assert ears.timeouts[1] == pytest.approx(15, abs=1)
    assert history(turn) == ["what time is it", "Eight o'clock."]  # greetings are not history


async def test_talking_over_the_greeting_goes_straight_to_the_request() -> None:
    ui = EventUi()
    turn, _ = make_turn(FakeLLM(["Noon."]), ScriptedSTT("Hey Yarvis.", "what time"), ui)
    interrupted = False

    async def interruption() -> Interruption:
        nonlocal interrupted
        if interrupted or not any(event.startswith("said:") for event in ui.events):
            return await forever()
        interrupted = True  # the user talks while Yarvis is saying hello
        return Interruption(PREFIX[:1], wake_word=False)

    ears = FakeEars(in_order(utterance(), utterance()), wake=wake_once(), interruption=interruption)
    await Assistant(
        turn=turn,
        lines=after(ui.replied, "q"),
        ui=ui,
        ears=ears,
        turn_taking=NO_SPECULATION,
        greeter=evening_greeter(),
    ).run()
    assert "interrupted" in ui.events
    assert ears.prefixes[:2] == [PREFIX, PREFIX[:1]]
    assert "reply:Noon." in ui.events


# --------------------------------------------------------------------------- conversation


def always_heard(calls: list[bool]) -> ListenFn:
    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        calls.append(follow_up)
        return utterance()

    return listen


async def test_noise_does_not_end_the_conversation_but_time_does() -> None:
    ui = EventUi()
    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT("what time is it", "", "", ""), ui)
    speaker.finish.set()
    calls: list[bool] = []
    ears = FakeEars(always_heard(calls))
    await Assistant(
        turn=turn,
        lines=talk_then_quit(ui.replied),
        ui=ui,
        ears=ears,
        turn_taking=TurnTaking(speculate=False, follow_up_s=15),
        clock=TickingClock(start=0, step=4),
    ).run()
    # Opened at t=4 for 15 s; noise at t=8, 12 and 16; at t=20 the window has closed.
    assert ears.timeouts == [None, 11, 7, 3]
    assert calls == [False, True, True, True]
    assert "nothing" not in ui.events  # noise during a conversation is not an error


async def test_thanks_closes_the_conversation() -> None:
    ui = EventUi()
    turn, speaker = make_turn(
        FakeLLM(["Noon."], ["Anytime."]), ScriptedSTT("what time is it", "Thanks, Yarvis."), ui
    )
    speaker.finish.set()
    calls: list[bool] = []
    await Assistant(
        turn=turn,
        lines=_typed("", "q"),  # "q" is only read once the conversation is over
        ui=ui,
        ears=FakeEars(always_heard(calls)),
        turn_taking=NO_SPECULATION,
    ).run()
    assert calls == [False, True]  # no third listen after "thanks"
    assert "reply:Anytime." in ui.events


async def test_hey_yarvis_during_a_conversation_is_greeted_too() -> None:
    ui = EventUi()
    turn, speaker = make_turn(
        FakeLLM(["Noon."], ["Rain."]),
        ScriptedSTT("what time is it", "Hey Yarvis.", "and the weather?"),
        ui,
    )
    speaker.finish.set()
    await Assistant(
        turn=turn,
        lines=_typed("", "q"),
        ui=ui,
        ears=FakeEars(in_order(utterance(), utterance(), utterance())),
        turn_taking=NO_SPECULATION,
        greeter=evening_greeter(),
    ).run()
    assert "said:Good evening, Victor." in ui.events
    assert "reply:Rain." in ui.events


# --------------------------------------------------------------------------- timers going off


class AlertUi(EventUi):
    def __init__(self) -> None:
        super().__init__()
        self.alerted = asyncio.Event()

    def alert(self, text: str) -> None:
        super().alert(text)
        self.alerted.set()


async def test_a_due_timer_is_announced_while_waiting() -> None:
    ui = AlertUi()
    turn, speaker = make_turn(FakeLLM(), ScriptedSTT(), ui)
    speaker.finish.set()
    queue: asyncio.Queue[str] = asyncio.Queue()
    ears = FakeEars(in_order())

    async def lines() -> AsyncIterator[str]:
        await ui.alerted.wait()
        yield "q"

    async def ring() -> None:
        await asyncio.sleep(0.01)
        queue.put_nowait("Your 1-minute timer is done.")

    ringing = asyncio.create_task(ring())
    await Assistant(turn=turn, lines=lines(), ui=ui, ears=ears, announcements=queue).run()
    await ringing
    assert ui.events[:3] == ["ready:True", "alert:Your 1-minute timer is done.", "listening"]
    assert speaker.played[0] == len(CHIME)
    tts = turn._tts  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tts, FakeTTS)
    assert tts.spoken == ["Your 1-minute timer is done."]
    assert ears.timeouts == [pytest.approx(15, abs=1)]  # the user can answer ("thanks")


async def test_a_timer_waits_until_yarvis_has_finished_talking() -> None:
    ui = AlertUi()
    turn, speaker = make_turn(FakeLLM(["A long answer."]), ScriptedSTT(), ui)
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def ring_while_talking() -> None:
        await speaker.playing.wait()
        queue.put_nowait("Reminder: call mom.")
        await asyncio.sleep(0.01)
        speaker.finish.set()

    ringing = asyncio.create_task(ring_while_talking())
    await Assistant(
        turn=turn,
        lines=_typed("tell me something", "q"),
        ui=ui,
        ears=FakeEars(in_order()),
        announcements=queue,
    ).run()
    await ringing
    assert ui.events.index("reply:A long answer.") < ui.events.index("alert:Reminder: call mom.")


async def test_a_typed_line_and_a_timer_at_the_same_moment() -> None:
    ui = AlertUi()
    turn, speaker = make_turn(FakeLLM(["Hi."]), ScriptedSTT(), ui)
    speaker.finish.set()
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def lines() -> AsyncIterator[str]:
        queue.put_nowait("Your 2-minute timer is done.")  # rings as the user types
        yield "hello"
        await ui.alerted.wait()
        yield "q"

    await Assistant(
        turn=turn, lines=lines(), ui=ui, ears=FakeEars(in_order()), announcements=queue
    ).run()
    assert "alert:Your 2-minute timer is done." in ui.events  # not lost
    assert "reply:Hi." in ui.events


# --------------------------------------------------------------------------- music starting


async def test_music_started_by_a_tool_cannot_interrupt_and_ends_the_conversation() -> None:
    """The song is not the user: only "Hey Yarvis" interrupts, and nothing more is heard."""
    from test_agent_tools import Play  # noqa: PLC0415 - a fake media tool
    from test_listen_and_assistant import RecordingDucker  # noqa: PLC0415
    from yarvis.agent.toolbox import Toolbox  # noqa: PLC0415
    from yarvis.llm.base import ToolCall  # noqa: PLC0415
    from yarvis.security.gate import Gate  # noqa: PLC0415

    ui = EventUi()
    toolbox = Toolbox([Play()], Gate())
    llm = FakeLLM([ToolCall("1", "play", {"text": "jazz"})], ["Playing jazz."])
    conversation = Conversation(
        llm, system_prompt=lambda: "S", max_tokens=32, clock=TickingClock(), toolbox=toolbox
    )
    speaker = PlayingSpeaker()
    speaker.finish.set()  # each reply plays to the end
    turn = Turn(
        conversation,
        stt=ScriptedSTT("play some jazz"),
        tts=FakeTTS(),
        speaker=speaker,
        observer=ui,
        chime=CHIME,
    )
    listened: list[bool] = []

    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        listened.append(follow_up)
        return utterance()

    ears = FakeEars(listen, wake=forever, interruption=forever)
    assistant = Assistant(
        turn=turn,
        lines=talk_then_quit(ui.replied),
        ui=ui,
        ears=ears,
        turn_taking=NO_SPECULATION,
        ducker=RecordingDucker(ui),
    )
    toolbox.on_playback = assistant.media_started
    await assistant.run()
    assert "reply:Playing jazz." in ui.events
    assert ears.wake_word_only == [False, True]  # voice at first, then only the wake word
    assert listened == [False]  # no follow-up: the music would be heard as the user
    # Down when the conversation starts, and again once the new song is playing.
    assert ui.events.count("music down") == 2
    assert ui.events.index("music down", 2) > ui.events.index("you:play some jazz")
    assert ui.events.count("music up") == 1  # back up once the conversation is over


async def test_after_acting_an_interruption_is_a_new_request() -> None:
    """Cut off after a tool ran: the next words are not glued to the old request (which
    would make the model do it all again: "skip" skipping two songs)."""
    from test_agent_tools import Play, TextArgs  # noqa: PLC0415
    from yarvis.agent.toolbox import Toolbox  # noqa: PLC0415
    from yarvis.llm.base import ToolCall  # noqa: PLC0415
    from yarvis.security.gate import Gate  # noqa: PLC0415

    acted = asyncio.Event()
    blocked = asyncio.Event()

    class Signalling(Play):
        async def run(self, args: TextArgs) -> str:
            acted.set()
            await blocked.wait()  # still busy when the user talks over it
            return "never"  # pragma: no cover

    class ThinkingLLM(FakeLLM):
        """After the tool call, takes its time to answer."""

        async def stream(
            self, messages: Sequence[Message], **kwargs: Any
        ) -> AsyncGenerator[str | ToolCall | WebSearch]:
            if self.calls:
                await blocked.wait()
            async for event in super().stream(messages, **kwargs):
                yield event

    def make(tool: Play) -> Turn:
        llm = ThinkingLLM([ToolCall("1", "play", {"text": "jazz"})], ["Playing jazz."])
        conversation = Conversation(
            llm,
            system_prompt=lambda: "S",
            max_tokens=32,
            clock=TickingClock(),
            toolbox=Toolbox([tool], Gate()),
        )
        speaker = PlayingSpeaker()
        speaker.finish.set()
        return Turn(
            conversation,
            stt=ScriptedSTT("play some jazz"),
            tts=FakeTTS(),
            speaker=speaker,
            observer=EventUi(),
            chime=CHIME,
        )

    # Interrupted while the tool is still running: nothing done yet, the request goes on.
    turn = make(Signalling())
    replying = asyncio.create_task(turn.from_audio(PREFIX[0], speech_end=0, turn_detected=0))
    await acted.wait()
    replying.cancel()
    await asyncio.wait({replying})
    assert turn.unanswered == "play some jazz"

    # Interrupted after the tool ran, before a word was said: a new request.
    acted.clear()

    class Done(Play):
        async def run(self, args: TextArgs) -> str:
            acted.set()
            return await super().run(args)

    turn = make(Done())
    replying = asyncio.create_task(turn.from_audio(PREFIX[0], speech_end=0, turn_detected=0))
    await acted.wait()
    await asyncio.sleep(0)  # the tool returns
    replying.cancel()
    await asyncio.wait({replying})
    assert turn.unanswered is None
    assert history(turn) == ["play some jazz", "[Interrupted. Already done: echo 'jazz'.]"]


# --------------------------------------------------------------------------- staying open

STAY_OPEN = TurnTaking(speculate=False, follow_up_s=15, stay_open_s=300)


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def heard_in_turn(clock: ManualClock, *gaps: float | None) -> ListenFn:
    """Each listen: ``gap`` seconds later someone speaks, or (None) a full stretch of quiet."""
    pending = list(gaps)

    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        if not pending:  # quiet for good: the conversation runs out
            clock.now += 1_000
            return None
        gap = pending.pop(0)
        clock.now += 20.0 if gap is None else gap
        return None if gap is None else utterance()

    return listen


async def test_staying_open_only_answers_what_is_meant_for_yarvis() -> None:
    """Talking to your brother is ignored; "open Spotify" a minute later is not."""
    ui = EventUi()
    llm = FakeLLM(["Noon."], ["Spotify's open."])
    turn, speaker = make_turn(
        llm,
        ScriptedSTT(
            "what time is it",
            "So how was school today?",  # to the brother, 30 s later
            "Do you want to play with me?",
            "Open Spotify.",
        ),
        ui,
    )
    speaker.finish.set()
    clock = ManualClock()
    ears = FakeEars(heard_in_turn(clock, 0, 30, 10, 20))
    await Assistant(
        turn=turn,
        lines=talk_then_quit(ui.replied),  # "q" is read once the conversation is over
        ui=ui,
        ears=ears,
        turn_taking=STAY_OPEN,
        clock=clock,
    ).run()
    said = [event for event in ui.events if event.startswith(("you:", "reply:"))]
    assert said == [
        "you:what time is it",
        "reply:Noon.",
        "you:Open Spotify.",
        "reply:Spotify's open.",
    ]
    assert len(llm.calls) == 2  # nothing about school ever reached the model
    assert ears.timeouts[1:4] == [20, 20, 20]  # listening in stretches


async def test_after_a_question_any_answer_counts() -> None:
    ui = EventUi()
    llm = FakeLLM(["Which one, the red or the blue?"], ["Blue it is."])
    turn, speaker = make_turn(llm, ScriptedSTT("pick a colour", "the blue one"), ui)
    speaker.finish.set()
    clock = ManualClock()
    ears = FakeEars(heard_in_turn(clock, 0, 5))
    await Assistant(
        turn=turn,
        lines=talk_then_quit(ui.replied),
        ui=ui,
        ears=ears,
        turn_taking=STAY_OPEN,
        clock=clock,
    ).run()
    assert "you:the blue one" in ui.events  # not a command, but it answers the question


async def test_overheard_thanks_ends_it_quietly_and_time_does_too() -> None:
    ui = EventUi()
    llm = FakeLLM(["Noon."])
    turn, speaker = make_turn(llm, ScriptedSTT("what time is it", "Thanks, man."), ui)
    speaker.finish.set()
    clock = ManualClock()
    ears = FakeEars(heard_in_turn(clock, 0, 40))
    await Assistant(
        turn=turn, lines=talk_then_quit(ui.replied), ui=ui, ears=ears, turn_taking=STAY_OPEN,
        clock=clock,
    ).run()  # fmt: skip
    assert len(llm.calls) == 1  # no "You're welcome" to someone else's thanks
    assert "you:Thanks, man." not in ui.events

    ui = EventUi()
    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT("what time is it"), ui)
    speaker.finish.set()
    clock = ManualClock()
    ears = FakeEars(heard_in_turn(clock, 0, *[None] * 15))  # 15 quiet stretches = 300 s
    await Assistant(
        turn=turn, lines=talk_then_quit(ui.replied), ui=ui, ears=ears, turn_taking=STAY_OPEN,
        clock=clock,
    ).run()  # fmt: skip
    assert len(ears.timeouts) == 16  # the request, then stretches until five minutes passed
    assert ui.events.count("listening") == 2  # quiet stretches are not announced again


async def test_a_timer_goes_off_while_the_conversation_stays_open() -> None:
    ui = EventUi()
    turn, speaker = make_turn(FakeLLM(["Noon."]), ScriptedSTT("what time is it"), ui)
    speaker.finish.set()
    clock = ManualClock()
    announcements: asyncio.Queue[str] = asyncio.Queue()
    stretches = 0

    async def listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        nonlocal stretches
        if not follow_up:
            return utterance()
        stretches += 1
        if stretches == 2:
            announcements.put_nowait("Your pasta timer is done.")
        clock.now += 20 if stretches <= 3 else 1_000  # quiet, then quiet for good
        return None

    await Assistant(
        turn=turn,
        lines=talk_then_quit(ui.replied),
        ui=ui,
        ears=FakeEars(listen),
        turn_taking=STAY_OPEN,
        announcements=announcements,
        clock=clock,
    ).run()
    assert "alert:Your pasta timer is done." in ui.events


async def test_overheard_speech_is_never_drafted_or_shown() -> None:
    """Speculation starts during pauses: it must not send overheard talk to the model."""
    ui = EventUi()
    llm = FakeLLM(["never"])
    turn, _ = make_turn(llm, ScriptedSTT("So how was school today?", "Open Spotify."), ui)
    not_for_me = turn.speculate(PREFIX[0], accept=lambda text: "spotify" in text.lower())
    assert await not_for_me.transcript() == ""
    assert not_for_me.draft is None and turn.overheard and not turn.dismissed
    for_me = turn.speculate(PREFIX[0], accept=lambda text: "spotify" in text.lower())
    assert await for_me.transcript() == "Open Spotify."
    assert for_me.draft is not None and not turn.overheard
    for_me.discard()
    assert not any(event.startswith("you:") for event in ui.events)
