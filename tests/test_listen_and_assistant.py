from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fakes import FakeEars, FakeLLM, FakeSpeaker, FakeSTT, FakeTTS, TickingClock
from test_speech_engines import frames, speech_clip
from yarvis.agent.conversation import Conversation
from yarvis.assistant import Assistant, AssistantControls, TurnTaking
from yarvis.audio.devices import AudioBackendError
from yarvis.listen import (
    MicrophoneSilentError,
    MicrophoneStalledError,
    PauseHooks,
    Utterance,
    listen,
)
from yarvis.llm.base import LLMError
from yarvis.pipeline import Turn, TurnResult
from yarvis.speech.endpointing import EndpointerSettings
from yarvis.speech.silero import SileroVAD

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from yarvis.speech.base import AudioArray


async def stream(items: Iterable[AudioArray]) -> AsyncIterator[AudioArray]:
    for item in items:
        yield item


# --------------------------------------------------------------------------- listen


async def test_listen_captures_real_speech() -> None:
    started: list[bool] = []
    levels: list[float] = []
    utterance = await listen(
        stream(frames(speech_clip())),
        SileroVAD(),
        EndpointerSettings(),
        on_speech_start=lambda: started.append(True),
        on_level=levels.append,
        clock=lambda: 50.0,
    )
    assert utterance is not None
    assert started == [True]
    assert levels[0] == 0.0  # leading silence
    assert max(levels) > 0.01
    assert utterance.turn_detected == 50.0
    assert utterance.speech_end == pytest.approx(50.0 - 0.704)
    assert len(utterance.audio) > 2.5 * 16_000


async def test_listen_without_speech_returns_none() -> None:
    noise = [np.full(512, 0.001, np.float32)] * 300
    settings = EndpointerSettings(no_speech_timeout_ms=3_000)
    assert await listen(stream(noise), SileroVAD(), settings) is None


async def test_listen_detects_a_muted_microphone() -> None:
    silence = [np.zeros(512, np.float32)] * 100
    with pytest.raises(MicrophoneSilentError, match="Privacy & Security"):
        await listen(stream(silence), SileroVAD(), EndpointerSettings())


async def test_listen_returns_none_when_the_stream_ends() -> None:
    assert await listen(stream([]), SileroVAD(), EndpointerSettings()) is None


# --------------------------------------------------------------------------- assistant loop


class FakeControl:
    def __init__(self) -> None:
        self.required: list[int] = []

    def require_silence_ms(self, milliseconds: int) -> None:
        self.required.append(milliseconds)


class RecordingUi:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.levels: list[float] = []

    def ready(self, *, voice: bool, wake_word: bool = False) -> None:
        self.events.append(f"ready:{voice}" + (":wake" if wake_word else ""))

    def woke(self) -> None:
        self.events.append("woke")

    def interrupted(self) -> None:
        self.events.append("interrupted")

    def said(self, text: str) -> None:
        self.events.append(f"said:{text}")

    def alert(self, text: str) -> None:
        self.events.append(f"alert:{text}")

    def activity(self, text: str) -> None:
        self.events.append(f"did:{text}")

    def listening(self) -> None:
        self.events.append("listening")

    def speech_started(self) -> None:
        self.events.append("speech")

    def level(self, value: float) -> None:
        self.levels.append(value)

    def on_audio_started(self) -> None:
        self.events.append("audio")

    def nothing_heard(self) -> None:
        self.events.append("nothing")

    def reply_finished(self, result: TurnResult) -> None:
        self.events.append(f"reply:{result.reply}")

    def error(self, message: str) -> None:
        self.events.append(f"error:{message}")

    def stopped(self) -> None:
        self.events.append("stopped")

    def on_transcript(self, text: str) -> None:
        self.events.append(f"you:{text}")

    def on_reply_delta(self, text: str) -> None:
        pass


async def lines(*items: str) -> AsyncIterator[str]:
    for item in items:
        yield item


def make_turn(llm: FakeLLM, stt_text: str = "what time is it") -> Turn:
    conversation = Conversation(llm, system_prompt=lambda: "S", max_tokens=32, clock=TickingClock())
    return Turn(
        conversation,
        stt=FakeSTT(stt_text),
        tts=FakeTTS(),
        speaker=FakeSpeaker(),
        observer=RecordingUi(),
    )


def utterance() -> Utterance:
    return Utterance(audio=np.zeros(8000, np.float32), speech_end=1.0, turn_detected=1.7)


async def test_enter_listens_and_text_is_sent_directly() -> None:
    ui = RecordingUi()
    heard: list[bool] = []

    async def fake_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        heard.append(follow_up)
        return None if follow_up else utterance()

    llm = FakeLLM(["It is noon."], ["Hi Victor."])
    assistant = Assistant(
        turn=make_turn(llm),
        lines=lines("\n", "hello\n", "q\n", "never\n"),
        ui=ui,
        ears=FakeEars(fake_listen),
    )
    await assistant.run()
    assert heard == [False, True]  # the key press, then the automatic follow-up
    assert ui.events == [
        "ready:True",
        "listening",
        "reply:It is noon.",
        "listening",
        "ready:True",
        "reply:Hi Victor.",
        "ready:True",
    ]


async def test_nothing_heard_cases() -> None:
    ui = RecordingUi()
    results: list[Utterance | None] = [None, utterance()]

    async def fake_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        return None if follow_up else results.pop(0)

    turn = make_turn(FakeLLM(), stt_text="  ")
    await Assistant(turn=turn, lines=lines("", ""), ui=ui, ears=FakeEars(fake_listen)).run()
    assert ui.events.count("nothing") == 2


async def test_errors_are_shown_and_the_loop_continues() -> None:
    ui = RecordingUi()

    async def broken_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        raise AudioBackendError("mic unplugged")

    llm = FakeLLM([], error=LLMError("out of credits"))
    assistant = Assistant(
        turn=make_turn(llm), lines=lines("", "hello", "exit"), ui=ui, ears=FakeEars(broken_listen)
    )
    await assistant.run()
    assert "error:mic unplugged" in ui.events
    assert "error:out of credits" in ui.events


async def test_text_only_mode_ignores_empty_lines() -> None:
    ui = RecordingUi()
    await Assistant(turn=make_turn(FakeLLM()), lines=lines("", "  "), ui=ui, ears=None).run()
    assert ui.events == ["ready:False", "ready:False", "ready:False"]


async def test_listen_detects_a_stalled_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yarvis.listen.STALLED_MIC_SECONDS", 0.05)

    async def stalled() -> AsyncIterator[AudioArray]:
        yield np.full(512, 0.01, np.float32)
        await asyncio.sleep(10)
        yield np.zeros(512, np.float32)  # pragma: no cover

    with pytest.raises(MicrophoneStalledError, match="not delivering"):
        await listen(stalled(), SileroVAD(), EndpointerSettings())


async def test_stop_cancels_listening_and_the_loop_continues() -> None:
    ui = RecordingUi()
    started = asyncio.Event()

    async def endless_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        started.set()
        await asyncio.sleep(60)
        return None  # pragma: no cover

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def commands() -> AsyncIterator[str]:
        while (item := await queue.get()) is not None:
            yield item

    controls = AssistantControls()
    controls.stop()  # nothing attached yet: harmless
    assistant = Assistant(
        turn=make_turn(FakeLLM(["Hi."])), lines=commands(), ui=ui, ears=FakeEars(endless_listen)
    )
    controls.attach(assistant)
    runner = asyncio.create_task(assistant.run())
    await queue.put("")
    await asyncio.wait_for(started.wait(), 1)
    controls.stop()
    await queue.put("hello")
    await queue.put(None)
    await asyncio.wait_for(runner, 2)
    assert "stopped" in ui.events
    assert "reply:Hi." in ui.events


async def test_stop_without_a_running_command_is_harmless() -> None:
    assistant = Assistant(turn=make_turn(FakeLLM()), lines=lines(), ui=RecordingUi(), ears=None)
    assistant.stop()  # not running yet
    await assistant.run()
    assistant.stop()  # loop finished
    assistant._cancel_current()  # pyright: ignore[reportPrivateUsage]


async def test_unexpected_errors_are_reported_and_the_loop_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ui = RecordingUi()

    async def buggy_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        raise ZeroDivisionError

    llm = FakeLLM(["Still here."])
    await Assistant(
        turn=make_turn(llm), lines=lines("", "hi"), ui=ui, ears=FakeEars(buggy_listen)
    ).run()
    assert any(e.startswith("error:Something went wrong (ZeroDivisionError)") for e in ui.events)
    assert "reply:Still here." in ui.events
    assert "Unexpected error" in caplog.text


async def test_cancelling_the_whole_loop_cancels_the_command() -> None:
    started = asyncio.Event()

    async def endless_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        started.set()
        await asyncio.sleep(60)
        return None  # pragma: no cover

    assistant = Assistant(
        turn=make_turn(FakeLLM()), lines=lines(""), ui=RecordingUi(), ears=FakeEars(endless_listen)
    )
    runner = asyncio.create_task(assistant.run())
    await asyncio.wait_for(started.wait(), 1)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner


async def test_assistant_speculates_on_pauses_and_discards_on_resume() -> None:
    ui = RecordingUi()
    first, second = np.zeros(8_000, np.float32), np.zeros(12_000, np.float32)

    async def talker(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        if follow_up:
            return None
        hooks.on_pause(first, FakeControl())  # a thinking pause...
        await asyncio.sleep(0.01)
        hooks.on_resume()  # ...then more words
        hooks.on_pause(second, FakeControl())
        await asyncio.sleep(0.01)
        return Utterance(audio=second, speech_end=1.0, turn_detected=1.7)

    llm = FakeLLM(["discarded draft"], ["The real answer."])
    turn = make_turn(llm, stt_text="the full question")
    await Assistant(turn=turn, lines=lines(""), ui=ui, ears=FakeEars(talker)).run()
    stt = turn._stt  # pyright: ignore[reportPrivateUsage]
    assert isinstance(stt, FakeSTT)
    assert stt.received == [8_000, 12_000]  # one speculation per pause, none after
    assert "reply:The real answer." in ui.events
    history = [m.text for m in turn._conversation.history]  # pyright: ignore[reportPrivateUsage]
    assert history == ["the full question", "The real answer."]


async def test_speculation_can_be_disabled() -> None:
    ui = RecordingUi()

    async def talker(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        if follow_up:
            return None
        hooks.on_pause(np.zeros(100, np.float32), FakeControl())
        return Utterance(audio=np.zeros(200, np.float32), speech_end=1.0, turn_detected=1.7)

    turn = make_turn(FakeLLM(["ok"]))
    await Assistant(
        turn=turn,
        lines=lines(""),
        ui=ui,
        ears=FakeEars(talker),
        turn_taking=TurnTaking(speculate=False),
    ).run()
    stt = turn._stt  # pyright: ignore[reportPrivateUsage]
    assert isinstance(stt, FakeSTT)
    assert stt.received == [200]  # transcribed only after the turn ended


async def test_real_speech_triggers_a_pause_before_the_end_of_turn() -> None:
    events: list[str] = []
    hooks = PauseHooks(
        on_pause=lambda audio, control: events.append(f"pause:{len(audio) > 30_000}"),
        on_resume=lambda: events.append("resume"),
    )
    utterance = await listen(
        stream(frames(speech_clip())), SileroVAD(), EndpointerSettings(), hooks=hooks
    )
    assert utterance is not None
    assert events[-1] == "pause:True"  # the last pause holds the whole sentence


# --------------------------------------------------------------------------- turn-taking


def pausing_talker(controls: list[FakeControl]):
    async def talker(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        if follow_up:
            return None
        control = FakeControl()
        controls.append(control)
        hooks.on_pause(np.zeros(8_000, np.float32), control)
        await asyncio.sleep(0.02)  # the speculative transcript arrives during the pause
        return Utterance(audio=np.zeros(8_000, np.float32), speech_end=1.0, turn_detected=1.9)

    return talker


async def test_unfinished_sentences_get_more_time() -> None:
    controls: list[FakeControl] = []
    turn = make_turn(FakeLLM(["Go on."]), stt_text="So I was thinking about the")
    await Assistant(
        turn=turn, lines=lines(""), ui=RecordingUi(), ears=FakeEars(pausing_talker(controls))
    ).run()
    assert controls[0].required == [2_000]


async def test_finished_sentences_get_the_normal_wait() -> None:
    controls: list[FakeControl] = []
    turn = make_turn(FakeLLM(["Noon."]), stt_text="What time is it?")
    await Assistant(
        turn=turn, lines=lines(""), ui=RecordingUi(), ears=FakeEars(pausing_talker(controls))
    ).run()
    assert controls[0].required == []


async def test_adaptive_waiting_can_be_disabled() -> None:
    controls: list[FakeControl] = []
    turn = make_turn(FakeLLM(["Go on."]), stt_text="So I was thinking about the")
    assistant = Assistant(
        turn=turn,
        lines=lines(""),
        ui=RecordingUi(),
        ears=FakeEars(pausing_talker(controls)),
        turn_taking=TurnTaking(thinking_pause_ms=None),
    )
    await assistant.run()
    assert controls[0].required == []


async def test_follow_up_continues_the_conversation_without_a_key_press() -> None:
    ui = RecordingUi()
    calls: list[bool] = []
    answers = [utterance(), utterance(), None]  # question, follow-up question, then silence

    async def talker(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        calls.append(follow_up)
        return answers.pop(0)

    llm = FakeLLM(["First answer."], ["Second answer."])
    await Assistant(turn=make_turn(llm), lines=lines(""), ui=ui, ears=FakeEars(talker)).run()
    assert calls == [False, True, True]
    assert ui.events.count("listening") == 3
    assert "reply:First answer." in ui.events
    assert "reply:Second answer." in ui.events
    assert "nothing" not in ui.events  # silence after a reply is not an error
    assert ui.events.count("ready:True") == 2  # before the first question, and at the end


async def test_follow_up_can_be_disabled() -> None:
    calls: list[bool] = []

    async def talker(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        calls.append(follow_up)
        return utterance()

    turn = make_turn(FakeLLM(["Answer."]))
    assistant = Assistant(
        turn=turn,
        lines=lines(""),
        ui=RecordingUi(),
        ears=FakeEars(talker),
        turn_taking=TurnTaking(follow_up_s=0),
    )
    await assistant.run()
    assert calls == [False]


async def test_typed_messages_do_not_trigger_follow_up() -> None:
    calls: list[bool] = []

    async def talker(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        calls.append(follow_up)
        return None  # pragma: no cover

    await Assistant(
        turn=make_turn(FakeLLM(["Hi."])),
        lines=lines("hello"),
        ui=RecordingUi(),
        ears=FakeEars(talker),
    ).run()
    assert calls == []


class RecordingDucker:
    def __init__(self, ui: RecordingUi) -> None:
        self.ui = ui

    async def duck(self) -> None:
        self.ui.events.append("music down")

    async def restore(self) -> None:
        self.ui.events.append("music up")


async def test_music_is_turned_down_for_the_whole_conversation() -> None:
    ui = RecordingUi()

    async def fake_listen(hooks: PauseHooks, *, follow_up: bool = False) -> Utterance | None:
        return None if follow_up else utterance()

    assistant = Assistant(
        turn=make_turn(FakeLLM(["It is noon."], ["Typed."])),
        lines=lines("", "typed", "q"),
        ui=ui,
        ears=FakeEars(fake_listen),
        ducker=RecordingDucker(ui),
    )
    await assistant.run()
    assert ui.events == [
        "ready:True",
        "music down",
        "listening",
        "reply:It is noon.",
        "listening",  # the follow-up window: still quiet
        "music up",  # the conversation is over
        "ready:True",
        "reply:Typed.",  # typing needs no quiet
        "ready:True",
    ]
