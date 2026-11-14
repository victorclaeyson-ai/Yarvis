from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fakes import FakeLLM, FakeSpeaker, FakeSTT, FakeTTS, TickingClock
from test_speech_engines import frames, speech_clip
from yarvis.agent.conversation import Conversation
from yarvis.assistant import Assistant, AssistantControls
from yarvis.audio.devices import AudioBackendError
from yarvis.listen import MicrophoneSilentError, MicrophoneStalledError, Utterance, listen
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


class RecordingUi:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.levels: list[float] = []

    def ready(self, *, voice: bool) -> None:
        self.events.append(f"ready:{voice}")

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
    conversation = Conversation(llm, system_prompt=lambda: "S", max_tokens=32)
    return Turn(
        conversation,
        stt=FakeSTT(stt_text),
        tts=FakeTTS(),
        speaker=FakeSpeaker(),
        observer=RecordingUi(),
        clock=TickingClock(),
    )


def utterance() -> Utterance:
    return Utterance(audio=np.zeros(8000, np.float32), speech_end=1.0, turn_detected=1.7)


async def test_enter_listens_and_text_is_sent_directly() -> None:
    ui = RecordingUi()
    heard: list[bool] = []

    async def fake_listen() -> Utterance | None:
        heard.append(True)
        return utterance()

    llm = FakeLLM(["It is noon."], ["Hi Victor."])
    assistant = Assistant(
        turn=make_turn(llm),
        lines=lines("\n", "hello\n", "q\n", "never\n"),
        ui=ui,
        listen=fake_listen,
    )
    await assistant.run()
    assert heard == [True]
    assert ui.events == [
        "ready:True",
        "listening",
        "reply:It is noon.",
        "ready:True",
        "reply:Hi Victor.",
        "ready:True",
    ]


async def test_nothing_heard_cases() -> None:
    ui = RecordingUi()
    results: list[Utterance | None] = [None, utterance()]

    async def fake_listen() -> Utterance | None:
        return results.pop(0)

    turn = make_turn(FakeLLM(), stt_text="  ")
    await Assistant(turn=turn, lines=lines("", ""), ui=ui, listen=fake_listen).run()
    assert ui.events.count("nothing") == 2


async def test_errors_are_shown_and_the_loop_continues() -> None:
    ui = RecordingUi()

    async def broken_listen() -> Utterance | None:
        raise AudioBackendError("mic unplugged")

    llm = FakeLLM([], error=LLMError("out of credits"))
    assistant = Assistant(
        turn=make_turn(llm), lines=lines("", "hello", "exit"), ui=ui, listen=broken_listen
    )
    await assistant.run()
    assert "error:mic unplugged" in ui.events
    assert "error:out of credits" in ui.events


async def test_text_only_mode_ignores_empty_lines() -> None:
    ui = RecordingUi()
    await Assistant(turn=make_turn(FakeLLM()), lines=lines("", "  "), ui=ui, listen=None).run()
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

    async def endless_listen() -> Utterance | None:
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
        turn=make_turn(FakeLLM(["Hi."])), lines=commands(), ui=ui, listen=endless_listen
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
    assistant = Assistant(turn=make_turn(FakeLLM()), lines=lines(), ui=RecordingUi(), listen=None)
    assistant.stop()  # not running yet
    await assistant.run()
    assistant.stop()  # loop finished
    assistant._cancel_current()  # pyright: ignore[reportPrivateUsage]


async def test_unexpected_errors_are_reported_and_the_loop_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ui = RecordingUi()

    async def buggy_listen() -> Utterance | None:
        raise ZeroDivisionError

    llm = FakeLLM(["Still here."])
    await Assistant(turn=make_turn(llm), lines=lines("", "hi"), ui=ui, listen=buggy_listen).run()
    assert any(e.startswith("error:Something went wrong (ZeroDivisionError)") for e in ui.events)
    assert "reply:Still here." in ui.events
    assert "Unexpected error" in caplog.text


async def test_cancelling_the_whole_loop_cancels_the_command() -> None:
    started = asyncio.Event()

    async def endless_listen() -> Utterance | None:
        started.set()
        await asyncio.sleep(60)
        return None  # pragma: no cover

    assistant = Assistant(
        turn=make_turn(FakeLLM()), lines=lines(""), ui=RecordingUi(), listen=endless_listen
    )
    runner = asyncio.create_task(assistant.run())
    await asyncio.wait_for(started.wait(), 1)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
