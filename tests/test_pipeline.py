from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fakes import FakeLLM, FakeSpeaker, FakeSTT, FakeTTS, RecordingObserver, TickingClock
from yarvis.agent.conversation import Conversation
from yarvis.llm.base import LLMError
from yarvis.pipeline import Turn, TurnTimings

if TYPE_CHECKING:
    from yarvis.speech.base import AudioArray


def make_turn(
    llm: FakeLLM,
    *,
    stt: FakeSTT | None = None,
    tts: FakeTTS | None = None,
    speaker: FakeSpeaker | None = None,
) -> tuple[Turn, RecordingObserver]:
    observer = RecordingObserver()
    conversation = Conversation(llm, system_prompt=lambda: "S", max_tokens=64, clock=TickingClock())
    turn = Turn(conversation, stt=stt, tts=tts, speaker=speaker, observer=observer)
    return turn, observer


async def test_voice_turn_speaks_sentence_by_sentence() -> None:
    llm = FakeLLM(["Sure. ", "It is **sunny** in Leuven ", "and about twelve degrees today."])
    stt, tts, speaker = FakeSTT("  weather?  "), FakeTTS(), FakeSpeaker()
    turn, observer = make_turn(llm, stt=stt, tts=tts, speaker=speaker)

    result = await turn.from_audio(
        np.zeros(16_000, np.float32), speech_end=99.0, turn_detected=99.7
    )

    assert result is not None
    assert result.transcript == "weather?"
    assert observer.transcripts == ["weather?"]
    assert "".join(observer.deltas) == result.reply
    assert tts.spoken == ["Sure.", "It is sunny in Leuven and about twelve degrees today."]
    assert len(speaker.played) == 2
    assert speaker.drained == 1
    assert observer.audio_started == 1
    assert stt.received == [16_000]


async def test_timings_are_ordered_and_complete() -> None:
    turn, _ = make_turn(FakeLLM(["Hi there."]), stt=FakeSTT(), tts=FakeTTS(), speaker=FakeSpeaker())
    result = await turn.from_audio(np.zeros(10, np.float32), speech_end=99.0, turn_detected=99.7)
    assert result is not None
    t = result.timings
    stamps = [t.speech_end, t.turn_detected, t.transcribed, t.first_token, t.first_sentence]
    stamps += [t.first_audio, t.finished]
    assert all(stamp is not None for stamp in stamps)
    assert stamps == sorted(stamps)  # type: ignore[type-var]
    assert t.end_of_turn_ms == 700
    assert t.voice_to_voice_ms is not None
    assert t.voice_to_voice_ms > t.end_of_turn_ms
    for value in (t.stt_ms, t.llm_ms, t.tts_ms):
        assert value is not None
        assert value >= 0


async def test_empty_transcript_is_ignored() -> None:
    llm = FakeLLM(["never"])
    turn, observer = make_turn(llm, stt=FakeSTT("   "), tts=FakeTTS(), speaker=FakeSpeaker())
    result = await turn.from_audio(np.zeros(10, np.float32), speech_end=1.0, turn_detected=1.5)
    assert result is None
    assert llm.calls == []
    assert observer.transcripts == []


async def test_text_turn_without_voice_output() -> None:
    turn, observer = make_turn(FakeLLM(["Answer one. ", "Answer two."]))
    result = await turn.from_text("  question  ")
    assert result.transcript == "question"
    assert result.reply == "Answer one. Answer two."
    assert result.timings.first_audio is None
    assert result.timings.voice_to_voice_ms is None
    assert observer.deltas == ["Answer one. ", "Answer two."]


async def test_unspeakable_sentences_are_skipped() -> None:
    tts = FakeTTS()
    turn, _ = make_turn(FakeLLM(["```\ncode\n```\n"]), tts=tts, speaker=FakeSpeaker())
    await turn.from_text("show code")
    assert tts.spoken == []


async def test_llm_errors_surface_unwrapped() -> None:
    llm = FakeLLM(["Partial. "], error=LLMError("API key rejected"))
    turn, _ = make_turn(llm, tts=FakeTTS(), speaker=FakeSpeaker())
    with pytest.raises(LLMError, match="API key rejected"):
        await turn.from_text("hi")


async def test_audio_turn_requires_stt() -> None:
    turn, _ = make_turn(FakeLLM())
    with pytest.raises(RuntimeError, match="speech-to-text"):
        await turn.from_audio(np.zeros(1, np.float32), speech_end=0.0, turn_detected=0.0)


def test_missing_timestamps_give_none() -> None:
    timings = TurnTimings()
    assert timings.stt_ms is None
    assert timings.voice_to_voice_ms is None


async def test_stopping_a_turn_silences_the_speaker() -> None:
    class SlowLLM(FakeLLM):
        async def stream(self, messages, *, system, max_tokens):  # type: ignore[override]
            yield "First sentence. "
            await asyncio.sleep(60)
            yield "never"  # pragma: no cover

    class SignallingSpeaker(FakeSpeaker):
        playing = asyncio.Event()

        async def play(self, audio: AudioArray) -> None:
            await super().play(audio)
            self.playing.set()

    speaker = SignallingSpeaker()
    turn, _ = make_turn(SlowLLM(), tts=FakeTTS(), speaker=speaker)
    task = asyncio.create_task(turn.from_text("talk to me"))
    await asyncio.wait_for(speaker.playing.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert speaker.stopped == 1


async def test_errors_also_silence_the_speaker() -> None:
    speaker = FakeSpeaker()
    turn, _ = make_turn(FakeLLM(["Partial. "], error=LLMError("x")), tts=FakeTTS(), speaker=speaker)
    with pytest.raises(LLMError):
        await turn.from_text("hi")
    assert speaker.stopped == 1


# --------------------------------------------------------------------------- speculation


async def test_speculation_is_used_at_end_of_turn() -> None:
    llm = FakeLLM(["Quarter past two."])
    stt, tts, speaker = FakeSTT("what time is it"), FakeTTS(), FakeSpeaker()
    turn, observer = make_turn(llm, stt=stt, tts=tts, speaker=speaker)
    audio = np.zeros(16_000, np.float32)

    speculation = turn.speculate(audio)
    assert await speculation.transcript() == "what time is it"
    assert observer.transcripts == []  # nothing shown until the turn is confirmed

    result = await turn.from_audio(
        audio, speech_end=1.0, turn_detected=1.7, speculation=speculation
    )
    assert result is not None
    assert result.reply == "Quarter past two."
    assert result.timings.speculative
    assert stt.received == [16_000]  # transcribed once, during the pause
    assert len(llm.calls) == 1
    assert observer.transcripts == ["what time is it"]
    assert [m.text for m in turn._conversation.history] == ["what time is it", "Quarter past two."]  # pyright: ignore[reportPrivateUsage]


async def test_discarded_speculation_leaves_no_trace() -> None:
    llm = FakeLLM(["Half a thought."])
    turn, observer = make_turn(llm, stt=FakeSTT("what time"), tts=FakeTTS(), speaker=FakeSpeaker())
    speculation = turn.speculate(np.zeros(8_000, np.float32))
    await speculation.transcript()
    await asyncio.sleep(0)  # let the draft start
    speculation.discard()
    await asyncio.sleep(0)
    assert turn._conversation.history == ()  # pyright: ignore[reportPrivateUsage]
    assert observer.transcripts == []
    assert observer.deltas == []


async def test_discarding_before_transcription_finishes() -> None:
    class SlowSTT(FakeSTT):
        async def transcribe(self, audio: AudioArray) -> str:
            await asyncio.sleep(60)
            return "never"  # pragma: no cover

    turn, _ = make_turn(FakeLLM(), stt=SlowSTT(), tts=FakeTTS(), speaker=FakeSpeaker())
    speculation = turn.speculate(np.zeros(10, np.float32))
    speculation.discard()
    with pytest.raises(asyncio.CancelledError):
        await speculation.transcript()


async def test_speculation_with_empty_transcript() -> None:
    llm = FakeLLM(["unused"])
    turn, _ = make_turn(llm, stt=FakeSTT("   "), tts=FakeTTS(), speaker=FakeSpeaker())
    audio = np.zeros(10, np.float32)
    speculation = turn.speculate(audio)
    assert (
        await turn.from_audio(audio, speech_end=0.0, turn_detected=0.5, speculation=speculation)
        is None
    )
    assert speculation.draft is None
    assert llm.calls == []
