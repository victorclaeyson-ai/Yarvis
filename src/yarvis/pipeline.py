"""One conversational turn: transcript in, spoken reply out, with stage timings.

The reply is pipelined: while the model is still writing, finished sentences are already
being synthesized, and while sentence one plays, sentence two is being synthesized.

    LLM deltas ──▶ SentenceSplitter ──▶ queue ──▶ TTS ──▶ speaker
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from yarvis.speech.sentences import SentenceSplitter, SpeechFilter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from yarvis.agent.conversation import Conversation
    from yarvis.speech.base import AudioArray, AudioSink, SpeechToText, TextToSpeech


def _ms(start: float | None, end: float | None) -> int | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000)


@dataclass(slots=True)
class TurnTimings:
    """Monotonic timestamps (seconds) of the moments that matter for latency."""

    speech_end: float | None = None
    """The user stopped speaking (start of the end-of-turn silence)."""
    turn_detected: float | None = None
    """The end-of-turn silence elapsed; we know the user is done."""
    transcribed: float | None = None
    first_token: float | None = None
    first_sentence: float | None = None
    first_audio: float | None = None
    finished: float | None = None

    @property
    def end_of_turn_ms(self) -> int | None:
        """Time spent waiting to be sure the user finished speaking."""
        return _ms(self.speech_end, self.turn_detected)

    @property
    def stt_ms(self) -> int | None:
        """Speech-to-text time."""
        return _ms(self.turn_detected, self.transcribed)

    @property
    def llm_ms(self) -> int | None:
        """Time to the model's first token."""
        return _ms(self.transcribed, self.first_token)

    @property
    def tts_ms(self) -> int | None:
        """Time to synthesize the first sentence."""
        return _ms(self.first_sentence, self.first_audio)

    @property
    def voice_to_voice_ms(self) -> int | None:
        """From the end of the user's speech to the first reply audio."""
        return _ms(self.speech_end or self.transcribed, self.first_audio)


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What was said and how long it took."""

    transcript: str
    reply: str
    timings: TurnTimings


class TurnObserver(Protocol):
    """Receives progress updates, e.g. to print the conversation."""

    def on_transcript(self, text: str) -> None:
        """The user's words are known."""
        ...

    def on_reply_delta(self, text: str) -> None:
        """A fragment of the reply was generated."""
        ...

    def on_audio_started(self) -> None:
        """The first audio of the reply is being played."""
        ...


class Turn:
    """Runs turns against a conversation, speech engines and a speaker."""

    def __init__(
        self,
        conversation: Conversation,
        *,
        stt: SpeechToText | None,
        tts: TextToSpeech | None,
        speaker: AudioSink | None,
        observer: TurnObserver,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._conversation = conversation
        self._stt = stt
        self._tts = tts
        self._speaker = speaker
        self._observer = observer
        self._clock = clock

    async def from_audio(
        self,
        audio: AudioArray,
        *,
        speech_end: float,
        turn_detected: float,
    ) -> TurnResult | None:
        """Transcribe ``audio`` and reply. Returns None if nothing intelligible was said."""
        if self._stt is None:
            msg = "no speech-to-text engine configured"
            raise RuntimeError(msg)
        timings = TurnTimings(speech_end=speech_end, turn_detected=turn_detected)
        transcript = (await self._stt.transcribe(audio)).strip()
        timings.transcribed = self._clock()
        if not transcript:
            return None
        return await self._reply(transcript, timings)

    async def from_text(self, text: str) -> TurnResult:
        """Reply to typed text."""
        return await self._reply(text.strip(), TurnTimings(transcribed=self._clock()))

    async def _reply(self, transcript: str, timings: TurnTimings) -> TurnResult:
        self._observer.on_transcript(transcript)
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        parts: list[str] = []

        async def produce() -> None:
            splitter = SentenceSplitter()
            try:
                async for delta in self._conversation.respond(transcript):
                    if timings.first_token is None:
                        timings.first_token = self._clock()
                    parts.append(delta)
                    self._observer.on_reply_delta(delta)
                    for sentence in splitter.feed(delta):
                        self._sentence_ready(timings)
                        await sentences.put(sentence)
                if tail := splitter.flush():
                    self._sentence_ready(timings)
                    await sentences.put(tail)
            finally:
                await sentences.put(None)

        async def speak() -> None:
            speakable_text = SpeechFilter()
            while (sentence := await sentences.get()) is not None:
                if self._tts is None or self._speaker is None:
                    continue
                speakable = speakable_text(sentence)
                if not speakable:
                    continue
                audio = await self._tts.synthesize(speakable)
                if timings.first_audio is None:
                    timings.first_audio = self._clock()
                    self._observer.on_audio_started()
                await self._speaker.play(audio)

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(produce())
                group.create_task(speak())
            if self._speaker is not None:
                await self._speaker.drain()
        except ExceptionGroup as failures:
            # Surface the original error (e.g. LLMError) instead of a group wrapper.
            self._silence()
            raise failures.exceptions[0] from None
        except asyncio.CancelledError:
            self._silence()  # stopped by the user: cut the voice off immediately
            raise
        timings.finished = self._clock()
        logger.info(
            "Turn: %s ms to first word (end of turn %s, stt %s, llm %s, tts %s)",
            timings.voice_to_voice_ms,
            timings.end_of_turn_ms,
            timings.stt_ms,
            timings.llm_ms,
            timings.tts_ms,
        )
        return TurnResult(transcript=transcript, reply="".join(parts).strip(), timings=timings)

    def _silence(self) -> None:
        if self._speaker is not None:
            self._speaker.stop()

    def _sentence_ready(self, timings: TurnTimings) -> None:
        if timings.first_sentence is None:
            timings.first_sentence = self._clock()
