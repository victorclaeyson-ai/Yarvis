"""One conversational turn: transcript in, spoken reply out, with stage timings.

The reply is pipelined: while the model is still writing, finished sentences are already
being synthesized, and while sentence one plays, sentence two is being synthesized.

    LLM deltas ──▶ SentenceSplitter ──▶ queue ──▶ TTS ──▶ speaker

**Speculation.** When the user pauses, we do not yet know whether they are done. Waiting
for the full end-of-turn silence before doing anything wastes that time, so at the start
of a pause a ``Speculation`` already transcribes the audio and asks Claude for a draft
reply. If the pause turns out to be the end of the turn, the reply is (nearly) ready; if
the user keeps talking, the speculation is discarded and nothing is remembered.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

from yarvis.speech.sentences import SentenceSplitter, SpeechFilter
from yarvis.speech.turn_taking import DEFAULT_WAKE_NAMES, strip_wake_phrase

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from yarvis.agent.conversation import Conversation, Draft
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
    stt_started: float | None = None
    transcribed: float | None = None
    llm_started: float | None = None
    first_token: float | None = None
    first_sentence: float | None = None
    first_audio: float | None = None
    finished: float | None = None
    speculative: bool = False
    """Whether transcription and the reply were started during the pause."""

    @property
    def end_of_turn_ms(self) -> int | None:
        """Time spent waiting to be sure the user finished speaking."""
        return _ms(self.speech_end, self.turn_detected)

    @property
    def stt_ms(self) -> int | None:
        """Speech-to-text time."""
        return _ms(self.stt_started, self.transcribed)

    @property
    def llm_ms(self) -> int | None:
        """Time to the model's first token."""
        return _ms(self.llm_started, self.first_token)

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


class Speculation:
    """Transcription plus a draft reply, started at a pause before the turn is confirmed."""

    def __init__(
        self,
        stt: SpeechToText,
        conversation: Conversation,
        audio: AudioArray,
        clock: Callable[[], float],
        clean: Callable[[str], str] = str.strip,
    ) -> None:
        self.started = clock()
        self.transcribed: float | None = None
        self.llm_started: float | None = None
        self.draft: Draft | None = None
        self._stt = stt
        self._conversation = conversation
        self._clock = clock
        self._clean = clean
        self._task = asyncio.create_task(self._run(audio))

    async def _run(self, audio: AudioArray) -> str:
        text = self._clean(await self._stt.transcribe(audio))
        self.transcribed = self._clock()
        if text:
            self.llm_started = self._clock()
            self.draft = self._conversation.draft(text)
        return text

    async def transcript(self) -> str:
        """Wait for the transcription."""
        return await self._task

    def when_transcribed(self, callback: Callable[[str], None]) -> None:
        """Call ``callback(transcript)`` as soon as the transcript is known."""

        def done(task: asyncio.Task[str]) -> None:
            if not task.cancelled() and task.exception() is None:
                callback(task.result())

        self._task.add_done_callback(done)

    def discard(self) -> None:
        """Abandon this speculation (the user kept talking, or the turn was stopped)."""
        if not self._task.done():
            self._task.cancel()
        if self.draft is not None:
            self.draft.discard()


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
        chime: AudioArray | None = None,
        names: Sequence[str] = DEFAULT_WAKE_NAMES,
    ) -> None:
        self._conversation = conversation
        self._stt = stt
        self._tts = tts
        self._speaker = speaker
        self._observer = observer
        self._chime = chime
        self._names = tuple(names)
        self.unanswered: str | None = None
        """The request of a reply that was stopped before any of it was spoken."""
        self.only_wake_phrase = False
        """The latest transcript held nothing but "Hey Yarvis" (the user wants attention)."""
        self._phrases: dict[str, AudioArray] = {}
        # One time source for every timestamp, shared with the conversation's drafts.
        self._clock = conversation.clock

    def speculate(
        self, audio: AudioArray, *, woken: bool = False, preface: str = ""
    ) -> Speculation:
        """Start transcribing and drafting a reply for ``audio`` right now."""
        clean = self._cleaner(woken=woken, preface=preface)
        return Speculation(self._require_stt(), self._conversation, audio, self._clock, clean)

    async def prepare(self, phrases: Iterable[str]) -> None:
        """Synthesize fixed phrases ahead of time, so ``say`` can play them instantly."""
        if self._tts is None:
            return
        for phrase in phrases:
            if phrase not in self._phrases:
                self._phrases[phrase] = await self._tts.synthesize(phrase)

    async def say(self, phrase: str) -> None:
        """Speak a fixed phrase (no Claude), e.g. a greeting; returns when it has been heard."""
        if self._tts is None or self._speaker is None:
            return
        audio = self._phrases.get(phrase)
        if audio is None:
            audio = self._phrases[phrase] = await self._tts.synthesize(phrase)
        try:
            await self._speaker.play(audio)
            await self._speaker.drain()
        except asyncio.CancelledError:
            self._silence()
            raise

    async def chime(self) -> None:
        """Play the "I'm listening" sound (if there is a speaker and the chime is on)."""
        if self._speaker is not None and self._chime is not None:
            await self._speaker.play(self._chime)

    async def from_audio(
        self,
        audio: AudioArray,
        *,
        speech_end: float,
        turn_detected: float,
        speculation: Speculation | None = None,
        woken: bool = False,
        preface: str = "",
    ) -> TurnResult | None:
        """Reply to ``audio``. Returns None if nothing intelligible was said.

        If ``speculation`` was started for this same audio during the pause, its transcript
        and draft reply are used instead of starting from scratch. ``woken`` means the audio
        starts with the wake word's pre-roll, so "Hey Yarvis" is cut from the transcript.
        ``preface`` is an earlier request that was never answered (see ``unanswered``); the
        new words are added to it.
        """
        self.unanswered = None
        timings = TurnTimings(speech_end=speech_end, turn_detected=turn_detected)
        draft: Draft | None = None
        if speculation is not None:
            transcript = await speculation.transcript()
            timings.speculative = True
            timings.stt_started = speculation.started
            timings.transcribed = speculation.transcribed
            timings.llm_started = speculation.llm_started
            draft = speculation.draft
        else:
            stt = self._require_stt()
            timings.stt_started = self._clock()
            transcript = self._cleaner(woken=woken, preface=preface)(await stt.transcribe(audio))
            timings.transcribed = self._clock()
        if not transcript:
            return None
        return await self._reply(transcript, timings, draft)

    async def from_text(self, text: str) -> TurnResult:
        """Reply to typed text."""
        self.unanswered = None
        now = self._clock()
        return await self._reply(text.strip(), TurnTimings(transcribed=now))

    def retract(self, text: str) -> None:
        """Forget an exchange the user never heard (see ``Conversation.retract``)."""
        self._conversation.retract(text)

    def _cleaner(self, *, woken: bool, preface: str = "") -> Callable[[str], str]:
        """Transcript clean-up: drop the wake phrase so only the request reaches Claude."""
        strip = partial(strip_wake_phrase, anywhere=woken, names=self._names)

        def clean(text: str) -> str:
            request = strip(text)
            self.only_wake_phrase = bool(text.strip()) and not request
            return f"{preface} {request}".strip() if preface else request

        return clean

    def _require_stt(self) -> SpeechToText:
        if self._stt is None:
            msg = "no speech-to-text engine configured"
            raise RuntimeError(msg)
        return self._stt

    async def _reply(
        self, transcript: str, timings: TurnTimings, draft: Draft | None = None
    ) -> TurnResult:
        if draft is None:
            timings.llm_started = self._clock()
            draft = self._conversation.draft(transcript)
        self._observer.on_transcript(transcript)
        self.unanswered = transcript
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        parts: list[str] = []
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._produce(draft, timings, sentences, parts))
                group.create_task(self._speak(timings, sentences))
            if self._speaker is not None:
                await self._speaker.drain()
        except ExceptionGroup as failures:
            # Surface the original error (e.g. LLMError) instead of a group wrapper.
            self._silence()
            raise failures.exceptions[0] from None
        except asyncio.CancelledError:
            self._silence()  # stopped by the user: cut the voice off immediately
            if draft.done:  # it already acted: a new request, not more of this one
                self.unanswered = None
            raise
        finally:
            draft.discard()  # no-op if finished; stops generation if we were interrupted
        self.unanswered = None
        timings.finished = self._clock()
        logger.info(
            "Turn: %s ms to first word (%s; end of turn %s, stt %s, llm %s, tts %s)",
            timings.voice_to_voice_ms,
            "speculative" if timings.speculative else "sequential",
            timings.end_of_turn_ms,
            timings.stt_ms,
            timings.llm_ms,
            timings.tts_ms,
        )
        return TurnResult(transcript=transcript, reply="".join(parts).strip(), timings=timings)

    async def _produce(
        self,
        draft: Draft,
        timings: TurnTimings,
        sentences: asyncio.Queue[str | None],
        parts: list[str],
    ) -> None:
        """LLM deltas -> sentences queue."""
        splitter = SentenceSplitter()
        try:
            # aclosing: if we are interrupted, the draft records what was heard right away.
            async with aclosing(draft.deltas()) as deltas:
                async for delta in deltas:
                    if timings.first_token is None:
                        timings.first_token = draft.first_token_at or self._clock()
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

    async def _speak(self, timings: TurnTimings, sentences: asyncio.Queue[str | None]) -> None:
        """Sentences queue -> TTS -> speaker."""
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
                self.unanswered = None
                self._observer.on_audio_started()
            await self._speaker.play(audio)

    def _silence(self) -> None:
        if self._speaker is not None:
            self._speaker.stop()

    def _sentence_ready(self, timings: TurnTimings) -> None:
        if timings.first_sentence is None:
            timings.first_sentence = self._clock()
