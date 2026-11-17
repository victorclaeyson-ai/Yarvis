"""AssemblyAI live speech recognition, against a fake service (and once, a real socket)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from pydantic import SecretStr
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from fakes import ASSEMBLYAI_KEY, FakeSTT, RecordingReporter
from test_speech_engines import frames, speech_clip
from yarvis.listen import AudioTap, PauseHooks, listen
from yarvis.speech import assemblyai
from yarvis.speech.assemblyai import (
    API_VERSION,
    CAREFUL_PACE,
    MAX_KEYTERMS,
    MODEL,
    AssemblyAISTT,
    CloudSttError,
    LiveStream,
    Pace,
    RejectedKeyError,
    keyterms,
    open_connection,
    stream_url,
)
from yarvis.speech.endpointing import EndpointerSettings
from yarvis.speech.silero import SileroVAD

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping

    from websockets.asyncio.server import Server, ServerConnection
    from websockets.http11 import Request, Response

    from yarvis.listen import PauseControl
    from yarvis.speech.assemblyai import Connection
    from yarvis.speech.base import AudioArray


class Backup(FakeSTT):
    """The engine on this computer."""

    loaded = False

    def load(self) -> None:
        self.loaded = True


KEY = SecretStr(ASSEMBLYAI_KEY)
FRAME = 512
FORCE = {"type": "ForceEndpoint"}
TERMINATE = {"type": "Terminate"}


def tone(frames_count: int, level: float = 0.25) -> list[AudioArray]:
    """Distinct non-silent frames (so a prefix check can tell them apart)."""
    rng = np.random.default_rng(frames_count)
    return [(rng.uniform(-level, level, FRAME)).astype(np.float32) for _ in range(frames_count)]


def joined(chunks: Iterable[AudioArray]) -> AudioArray:
    return np.concatenate(list(chunks)).astype(np.float32)


def turn(text: str, *, final: bool = True) -> str:
    return json.dumps(
        {
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": final,
            "turn_is_formatted": final,
            "transcript": text,
            "words": [],
        }
    )


def error(code: int, text: str) -> str:
    return json.dumps({"type": "Error", "error_code": code, "error": text})


class FakeService:
    """Plays AssemblyAI: records what the client sends and answers like the real service.

    Each ``ForceEndpoint`` is answered with the next of ``finals`` (None: no answer).
    """

    def __init__(
        self,
        finals: Iterable[str | None] = (),
        *,
        model: str = MODEL,
        greeting: Iterable[str] = (),
        fail_connect: Exception | None = None,
    ) -> None:
        self.finals = deque(finals)
        self.model = model
        self.greeting = list(greeting)
        self.fail_connect = fail_connect
        self.connections: list[FakeConnection] = []

    async def connect(self, url: str, headers: Mapping[str, str]) -> Connection:
        if self.fail_connect is not None:
            raise self.fail_connect
        connection = FakeConnection(self, url, dict(headers))
        self.connections.append(connection)
        return connection

    @property
    def last(self) -> FakeConnection:
        return self.connections[-1]


class FakeConnection:
    def __init__(self, service: FakeService, url: str, headers: dict[str, str]) -> None:
        self.service = service
        self.url = url
        self.headers = headers
        self.audio: list[bytes] = []
        self.controls: list[dict[str, Any]] = []
        self.order: list[str] = []
        self.closed = False
        self.send_termination = True
        self.billed: float | None = None
        """The session length the server reports when it terminates (None: none)."""
        self.answer_while_sending = False
        self.broken: Exception | None = None
        self.inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        begin = {"type": "Begin", "id": "x", "configuration": {"model": service.model}}
        self.inbox.put_nowait(json.dumps(begin))
        for message in service.greeting:
            self.inbox.put_nowait(message)

    async def send(self, message: str | bytes, /) -> None:
        if self.broken is not None:
            raise self.broken
        if isinstance(message, bytes):
            self.audio.append(message)
            self.order.append("audio")
            return
        control: dict[str, Any] = json.loads(message)
        self.controls.append(control)
        self.order.append(control["type"])
        if control == FORCE and self.service.finals:
            final = self.service.finals.popleft()
            if final is not None:
                self.inbox.put_nowait(turn(final))
            if self.answer_while_sending:  # the answer arrives before send() returns
                for _ in range(5):
                    await asyncio.sleep(0)
        elif control == TERMINATE and self.send_termination:
            termination: dict[str, Any] = {"type": "Termination"}
            if self.billed is not None:
                termination["session_duration_seconds"] = self.billed
            self.inbox.put_nowait(json.dumps(termination))

    def say(self, message: str) -> None:
        self.inbox.put_nowait(message)

    def drop(self) -> None:
        """The server closes the connection."""
        self.inbox.put_nowait(None)

    async def recv(self) -> str | bytes:
        if isinstance(self.broken, OSError):
            raise self.broken
        message = await self.inbox.get()
        if message is None:
            raise ConnectionClosed(Close(1011, "gone"), None)
        return message

    async def close(self) -> None:
        self.closed = True
        self.inbox.put_nowait(None)

    @property
    def forces(self) -> int:
        return self.controls.count(FORCE)

    @property
    def sent_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.audio)


FAST = Pace(burst_s=1_000.0, rate=1_000.0)


def make_stt(service: FakeService, backup: Backup | None = None, **kwargs: Any) -> AssemblyAISTT:
    kwargs.setdefault("pace", FAST)
    return AssemblyAISTT(
        KEY, backup or Backup("from this computer"), connect=service.connect, **kwargs
    )


@pytest.fixture(autouse=True)
async def close_sessions() -> AsyncIterator[None]:
    yield
    sessions = list(assemblyai._background)  # pyright: ignore[reportPrivateUsage]
    for task in sessions:
        task.cancel()
    await asyncio.gather(*sessions, return_exceptions=True)


async def settle() -> None:
    """Let the session tasks run."""
    for _ in range(20):
        await asyncio.sleep(0)


async def stream_utterance(stt: AssemblyAISTT, chunks: list[AudioArray], start: int = 10) -> None:
    stt.begin(joined(chunks[:start]))
    for frame in chunks[start:]:
        stt.feed(frame)
    await settle()


# --------------------------------------------------------------------------- settings


def test_the_url_asks_for_the_model_and_leaves_turn_taking_to_yarvis() -> None:
    url = urlparse(stream_url("eu", ["Yarvis", "Leuven"]))
    params = {name: values[0] for name, values in parse_qs(url.query).items()}
    assert url.scheme == "wss"
    assert url.netloc == "streaming.eu.assemblyai.com"
    assert url.path == "/v3/ws"
    assert params["speech_model"] == "universal-3-6-pro"
    assert params["encoding"] == "pcm_s16le"
    assert params["sample_rate"] == "16000"
    assert json.loads(params["language_codes"]) == ["en"]
    assert json.loads(params["keyterms_prompt"]) == ["Yarvis", "Leuven"]
    assert int(params["min_turn_silence"]) >= 10_000
    assert int(params["max_turn_silence"]) >= 10_000
    assert params["inactivity_timeout"] == "10"
    assert "Authorization" not in url.query  # the key travels in a header only


def test_global_region_and_no_terms() -> None:
    url = urlparse(stream_url("global"))
    assert url.netloc == "streaming.assemblyai.com"
    assert "keyterms_prompt" not in parse_qs(url.query)


def test_keyterms_are_unique_short_and_limited() -> None:
    words = ["Yarvis", " Yarvis ", "", "x" * 51, *(f"name{n}" for n in range(200))]
    terms = keyterms(words)
    assert terms[0] == "Yarvis"
    assert terms.count("Yarvis") == 1
    assert "" not in terms
    assert all(len(term) <= 50 for term in terms)
    assert len(terms) == MAX_KEYTERMS


def test_pace_allows_a_burst_then_catches_up_faster_than_real_time() -> None:
    pace = Pace(burst_s=1.0, rate=2.0)
    second = 32_000  # bytes of one second of audio
    assert pace.wait_s(second, elapsed_s=0.0) == 0.0
    assert pace.wait_s(3 * second, elapsed_s=0.0) == pytest.approx(1.0)
    assert pace.wait_s(3 * second, elapsed_s=0.4) == pytest.approx(0.6)
    assert pace.wait_s(3 * second, elapsed_s=2.0) == 0.0


def test_the_cloud_engine_reports_where_it_runs() -> None:
    stt = make_stt(FakeService(), region="global")
    assert stt.region == "global"
    assert stt.device == "cloud"
    assert stt.model_name == MODEL
    assert stt.enabled
    assert isinstance(stt, AudioTap)


def test_loading_loads_the_backup() -> None:
    backup = Backup()
    stt = make_stt(FakeService(), backup)
    stt.load()
    stt.feed(tone(1)[0])  # nothing is streaming: harmless
    stt.end(None)
    assert stt.backup is backup
    assert backup.loaded


# --------------------------------------------------------------------------- one utterance


async def test_a_turn_is_transcribed_in_the_cloud() -> None:
    service = FakeService(["What time is it in Tokyo?"])
    backup = Backup()
    stt = make_stt(service, backup, terms=["Leuven"])
    chunks = tone(60)
    await stream_utterance(stt, chunks)
    audio = joined(chunks[:55])  # the utterance minus its trailing silence
    stt.end(audio)
    assert await stt.transcribe(audio) == "What time is it in Tokyo?"
    await settle()
    connection = service.last
    assert connection.headers == {
        "Authorization": KEY.get_secret_value(),
        "AssemblyAI-Version": API_VERSION,
    }
    assert "Leuven" in connection.url
    assert connection.sent_bytes == 60 * FRAME * 2
    assert all(640 <= len(chunk) <= 32_000 for chunk in connection.audio)
    assert connection.order[-2:] == ["ForceEndpoint", "Terminate"]
    assert "ForceEndpoint" not in connection.order[:-2]
    assert connection.closed
    assert backup.received == []


async def test_a_short_last_chunk_is_padded_with_silence() -> None:
    service = FakeService(["Hi."])
    stt = make_stt(service)
    audio = joined(tone(1))[:100]  # 6 ms: shorter than AssemblyAI accepts
    stt.begin(audio)
    stt.end(audio)
    assert await stt.transcribe(audio) == "Hi."
    assert [len(chunk) for chunk in service.last.audio] == [640]


async def test_an_answer_can_arrive_while_the_request_is_being_sent() -> None:
    service = FakeService(["Quick."])
    stt = make_stt(service)
    chunks = tone(10)
    await stream_utterance(stt, chunks)
    service.last.answer_while_sending = True
    assert await stt.transcribe(joined(chunks)) == "Quick."


async def test_only_streamed_audio_can_be_asked_for() -> None:
    service = FakeService()
    stream = LiveStream(lambda: service.connect("wss://test", {}), pace=FAST)
    stream.feed(joined(tone(2)))
    stream.feed(np.zeros(0, np.float32))  # nothing to add
    with pytest.raises(CloudSttError, match="not streamed"):
        await stream.transcript(joined(tone(3)))
    stream.end(None)


async def test_samples_are_sent_as_16_bit_pcm() -> None:
    service = FakeService(["ok"])
    stt = make_stt(service)
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0] * 200, dtype=np.float32)
    stt.begin(audio)
    stt.end(audio)
    await stt.transcribe(audio)
    sent = np.frombuffer(b"".join(service.last.audio), dtype="<i2")
    assert list(sent[:6]) == [0, 16383, -16383, 32767, -32767, 32767]


async def test_a_pause_asks_for_the_words_and_the_end_reuses_them() -> None:
    service = FakeService(["Set a timer for ten minutes."])
    stt = make_stt(service)
    chunks = tone(40)
    await stream_utterance(stt, chunks[:30])
    at_pause = joined(chunks[:28])
    assert await stt.transcribe(at_pause) == "Set a timer for ten minutes."
    for frame in chunks[30:]:  # silence until the end of the turn
        stt.feed(frame)
    stt.end(at_pause)  # same audio: nothing new was said
    assert await stt.transcribe(at_pause) == "Set a timer for ten minutes."
    await settle()
    assert service.last.forces == 1
    assert service.last.order[-1] == "Terminate"


async def test_speech_after_a_pause_is_added() -> None:
    service = FakeService(["Play.", "Bohemian Rhapsody by Queen."])
    stt = make_stt(service)
    chunks = tone(80)
    await stream_utterance(stt, chunks[:30])
    assert await stt.transcribe(joined(chunks[:30])) == "Play."
    for frame in chunks[30:]:
        stt.feed(frame)
    await settle()
    everything = joined(chunks)
    stt.end(everything)
    assert await stt.transcribe(everything) == "Play. Bohemian Rhapsody by Queen."
    assert service.last.forces == 2


async def test_a_cancelled_request_still_serves_the_next_one() -> None:
    service = FakeService([None])  # the answer is held back
    stt = make_stt(service)
    chunks = tone(30)
    await stream_utterance(stt, chunks)
    audio = joined(chunks)
    waiting = asyncio.create_task(stt.transcribe(audio))
    await settle()
    waiting.cancel()  # the user kept talking: the speculation is discarded
    with pytest.raises(asyncio.CancelledError):
        await waiting
    service.last.say(turn("Hello there."))
    assert await stt.transcribe(audio) == "Hello there."
    assert service.last.forces == 1


async def test_audio_that_was_not_streamed_goes_to_the_backup() -> None:
    service = FakeService(["never"])
    backup = Backup("from this computer")
    stt = make_stt(service, backup)
    assert await stt.transcribe(np.zeros(8_000, np.float32)) == "from this computer"  # warm-up
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    other = joined(tone(25))
    assert await stt.transcribe(other) == "from this computer"
    assert await stt.transcribe(np.zeros(0, np.float32)) == "from this computer"
    assert service.last.forces == 0


async def test_an_abandoned_utterance_closes_the_session() -> None:
    service = FakeService()
    stt = make_stt(service)
    await stream_utterance(stt, tone(20))
    stt.end(None)
    await settle()
    assert service.last.closed
    assert service.last.forces == 0
    stt.feed(tone(1)[0])  # late frames are ignored
    stt.end(None)


async def test_a_new_utterance_replaces_one_that_never_ended() -> None:
    service = FakeService(["Second."])
    stt = make_stt(service)
    await stream_utterance(stt, tone(20))
    chunks = tone(25)
    await stream_utterance(stt, chunks)
    await settle()
    assert service.connections[0].closed
    audio = joined(chunks)
    stt.end(audio)
    assert await stt.transcribe(audio) == "Second."


# --------------------------------------------------------------------------- failures


async def test_no_connection_means_this_computer_transcribes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = FakeService(fail_connect=CloudSttError("could not connect (OSError)"))
    stt = make_stt(service, Backup("local words"))
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    audio = joined(chunks)
    stt.end(audio)
    with caplog.at_level(logging.WARNING):
        assert await stt.transcribe(audio) == "local words"
    assert "could not connect" in caplog.text
    assert stt.enabled


async def test_a_refused_key_turns_the_cloud_off(caplog: pytest.LogCaptureFixture) -> None:
    service = FakeService(greeting=[error(1008, "Unauthorized Connection: Invalid API key")])
    stt = make_stt(service, Backup("local words"))
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    with caplog.at_level(logging.WARNING):
        assert await stt.transcribe(joined(chunks)) == "local words"
    assert not stt.enabled
    assert "yarvis key set assemblyai" in caplog.text
    await stream_utterance(stt, tone(20))
    assert len(service.connections) == 1  # no new sessions


async def test_too_many_new_sessions_is_temporary() -> None:
    service = FakeService(
        greeting=[error(1008, "Unauthorized Connection: Too many concurrent sessions")]
    )
    stt = make_stt(service)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    assert await stt.transcribe(joined(chunks)) == "from this computer"
    assert stt.enabled


async def test_audio_sent_too_fast_slows_down_later_sessions() -> None:
    service = FakeService(greeting=[error(3007, "Audio Transmission Rate Exceeded: 5 s in 1 s")])
    stt = make_stt(service)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    assert await stt.transcribe(joined(chunks)) == "from this computer"
    assert stt._pace == CAREFUL_PACE  # pyright: ignore[reportPrivateUsage]


async def test_other_server_errors_fall_back_for_this_turn() -> None:
    service = FakeService(greeting=[error(3005, "Session Cancelled: An error occurred")])
    stt = make_stt(service)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    assert await stt.transcribe(joined(chunks)) == "from this computer"
    assert stt.enabled


async def test_a_broken_connection_falls_back() -> None:
    service = FakeService(["never"])
    stt = make_stt(service)
    chunks = tone(20)
    stt.begin(joined(chunks))
    await settle()
    service.last.broken = OSError("network is down")
    service.last.say(json.dumps({"type": "Heartbeat"}))
    await settle()
    assert await stt.transcribe(joined(chunks)) == "from this computer"


async def test_a_failed_send_falls_back() -> None:
    service = FakeService(["never"])
    stt = make_stt(service)
    chunks = tone(20)
    stt.begin(joined(chunks[:10]))
    await settle()
    service.last.broken = RuntimeError("unexpected")
    for frame in chunks[10:]:
        stt.feed(frame)
    assert await stt.transcribe(joined(chunks)) == "from this computer"


async def test_no_session_outlives_its_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assemblyai, "MAX_SESSION_S", 0.02)
    service = FakeService()
    stt = make_stt(service)
    chunks = tone(10)
    await stream_utterance(stt, chunks)
    await asyncio.sleep(0.05)
    assert service.last.closed
    assert await stt.transcribe(joined(chunks)) == "from this computer"


async def test_an_unexpected_close_falls_back() -> None:
    service = FakeService()
    stt = make_stt(service)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    assert await stt.transcribe(joined(chunks)) == "from this computer"


async def test_a_slow_answer_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assemblyai, "FINAL_WAIT_S", 0.01)
    service = FakeService([None])
    stt = make_stt(service)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.say(turn("half a sen", final=False))  # speech was heard, no final came
    assert await stt.transcribe(joined(chunks)) == "from this computer"


async def test_silence_after_a_final_needs_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assemblyai, "FINAL_WAIT_S", 0.01)
    service = FakeService(["Turn the volume down.", None])  # no answer: nothing new was said
    stt = make_stt(service)
    chunks = tone(50)
    await stream_utterance(stt, chunks[:20])
    assert await stt.transcribe(joined(chunks[:20])) == "Turn the volume down."
    for frame in chunks[20:]:  # a cough: the endpointer heard "speech", AssemblyAI no words
        stt.feed(frame)
    await settle()
    assert await stt.transcribe(joined(chunks)) == "Turn the volume down."


async def test_a_session_that_cannot_send_in_time_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assemblyai, "SEND_WAIT_S", 0.01)
    never = asyncio.Event()

    async def hang(url: str, headers: Mapping[str, str]) -> Connection:
        await never.wait()
        raise AssertionError

    stt = AssemblyAISTT(KEY, Backup("local"), connect=hang)
    chunks = tone(10)
    await stream_utterance(stt, chunks)
    assert await stt.transcribe(joined(chunks)) == "local"
    stt.end(None)


async def test_connecting_too_slowly_fails_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assemblyai, "OPEN_TIMEOUT_S", 0.01)
    never = asyncio.Event()

    async def hang(url: str, headers: Mapping[str, str]) -> Connection:
        await never.wait()
        raise AssertionError

    stt = AssemblyAISTT(KEY, Backup("local"), connect=hang)
    chunks = tone(10)
    await stream_utterance(stt, chunks)
    await asyncio.sleep(0.05)
    assert await stt.transcribe(joined(chunks)) == "local"


async def test_a_different_model_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    service = FakeService(["fine"], model="universal-streaming-english")
    stt = make_stt(service)
    chunks = tone(10)
    with caplog.at_level(logging.WARNING):
        await stream_utterance(stt, chunks)
    assert "universal-streaming-english instead of universal-3-6-pro" in caplog.text


async def test_unknown_and_broken_messages_are_ignored() -> None:
    service = FakeService(
        ["Hi."],
        greeting=["not json", json.dumps({"no": "type"}), json.dumps({"type": "Heartbeat"})],
    )
    stt = make_stt(service)
    chunks = tone(10)
    await stream_utterance(stt, chunks)
    service.last.say(json.dumps({"type": "SpeechStarted", "timestamp": 10}))
    service.last.inbox.put_nowait(b"binary")
    assert await stt.transcribe(joined(chunks)) == "Hi."


async def test_a_session_is_closed_if_the_server_never_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assemblyai, "CLOSE_WAIT_S", 0.01)
    service = FakeService(["Done."])
    stt = make_stt(service)
    chunks = tone(10)
    await stream_utterance(stt, chunks)
    audio = joined(chunks)
    service.last.send_termination = False
    stt.end(audio)
    assert await stt.transcribe(audio) == "Done."
    await asyncio.sleep(0.05)
    assert service.last.closed


# --------------------------------------------------------------------------- pacing


async def test_the_backlog_is_paced() -> None:
    now = [0.0]
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    service = FakeService(["Hey Yarvis, what time is it?"])
    stt = AssemblyAISTT(
        KEY,
        Backup(),
        connect=service.connect,
        pace=Pace(burst_s=1.0, rate=2.0),
        clock=lambda: now[0],
        sleep=sleep,
    )
    backlog = joined(tone(93))  # 3 seconds recorded before the session opened
    stt.begin(backlog)
    stt.end(backlog)
    assert await stt.transcribe(backlog) == "Hey Yarvis, what time is it?"
    assert service.last.sent_bytes == len(backlog) * 2
    assert waits  # not all at once
    assert now[0] == pytest.approx((len(backlog) / 16_000 - 1.0) / 2.0, abs=0.05)


# --------------------------------------------------------------------------- reports


def reported(stt: AssemblyAISTT) -> RecordingReporter:
    reporter = RecordingReporter()
    stt.report_to(reporter)
    return reporter


async def test_each_session_reports_its_billed_length() -> None:
    service = FakeService(["Lights off."])
    stt = make_stt(service)
    reporter = reported(stt)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.billed = 2.4
    audio = joined(chunks)
    stt.end(audio)
    assert await stt.transcribe(audio) == "Lights off."
    await settle()
    assert reporter.speech_calls == [(MODEL, 2.4)]
    assert reporter.health_calls == [("speech", "ok", "AssemblyAI, EU servers")]


async def test_without_a_billed_length_the_measured_one_counts() -> None:
    now = [100.0]
    service = FakeService(["Hi."])
    stt = make_stt(service, clock=lambda: now[0])
    reporter = reported(stt)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    now[0] += 3.0
    audio = joined(chunks)
    stt.end(audio)
    await stt.transcribe(audio)
    await settle()
    assert reporter.speech_calls == [(MODEL, 3.0)]


async def test_a_session_that_never_connected_costs_nothing() -> None:
    service = FakeService(fail_connect=CloudSttError("could not connect (OSError)"))
    stt = make_stt(service)
    reporter = reported(stt)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    stt.end(joined(chunks))
    assert await stt.transcribe(joined(chunks)) == "from this computer"
    await settle()
    assert reporter.speech_calls == []
    ((key, state, detail),) = reporter.health_calls
    assert (key, state) == ("speech", "warn")
    assert "this computer took over" in detail


async def test_a_refused_key_shows_as_an_error() -> None:
    service = FakeService(greeting=[error(1008, "Unauthorized Connection: Invalid API key")])
    stt = make_stt(service)
    reporter = reported(stt)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    await stt.transcribe(joined(chunks))
    assert reporter.health_calls[-1][:2] == ("speech", "error")
    assert "refused the key" in reporter.health_calls[-1][2]


async def test_sending_too_fast_shows_as_a_warning() -> None:
    service = FakeService(greeting=[error(3007, "Audio Transmission Rate Exceeded: 5 s in 1 s")])
    stt = make_stt(service)
    reporter = reported(stt)
    chunks = tone(20)
    await stream_utterance(stt, chunks)
    service.last.drop()
    await settle()
    await stt.transcribe(joined(chunks))
    assert reporter.health_calls[-1][:2] == ("speech", "warn")


# --------------------------------------------------------------------------- a real socket


@pytest.fixture
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


async def test_a_real_websocket_session(no_proxy: None) -> None:
    received: list[str | bytes] = []
    headers: dict[str, str] = {}

    async def handler(connection: ServerConnection) -> None:
        if connection.request is not None:
            headers.update(connection.request.headers.raw_items())
        await connection.send(json.dumps({"type": "Begin", "configuration": {"model": MODEL}}))
        async for message in connection:
            received.append(message)
            if message == json.dumps(FORCE):
                await connection.send(turn("It works."))
            elif message == json.dumps(TERMINATE):
                await connection.send(json.dumps({"type": "Termination"}))
                return

    async with serve(handler, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{port_of(server)}/v3/ws"

        async def connect() -> Connection:
            return await open_connection(url, {"Authorization": "secret-key"})

        stream = LiveStream(connect, pace=FAST)
        audio = joined(tone(10))
        stream.feed(audio)
        stream.end(audio)
        assert await stream.transcript(audio) == "It works."
        for _ in range(100):
            if stream.done:
                break
            await asyncio.sleep(0.01)
        assert stream.done
    assert headers["Authorization"] == "secret-key"
    assert headers["User-Agent"] == "Yarvis"
    assert sum(len(m) for m in received if isinstance(m, bytes)) == len(audio) * 2


def port_of(server: Server) -> int:
    return int(next(iter(server.sockets)).getsockname()[1])


async def close(connection: ServerConnection) -> None:
    await connection.close()


async def test_refused_handshakes(no_proxy: None) -> None:
    status = [HTTPStatus.UNAUTHORIZED]

    def refuse(connection: ServerConnection, request: Request) -> Response:
        return connection.respond(status[0], "no\n")

    async with serve(close, "127.0.0.1", 0, process_request=refuse) as server:
        url = f"ws://127.0.0.1:{port_of(server)}/v3/ws"
        with pytest.raises(RejectedKeyError, match="401"):
            await open_connection(url, {})
        status[0] = HTTPStatus.SERVICE_UNAVAILABLE
        with pytest.raises(CloudSttError, match="503"):
            await open_connection(url, {})


async def test_an_unreachable_server_is_an_error(no_proxy: None) -> None:
    async with serve(close, "127.0.0.1", 0) as server:
        port = port_of(server)
    with pytest.raises(CloudSttError, match="could not connect"):
        await open_connection(f"ws://127.0.0.1:{port}/v3/ws", {})


# --------------------------------------------------------------------------- with listen()


async def stream_of(items: Iterable[AudioArray]) -> AsyncIterator[AudioArray]:
    for item in items:
        yield item


async def test_listen_streams_real_speech_and_pauses_are_transcribed_live() -> None:
    service = FakeService(["Hey Yarvis, what time is it in Tokyo right now?"])
    stt = make_stt(service)
    at_pause: list[asyncio.Task[str]] = []

    def on_pause(audio: AudioArray, control: PauseControl) -> None:
        at_pause.append(asyncio.create_task(stt.transcribe(audio)))

    utterance = await listen(
        stream_of(frames(speech_clip())),
        SileroVAD(),
        EndpointerSettings(),
        hooks=PauseHooks(on_pause=on_pause, on_resume=lambda: None),
        tap=stt,
    )
    assert utterance is not None
    assert at_pause
    assert await at_pause[-1] == "Hey Yarvis, what time is it in Tokyo right now?"
    assert (
        await stt.transcribe(utterance.audio) == "Hey Yarvis, what time is it in Tokyo right now?"
    )
    await settle()
    assert service.last.forces == 1  # the pause's answer covered the end of the turn
    assert service.last.order[-1] == "Terminate"


class RecordingTap:
    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    def begin(self, audio: AudioArray) -> None:
        self.events.append(("begin", len(audio)))

    def feed(self, frame: AudioArray) -> None:
        self.events.append(("feed", len(frame)))

    def end(self, audio: AudioArray | None) -> None:
        self.events.append(("end", None if audio is None else len(audio)))


async def test_listen_taps_the_whole_utterance() -> None:
    tap = RecordingTap()
    utterance = await listen(
        stream_of(frames(speech_clip())), SileroVAD(), EndpointerSettings(), tap=tap
    )
    assert utterance is not None
    kinds = [kind for kind, _ in tap.events]
    assert kinds[0] == "begin"
    assert kinds[-1] == "end"
    assert set(kinds[1:-1]) == {"feed"}
    streamed = sum(count or 0 for kind, count in tap.events if kind != "end")
    assert tap.events[-1] == ("end", len(utterance.audio))
    assert len(utterance.audio) <= streamed


async def test_an_utterance_cut_short_is_abandoned() -> None:
    tap = RecordingTap()
    clip = frames(speech_clip())
    cut = clip[: len(clip) // 2]
    assert await listen(stream_of(cut), SileroVAD(), EndpointerSettings(), tap=tap) is None
    assert tap.events[0][0] == "begin"
    assert tap.events[-1] == ("end", None)


async def test_silence_is_never_tapped() -> None:
    tap = RecordingTap()
    quiet = [np.full(512, 0.001, np.float32)] * 120
    settings = EndpointerSettings(no_speech_timeout_ms=3_000)
    assert await listen(stream_of(quiet), SileroVAD(), settings, tap=tap) is None
    assert tap.events == []
