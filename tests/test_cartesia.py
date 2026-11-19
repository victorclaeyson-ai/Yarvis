"""Cartesia's voice (ADR 0020), against a fake Cartesia: no network, no account.

``Any`` here is for JSON messages, which have no schema of their own.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import SecretStr
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from fakes import RecordingReporter
from yarvis.config.models import Config, VoiceConfig
from yarvis.doctor import DoctorContext, Status, check_voice
from yarvis.security.secrets import KEYRING_SERVICE
from yarvis.speech import cartesia
from yarvis.speech.cartesia import (
    API_VERSION,
    ARCHIE,
    MODEL,
    SAMPLE_RATE,
    URL,
    CartesiaError,
    CartesiaTTS,
    RejectedError,
    cloud_voice,
    open_connection,
)
from yarvis.telemetry import Setup, Telemetry, starting_links

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fakes import FakeAudioProbe, FakeKeyring
    from yarvis.paths import AppPaths
    from yarvis.security.secrets import SecretStore

KEY = "sk_car_" + "t3stK3y" * 4
SENTENCE = "Your exam is on Tuesday."


def pcm(*values: float) -> str:
    return base64.b64encode(np.array(values, dtype=np.float32).tobytes()).decode()


class Backup:
    """Kokoro's stand-in: records what it had to say."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.said: list[str] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    async def synthesize(self, text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        self.said.append(text)
        return np.full(4, 0.5, dtype=np.float32)


class FakeCartesia:
    """Plays Cartesia: answers each request according to ``plan`` ("ok", "error", ...)."""

    def __init__(self, *plans: str, fail_connect: Exception | None = None) -> None:
        self.plans = deque(plans)
        self.fail_connect = fail_connect
        self.connections: list[FakeSocket] = []

    async def connect(self, url: str, headers: Mapping[str, str]) -> FakeSocket:
        if self.fail_connect is not None:
            raise self.fail_connect
        socket = FakeSocket(self, url, dict(headers))
        self.connections.append(socket)
        return socket


class FakeSocket:
    def __init__(self, service: FakeCartesia, url: str, headers: dict[str, str]) -> None:
        self.service = service
        self.url = url
        self.headers = headers
        self.sent: list[dict[str, Any]] = []
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str | bytes, /) -> None:
        data: dict[str, Any] = json.loads(message)
        self.sent.append(data)
        if data.get("cancel"):
            return
        context = data["context_id"]
        plan = self.service.plans.popleft() if self.service.plans else "ok"
        if plan == "ok":
            stray = {"type": "chunk", "context_id": "someone-else", "data": pcm(9.0)}
            self.inbox.put_nowait(json.dumps(stray))  # another context: ignored
            for values in ((0.1, 0.2), (0.3,)):
                chunk = {"type": "chunk", "context_id": context, "data": pcm(*values)}
                self.inbox.put_nowait(json.dumps(chunk))
            self.inbox.put_nowait(json.dumps({"type": "done", "context_id": context}))
        elif plan in {"error", "quota", "unauthorized"}:
            code = {"error": 500, "quota": 402, "unauthorized": 401}[plan]
            error_code = {"error": "internal", "quota": "quota_exceeded"}.get(plan, "invalid")
            error = {
                "type": "error",
                "context_id": context,
                "status_code": code,
                "error_code": error_code,
                "title": "Nope",
            }
            self.inbox.put_nowait(json.dumps(error))
        elif plan == "closed":
            self.inbox.put_nowait(None)  # Cartesia closed an idle connection
        elif plan == "truncated":  # 5 bytes: not a whole number of samples
            odd = base64.b64encode(b"\x00" * 5).decode()
            self.inbox.put_nowait(json.dumps({"type": "chunk", "context_id": context, "data": odd}))
            self.inbox.put_nowait(json.dumps({"type": "done", "context_id": context}))
        elif plan == "empty":  # done, without any audio
            self.inbox.put_nowait(json.dumps({"type": "done", "context_id": context}))
        elif plan == "endless":  # audio that never ends
            for _ in range(1_000):
                chunk = {"type": "chunk", "context_id": context, "data": pcm(0.1)}
                self.inbox.put_nowait(json.dumps(chunk))
        elif plan == "quota-without-context":  # an error for the whole connection
            error = {"type": "error", "status_code": 402, "error_code": "quota_exceeded"}
            self.inbox.put_nowait(json.dumps(error))
        # "silent": never answers

    async def recv(self) -> str | bytes:
        message = await self.inbox.get()
        if message is None:
            raise ConnectionClosedOK(Close(1000, "idle"), None)
        return message

    async def close(self) -> None:
        self.closed = True


def make(
    service: FakeCartesia, backup: Backup | None = None, **kwargs: Any
) -> tuple[CartesiaTTS, Backup, RecordingReporter]:
    backup = backup or Backup()
    reporter = RecordingReporter()
    kwargs.setdefault("connect", service.connect)
    tts = CartesiaTTS(SecretStr(KEY), backup, **kwargs)
    tts.report_to(reporter)
    return tts, backup, reporter


# --------------------------------------------------------------------------- speaking


async def test_a_sentence_is_spoken_by_cartesia() -> None:
    service = FakeCartesia()
    tts, backup, reporter = make(service, voice=ARCHIE, speed=1.1)
    audio = await tts.synthesize(SENTENCE)
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.1, 0.2, 0.3])  # chunks in order, others ignored
    assert backup.said == []
    socket = service.connections[0]
    assert socket.url == URL  # the key is never in the address
    assert KEY not in socket.url
    assert socket.headers["Authorization"] == f"Bearer {KEY}"
    assert socket.headers["Cartesia-Version"] == API_VERSION
    request = socket.sent[0]
    assert request["model_id"] == MODEL
    assert request["transcript"] == SENTENCE  # only the sentence, nothing else
    assert request["voice"] == {"id": ARCHIE}
    assert request["output_format"] == {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": SAMPLE_RATE,
    }
    assert request["language"] == "en"
    assert request["continue"] is False
    assert request["max_buffer_delay_ms"] == 0  # whole sentences: no server-side wait
    assert request["generation_config"] == {"speed": 1.1}
    assert reporter.voice_calls == [(MODEL, len(SENTENCE))]


@pytest.mark.parametrize(("rate", "speed"), [(0.5, 0.6), (1.0, 1.0), (2.0, 1.5)])
def test_the_speech_rate_fits_cartesia_s_range(rate: float, speed: float) -> None:
    tts, _, _ = make(FakeCartesia(), speed=rate)
    assert tts.speed == speed


async def test_one_connection_serves_every_sentence() -> None:
    service = FakeCartesia()
    tts, _, _ = make(service)
    await tts.synthesize("One.")
    await tts.synthesize("Two.")
    assert len(service.connections) == 1
    contexts = [request["context_id"] for request in service.connections[0].sent]
    assert len(set(contexts)) == 2  # each sentence its own context


async def test_a_connection_cartesia_closed_is_opened_again() -> None:
    service = FakeCartesia("ok", "closed", "ok")
    tts, backup, _ = make(service)
    await tts.synthesize("One.")
    audio = await tts.synthesize("Two.")  # the idle connection was closed: retried once
    assert audio.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert len(service.connections) == 2
    assert backup.said == []


async def test_load_also_loads_the_backup() -> None:
    tts, backup, _ = make(FakeCartesia())
    tts.load()
    assert backup.loaded


# --------------------------------------------------------------------------- Kokoro takes over


async def test_without_internet_kokoro_speaks() -> None:
    service = FakeCartesia(fail_connect=CartesiaError("could not connect (OSError)"))
    tts, backup, reporter = make(service)
    audio = await tts.synthesize(SENTENCE)
    assert audio.tolist() == [0.5] * 4
    assert backup.said == [SENTENCE]
    assert reporter.health_calls[-1][1] == "warn"
    assert reporter.voice_calls == []  # nothing was sent, nothing counted


async def test_an_error_from_cartesia_hands_that_sentence_to_kokoro() -> None:
    service = FakeCartesia("error", "ok")
    tts, backup, _ = make(service)
    await tts.synthesize("One.")
    await tts.synthesize("Two.")
    assert backup.said == ["One."]  # only the sentence that failed


async def test_a_slow_cartesia_hands_over_in_time() -> None:
    service = FakeCartesia("silent")
    tts, backup, _ = make(service, first_audio_timeout_s=0.05)
    await tts.synthesize(SENTENCE)
    assert backup.said == [SENTENCE]


@pytest.mark.parametrize("plan", ["quota", "unauthorized"])
async def test_a_refused_key_or_used_up_credits_switch_cartesia_off(plan: str) -> None:
    service = FakeCartesia(plan)
    tts, backup, reporter = make(service)
    await tts.synthesize("One.")
    await tts.synthesize("Two.")
    assert backup.said == ["One.", "Two."]
    assert len(service.connections[0].sent) == 1  # not asked again until Yarvis restarts
    state, detail = reporter.health_calls[-1][1:]
    assert state == "error"
    assert "Kokoro" in detail


async def test_a_refused_connection_switches_cartesia_off() -> None:
    service = FakeCartesia(fail_connect=RejectedError("the server answered HTTP 401"))
    tts, backup, _ = make(service)
    await tts.synthesize("One.")
    service.fail_connect = None
    await tts.synthesize("Two.")
    assert backup.said == ["One.", "Two."]
    assert service.connections == []


async def test_an_interrupted_sentence_is_cancelled_and_its_audio_dropped() -> None:
    service = FakeCartesia("silent", "ok")
    tts, _, _ = make(service, first_audio_timeout_s=5)
    task = asyncio.create_task(tts.synthesize("A long answer."))
    await asyncio.sleep(0.01)
    task.cancel()  # the user interrupted Yarvis
    with pytest.raises(asyncio.CancelledError):
        await task
    first = service.connections[0]
    assert first.closed  # nothing late can leak into the next sentence
    audio = await tts.synthesize("Next.")
    assert audio.tolist() == pytest.approx([0.1, 0.2, 0.3])


async def test_the_spoken_text_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    service = FakeCartesia("error", "ok")
    tts, _, _ = make(service)
    with caplog.at_level(logging.DEBUG):
        await tts.synthesize("My secret plans for Tuesday.")
        await tts.synthesize("Another private sentence.")
    assert "secret plans" not in caplog.text
    assert "private sentence" not in caplog.text
    assert KEY not in caplog.text


# --------------------------------------------------------------------------- wiring


def test_cartesia_is_used_only_when_chosen_and_a_key_is_stored(
    secrets: SecretStore, fake_keyring: FakeKeyring, caplog: pytest.LogCaptureFixture
) -> None:
    kokoro = Backup()
    chosen = VoiceConfig(tts="cartesia", speech_rate=1.2)
    assert cloud_voice(VoiceConfig(), kokoro, secrets) is kokoro  # the default: Kokoro
    assert cloud_voice(chosen, kokoro, secrets) is kokoro  # no key yet
    assert "yarvis key set cartesia" in caplog.text
    fake_keyring.store[(KEYRING_SERVICE, "cartesia")] = KEY
    voice = cloud_voice(chosen, kokoro, secrets)
    assert isinstance(voice, CartesiaTTS)
    assert voice.voice == ARCHIE
    assert voice.model == MODEL
    assert voice.speed == pytest.approx(1.2)


def test_voice_settings_and_defaults() -> None:
    voice = Config().voice
    assert voice.tts == "kokoro"
    assert voice.cartesia_voice == ARCHIE
    assert voice.cartesia_model == MODEL
    assert voice.cartesia_usd_per_million_chars == 0  # the Free plan
    with pytest.raises(ValueError, match="cartesia_voice"):
        Config.model_validate({"voice": {"cartesia_voice": "not an id"}})


def test_characters_are_counted_and_priced_by_plan() -> None:
    free = Telemetry()
    free.spoken(MODEL, 1_000)
    assert free.today.voice_chars == 1_000
    assert free.today.voice_usd == 0
    pro = Telemetry(voice_usd_per_million=65.0)
    pro.spoken(MODEL, 2_000)
    assert pro.today.voice_usd == pytest.approx(0.13)
    assert pro.today.total_usd == pytest.approx(0.13)


def test_the_dashboard_says_which_voice_speaks() -> None:
    def detail(*, cloud_voice: bool) -> str:
        setup = Setup(
            model="claude-haiku-5-5", hearing=None, cloud_hearing=False, voice="bm_george",
            wake_word=False, spotify=False, spotify_login=False, personal=False, skills=0,
            profile=False, cloud_voice=cloud_voice,
        )  # fmt: skip
        return next(link for link in starting_links(Config(), setup) if link.key == "voice").detail

    assert detail(cloud_voice=False) == "Kokoro · bm_george"
    assert detail(cloud_voice=True) == "Cartesia · Kokoro as backup"


def test_doctor_checks_the_cloud_voice(
    paths: AppPaths, secrets: SecretStore, audio: FakeAudioProbe, fake_keyring: FakeKeyring
) -> None:
    ctx = DoctorContext(
        paths=paths, secrets=secrets, audio=audio, python_version=(3, 12, 4), system="Linux",
        machine="x86_64",
    )  # fmt: skip
    assert check_voice(ctx, Config()).detail == "Kokoro, on this computer"
    chosen = Config.model_validate({"voice": {"tts": "cartesia"}})
    missing = check_voice(ctx, chosen)
    assert missing.status is Status.WARN
    assert "yarvis key set cartesia" in (missing.hint or "")
    fake_keyring.store[(KEYRING_SERVICE, "cartesia")] = KEY
    ready = check_voice(ctx, chosen)
    assert ready.status is Status.OK
    assert "data-controls" in (ready.hint or "")  # the training opt-out


# --------------------------------------------------------------------------- edge cases


def test_the_backup_must_speak_at_the_same_rate() -> None:
    class Slow(Backup):
        sample_rate = 16_000

    with pytest.raises(ValueError, match="24000 Hz"):
        CartesiaTTS(SecretStr(KEY), Slow(), connect=FakeCartesia().connect)


class Garbled(FakeSocket):
    """Answers with ``replies`` instead of audio."""

    replies: tuple[str, ...] = ()

    async def send(self, message: str | bytes, /) -> None:
        data: dict[str, Any] = json.loads(message)
        self.sent.append(data)
        for reply in self.replies:
            self.inbox.put_nowait(reply.replace("CTX", data.get("context_id", "")))


@pytest.mark.parametrize(
    "replies",
    [
        ("not json",),
        ('{"type": "chunk", "context_id": "CTX", "data": "%%%not base64"}',),
    ],
)
async def test_unreadable_answers_hand_the_sentence_to_kokoro(replies: tuple[str, ...]) -> None:
    service = FakeCartesia()

    async def connect(url: str, headers: Mapping[str, str]) -> FakeSocket:
        socket = Garbled(service, url, dict(headers))
        socket.replies = replies
        return socket

    tts, backup, _ = make(service, connect=connect)
    await tts.synthesize(SENTENCE)
    assert backup.said == [SENTENCE]


async def test_other_messages_are_skipped_and_done_in_a_chunk_ends_it() -> None:
    service = FakeCartesia()

    async def connect(url: str, headers: Mapping[str, str]) -> FakeSocket:
        socket = Garbled(service, url, dict(headers))
        socket.replies = (
            '{"type": "timestamps", "context_id": "CTX"}',
            json.dumps({"type": "chunk", "context_id": "CTX", "data": pcm(0.4), "done": True}),
        )
        return socket

    tts, backup, _ = make(service, connect=connect)
    assert (await tts.synthesize(SENTENCE)).tolist() == pytest.approx([0.4])
    assert backup.said == []


async def test_a_connection_that_closes_right_away_hands_over() -> None:
    service = FakeCartesia("closed", "closed", "closed")
    tts, backup, _ = make(service)
    await tts.synthesize("One.")  # a fresh connection closed: no retry
    assert backup.said == ["One."]
    service.plans = deque(["ok", "closed", "closed"])
    await tts.synthesize("Two.")  # new connection, works
    await tts.synthesize("Three.")  # reused and closed, retried, closed again
    assert backup.said == ["One.", "Three."]


async def test_without_a_reporter_it_still_speaks() -> None:
    service = FakeCartesia("ok", "error")
    tts = CartesiaTTS(SecretStr(KEY), Backup(), connect=service.connect)
    assert (await tts.synthesize("One.")).size == 3
    assert (await tts.synthesize("Two.")).size == 4


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (OSError("no route"), CartesiaError),
        ("401", RejectedError),
        ("503", CartesiaError),
    ],
)
async def test_opening_the_connection_explains_failures(
    monkeypatch: pytest.MonkeyPatch, raised: Exception | str, expected: type[CartesiaError]
) -> None:
    from websockets.asyncio import client  # noqa: PLC0415 - patched for this test only
    from websockets.datastructures import Headers  # noqa: PLC0415
    from websockets.exceptions import InvalidStatus  # noqa: PLC0415
    from websockets.http11 import Response  # noqa: PLC0415

    async def refuse(*_args: object, **_kwargs: object) -> None:
        if isinstance(raised, str):
            raise InvalidStatus(Response(int(raised), "No", Headers()))
        raise raised

    monkeypatch.setattr(client, "connect", refuse)
    with pytest.raises(expected) as caught:
        await open_connection(URL, {"Authorization": f"Bearer {KEY}"})
    assert KEY not in str(caught.value)


# --------------------------------------------------------------------------- security review


@pytest.mark.parametrize("plan", ["truncated", "empty"])
async def test_audio_that_is_not_a_whole_sentence_hands_over(plan: str) -> None:
    service = FakeCartesia(plan, "ok")
    tts, backup, _ = make(service)
    assert (await tts.synthesize("One.")).tolist() == [0.5] * 4  # Kokoro's, not an error
    assert backup.said == ["One."]
    await tts.synthesize("Two.")
    assert len(service.connections) == 2


async def test_a_connection_error_on_sending_hands_over() -> None:
    service = FakeCartesia()

    class Broken(FakeSocket):
        async def send(self, message: str | bytes, /) -> None:
            raise OSError("network is unreachable")

    async def connect(url: str, headers: Mapping[str, str]) -> FakeSocket:
        socket = Broken(service, url, dict(headers))
        service.connections.append(socket)
        return socket

    tts, backup, _ = make(service, connect=connect)
    await tts.synthesize("One.")
    await tts.synthesize("Two.")
    assert backup.said == ["One.", "Two."]
    assert len(service.connections) == 2


async def test_an_error_for_the_whole_connection_switches_cartesia_off() -> None:
    service = FakeCartesia("quota-without-context", "ok")
    tts, backup, _ = make(service, first_audio_timeout_s=5)
    await tts.synthesize("One.")
    await tts.synthesize("Two.")
    assert backup.said == ["One.", "Two."]
    assert len(service.connections[0].sent) == 1  # not asked again


async def test_a_sentence_that_never_ends_hits_its_deadline() -> None:
    service = FakeCartesia("endless")
    tts, backup, _ = make(service, sentence_timeout_s=0.05)
    await tts.synthesize(SENTENCE)
    assert backup.said == [SENTENCE]


async def test_more_audio_than_a_sentence_can_have_hands_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cartesia, "MAX_SENTENCE_BYTES", 1_000)  # 250 of its 1,000 samples
    service = FakeCartesia("endless", "ok")
    tts, backup, _ = make(service)
    await tts.synthesize(SENTENCE)
    assert backup.said == [SENTENCE]  # never buffered without limit
    assert len(await tts.synthesize(SENTENCE)) > 0  # a fresh connection for the next one
    assert len(service.connections) == 2
