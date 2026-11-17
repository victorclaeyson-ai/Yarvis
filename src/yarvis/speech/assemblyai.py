"""Live speech recognition by AssemblyAI (Universal-3.6 Pro), with this computer as backup.

Yarvis's own endpointer still decides when you start and stop talking. When it hears you
start, a session opens and your speech streams to AssemblyAI while you talk (connecting
takes place in the meantime, so it costs no time). Whenever Yarvis needs the words, at a
pause and at the end of your turn, it asks AssemblyAI to close the turn
(``ForceEndpoint``) and receives the formatted transcript a moment later. The server's own
turn detection is switched off: two judges of "are you done?" would only disagree.

Privacy: only your speech is sent, from the moment the endpointer hears you start talking
until the end of your turn. The wake word is detected on this computer, so nothing is sent
while Yarvis waits for it, and nothing is sent while Yarvis talks. By default the EU
servers are used (``voice.cloud_region``): AssemblyAI keeps their data in the EU and does
not train on it. Each session is closed right after your turn (they are billed by the
second); its length is reported to the dashboard, which shows what it cost.

Reliability: if anything goes wrong (no internet, a refused key, no answer in time), that
turn is transcribed on this computer instead, from the same audio. A refused key turns
AssemblyAI off until Yarvis restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol
from urllib.parse import urlencode

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from yarvis.speech.base import MIC_SAMPLE_RATE
from yarvis.telemetry import SPEECH, cloud_hearing

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

    from pydantic import SecretStr

    from yarvis.runtime import LoadableSTT
    from yarvis.speech.base import AudioArray
    from yarvis.telemetry import Reporter

logger = logging.getLogger(__name__)

type CloudRegion = Literal["eu", "global"]

MODEL: Final = "universal-3-6-pro"
API_VERSION: Final = "2025-05-12"
"""The streaming API version Yarvis was written against (pinned, so it cannot drift)."""
HOSTS: Final[Mapping[CloudRegion, str]] = {
    "eu": "streaming.eu.assemblyai.com",
    "global": "streaming.assemblyai.com",
}
LANGUAGES: Final = ("en",)
NEVER_MS: Final = 10_000
"""Turn-detection silences that never pass in a conversation: Yarvis ends turns itself."""
INACTIVITY_TIMEOUT_S: Final = 10
"""The server closes a session that receives nothing for this long (a safety net)."""
MAX_SESSION_S: Final = 60.0
"""No session outlives this, whatever happens (an utterance lasts 30 seconds at most)."""
OPEN_TIMEOUT_S: Final = 3.0
SEND_WAIT_S: Final = 4.0
"""Longest wait for the request for a transcript to go out (connecting, then pacing)."""
FINAL_WAIT_S: Final = 1.5
"""Longest wait for the transcript once it was asked for (it usually takes ~0.1 s)."""
CLOSE_WAIT_S: Final = 1.0
MAX_KEYTERMS: Final = 100
MAX_KEYTERM_CHARS: Final = 50

_BYTES_PER_SECOND: Final = MIC_SAMPLE_RATE * 2  # 16-bit samples
_MIN_CHUNK: Final = _BYTES_PER_SECOND * 20 // 1000
"""AssemblyAI takes 20 to 1000 ms of audio per message, and suggests 50 to 100."""
_CHUNK: Final = _BYTES_PER_SECOND * 96 // 1000
_MAX_CHUNK: Final = _BYTES_PER_SECOND
_FORCE_ENDPOINT: Final = json.dumps({"type": "ForceEndpoint"})
_TERMINATE: Final = json.dumps({"type": "Terminate"})
_ERROR_CHARS: Final = 160
_UNAUTHORIZED: Final = 1008
"""Close code for a refused key or account, and for too many new sessions a minute."""
_INPUT_VIOLATION: Final = 3007
"""Close code for audio chunks of the wrong size, or sent faster than real time."""

_background: set[asyncio.Task[None]] = set()
"""Sessions still closing after their turn (kept referenced until they finish)."""


class CloudSttError(Exception):
    """AssemblyAI could not transcribe this turn (the engine on this computer takes over)."""


class RejectedKeyError(CloudSttError):
    """AssemblyAI refused the API key or the account (invalid, disabled, out of funds)."""


class TooFastError(CloudSttError):
    """AssemblyAI received audio faster than it allows."""


class Connection(Protocol):
    """The part of a WebSocket client connection a session uses."""

    async def send(self, message: str | bytes, /) -> None:
        """Send a text (JSON) or binary (audio) message."""
        ...

    async def recv(self) -> str | bytes:
        """Receive the next message; raise ``ConnectionClosed`` at the end."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...


type OpenConnection = Callable[[str, Mapping[str, str]], Awaitable[Connection]]


async def open_connection(url: str, headers: Mapping[str, str]) -> Connection:
    """Open a WebSocket to AssemblyAI (TLS, with the certificate verified).

    Raises:
        RejectedKeyError: If the server refuses the credentials outright.
        CloudSttError: If the connection cannot be made.
    """
    from websockets.asyncio.client import connect  # noqa: PLC0415 - only when it is used

    try:
        return await connect(
            url,
            additional_headers=dict(headers),
            open_timeout=OPEN_TIMEOUT_S,
            compression=None,
            max_size=2**20,
            user_agent_header="Yarvis",
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        error = RejectedKeyError if status in {401, 403} else CloudSttError
        raise error(f"the server answered HTTP {status}") from exc
    except (OSError, TimeoutError, WebSocketException) as exc:
        msg = f"could not connect ({type(exc).__name__})"
        raise CloudSttError(msg) from exc


@dataclass(frozen=True, slots=True)
class Pace:
    """How fast audio may go out: ``burst_s`` seconds at once, then ``rate`` times real time.

    AssemblyAI expects audio at about the speed it is spoken. The start of an utterance
    (and, after the wake word, the phrase itself) is already recorded when the session
    opens; this catches up within a second without flooding the server.
    """

    burst_s: float = 1.0
    rate: float = 2.0

    def wait_s(self, total_bytes: int, elapsed_s: float) -> float:
        """How long to wait before ``total_bytes`` may have been sent."""
        seconds = total_bytes / _BYTES_PER_SECOND
        return max(0.0, (seconds - self.burst_s) / self.rate - elapsed_s)


CAREFUL_PACE: Final = Pace(burst_s=0.3, rate=1.25)
"""Used after AssemblyAI complained that audio arrived too fast."""


def keyterms(words: Iterable[str]) -> list[str]:
    """The distinct words and names to boost, within AssemblyAI's limits."""
    terms = (word.strip() for word in words)
    unique = dict.fromkeys(term for term in terms if 0 < len(term) <= MAX_KEYTERM_CHARS)
    return list(unique)[:MAX_KEYTERMS]


def stream_url(region: CloudRegion, terms: Iterable[str] = ()) -> str:
    """The session URL: model, audio format, and turn detection left to Yarvis."""
    params = {
        "speech_model": MODEL,
        "encoding": "pcm_s16le",
        "sample_rate": str(MIC_SAMPLE_RATE),
        "language_codes": json.dumps(list(LANGUAGES)),
        "min_turn_silence": str(NEVER_MS),
        "max_turn_silence": str(NEVER_MS),
        "inactivity_timeout": str(INACTIVITY_TIMEOUT_S),
    }
    if boosted := keyterms(terms):
        params["keyterms_prompt"] = json.dumps(boosted)
    return f"wss://{HOSTS[region]}/v3/ws?{urlencode(params)}"


class _Configuration(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None


class _Message(BaseModel):
    """Any server message; only the fields Yarvis uses (the rest is ignored)."""

    model_config = ConfigDict(extra="ignore")

    type: str
    transcript: str = ""
    end_of_turn: bool = False
    error_code: int | None = None
    error: str = ""
    configuration: _Configuration | None = None
    session_duration_seconds: float | None = None
    """In ``Termination``: the length of the session, as AssemblyAI bills it."""


def _parse(raw: str) -> _Message | None:
    try:
        return _Message.model_validate_json(raw)
    except ValidationError:
        logger.debug("Ignored a message from AssemblyAI that did not parse")
        return None


def _server_error(message: _Message) -> CloudSttError:
    """The exception for an ``Error`` message (see AssemblyAI's list of session errors)."""
    text = message.error[:_ERROR_CHARS] or "no details"
    code = message.error_code
    if code == _UNAUTHORIZED and "concurrent" in text.lower():
        return CloudSttError(f"too many new sessions this minute ({text})")
    if code == _UNAUTHORIZED:
        return RejectedKeyError(text)
    if code == _INPUT_VIOLATION and "rate" in text.lower():
        return TooFastError(text)
    return CloudSttError(f"error {code}: {text}")


def _pcm(audio: AudioArray) -> bytes:
    """Float samples in [-1, 1] as 16-bit little-endian PCM."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _retrieve[T](future: asyncio.Future[T]) -> None:
    """Mark a failure as seen: nobody may be waiting for this particular future."""
    if not future.cancelled():
        future.exception()


@dataclass(eq=False, slots=True)
class _Request:
    """A request for the transcript of everything streamed before it (``ForceEndpoint``)."""

    samples: int
    sent: asyncio.Future[None]
    result: asyncio.Future[str]


class LiveStream:
    """One utterance, streamed to AssemblyAI while the user speaks.

    ``feed`` audio as it arrives; ``transcript`` returns the words of any prefix of it;
    ``end`` says the utterance is complete (or abandoned) so the session can close.
    """

    def __init__(
        self,
        connect: Callable[[], Awaitable[Connection]],
        *,
        pace: Pace,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_closed: Callable[[float], None] | None = None,
    ) -> None:
        self._connect = connect
        self._pace = pace
        self._on_closed = on_closed
        self._billed: float | None = None
        """The session's length according to AssemblyAI (its ``Termination`` message)."""
        self._clock = clock
        self._sleep = sleep
        self._frames: list[AudioArray] = []
        self._samples = 0
        self._outbox: asyncio.Queue[bytes | _Request | None] = asyncio.Queue()
        self._requests: list[_Request] = []
        self._finals: list[str] = []
        self._heard = False
        """Speech was reported after the latest final transcript."""
        self._error: CloudSttError | None = None
        self._ended = False
        self._terminating = False
        self._task = _keep(asyncio.create_task(self._run()))

    @property
    def done(self) -> bool:
        """Whether the session is over (closed, failed or abandoned)."""
        return self._task.done()

    def feed(self, audio: AudioArray) -> None:
        """Add audio to the utterance (sent as soon as the connection and pace allow)."""
        if self._ended or not len(audio):
            return
        self._frames.append(audio)
        self._samples += len(audio)
        if self._error is None:
            self._outbox.put_nowait(_pcm(audio))

    def holds(self, audio: AudioArray) -> bool:
        """Whether ``audio`` is the start of what was fed (so this session can transcribe it)."""
        count = len(audio)
        if not count or count > self._samples:
            return False
        return bool(np.array_equal(audio, np.concatenate(self._frames)[:count]))

    async def transcript(self, audio: AudioArray) -> str:
        """The words in ``audio`` (a prefix of what was fed), formatted.

        Raises:
            CloudSttError: If AssemblyAI cannot deliver them in time.
        """
        request = self._covering(len(audio))
        try:
            async with asyncio.timeout(SEND_WAIT_S):
                await asyncio.shield(request.sent)
        except TimeoutError:
            msg = "the audio could not be sent in time"
            raise CloudSttError(msg) from None
        try:
            async with asyncio.timeout(FINAL_WAIT_S):
                return await asyncio.shield(request.result)
        except TimeoutError:
            if self._finals and not self._heard:
                # Nothing was said since the last transcript: AssemblyAI has nothing to add.
                return self._text()
            msg = "no transcript in time"
            raise CloudSttError(msg) from None

    def end(self, audio: AudioArray | None) -> None:
        """The utterance is complete (``audio``: what will be transcribed) or abandoned.

        The session closes as soon as the transcript of ``audio`` has arrived.
        """
        if self._ended:
            return
        self._ended = True
        if audio is None or not len(audio) or self._error is not None:
            self._task.cancel()
            return
        _keep(asyncio.create_task(self._close_after(self._covering(len(audio)))))

    def _covering(self, samples: int) -> _Request:
        """The request whose transcript covers the first ``samples`` (made if needed)."""
        for request in self._requests:
            if request.samples >= samples:
                return request
        if self._error is not None:
            raise self._error
        if samples > self._samples:
            msg = "that audio was not streamed"
            raise CloudSttError(msg)
        loop = asyncio.get_running_loop()
        request = _Request(self._samples, loop.create_future(), loop.create_future())
        request.sent.add_done_callback(_retrieve)
        request.result.add_done_callback(_retrieve)
        self._requests.append(request)
        self._outbox.put_nowait(request)
        return request

    def _text(self) -> str:
        return " ".join(text for text in self._finals if text)

    async def _close_after(self, request: _Request) -> None:
        with contextlib.suppress(Exception):  # whoever waits for it has been told
            async with asyncio.timeout(SEND_WAIT_S + FINAL_WAIT_S):
                await asyncio.shield(request.result)
        self._terminating = True
        self._outbox.put_nowait(None)
        try:
            async with asyncio.timeout(CLOSE_WAIT_S):
                await asyncio.shield(self._task)
        except TimeoutError:
            self._task.cancel()

    async def _run(self) -> None:
        connection: Connection | None = None
        opened = self._clock()
        try:
            async with asyncio.timeout(MAX_SESSION_S):
                connection = await self._open()
                sender = asyncio.create_task(self._send(connection))
                try:
                    await self._receive(connection)
                finally:
                    sender.cancel()
                    await asyncio.wait([sender])
        except CloudSttError as exc:
            self._fail(exc)
        except TimeoutError:
            self._fail(CloudSttError("the session took too long"))
        except (OSError, WebSocketException) as exc:
            self._fail(CloudSttError(f"the connection failed ({type(exc).__name__})"))
        finally:
            self._fail(CloudSttError("the session has ended"))
            if connection is not None:
                with contextlib.suppress(Exception):
                    await connection.close()
                self._closed(self._clock() - opened)

    def _closed(self, measured_s: float) -> None:
        """Report how long the session was open (billed by the second)."""
        if self._on_closed is not None:
            self._on_closed(self._billed if self._billed is not None else measured_s)

    async def _open(self) -> Connection:
        started = self._clock()
        try:
            async with asyncio.timeout(OPEN_TIMEOUT_S):
                connection = await self._connect()
        except TimeoutError:
            msg = "could not connect in time"
            raise CloudSttError(msg) from None
        logger.info("AssemblyAI connected in %d ms", (self._clock() - started) * 1000)
        return connection

    async def _send(self, connection: Connection) -> None:
        opened = self._clock()
        buffer = bytearray()
        sent = 0
        try:
            while True:
                item = await self._outbox.get()
                if isinstance(item, bytes):
                    buffer += item
                    if len(buffer) >= _MAX_CHUNK or (
                        self._outbox.empty() and len(buffer) >= _CHUNK
                    ):
                        sent = await self._flush(connection, buffer, sent, opened, everything=False)
                    continue
                sent = await self._flush(connection, buffer, sent, opened, everything=True)
                if item is None:
                    await connection.send(_TERMINATE)
                    return
                await connection.send(_FORCE_ENDPOINT)
                if not item.sent.done():
                    item.sent.set_result(None)
        except Exception as exc:  # noqa: BLE001 - whatever it is, this computer takes over
            self._fail(CloudSttError(f"sending failed ({type(exc).__name__})"))
            await connection.close()

    async def _flush(
        self,
        connection: Connection,
        buffer: bytearray,
        sent: int,
        opened: float,
        *,
        everything: bool,
    ) -> int:
        """Send what is buffered in allowed chunks, at the allowed pace. Returns bytes sent."""
        while buffer and (everything or len(buffer) >= _CHUNK):
            size = min(len(buffer), _MAX_CHUNK)
            if size < _MIN_CHUNK:  # pad the last bit with silence: shorter is refused
                buffer += bytes(_MIN_CHUNK - size)
                size = _MIN_CHUNK
            if (wait := self._pace.wait_s(sent + size, self._clock() - opened)) > 0:
                await self._sleep(wait)
            await connection.send(bytes(buffer[:size]))
            del buffer[:size]
            sent += size
        return sent

    async def _receive(self, connection: Connection) -> None:
        while True:
            try:
                raw = await connection.recv()
            except ConnectionClosed as exc:
                if self._error is None and not self._terminating:
                    code = exc.rcvd.code if exc.rcvd is not None else None
                    self._error = CloudSttError(f"the connection closed (code {code})")
                return
            if isinstance(raw, bytes) or (message := _parse(raw)) is None:
                continue
            if self._handle(message):
                return

    def _handle(self, message: _Message) -> bool:
        """Act on one server message. Returns True once the session has ended."""
        match message.type:
            case "Begin":
                model = message.configuration.model if message.configuration else None
                if model is not None and model != MODEL:
                    logger.warning("AssemblyAI is using %s instead of %s", model, MODEL)
            case "SpeechStarted":
                self._heard = True
            case "Turn" if message.end_of_turn:
                self._final(message.transcript.strip())
            case "Turn":
                self._heard = self._heard or bool(message.transcript.strip())
            case "Error":
                self._error = _server_error(message)
            case "Termination":
                self._billed = message.session_duration_seconds
                return True
            case _:
                pass
        return False

    def _final(self, text: str) -> None:
        """A turn was closed: its transcript answers the oldest open request."""
        self._finals.append(text)
        self._heard = False
        for request in self._requests:
            if not request.result.done():
                if not request.sent.done():
                    request.sent.set_result(None)
                request.result.set_result(self._text())
                return

    def _fail(self, error: CloudSttError) -> None:
        """Give every open request ``error`` (or the first error seen)."""
        self._error = self._error or error
        for request in self._requests:
            for future in (request.sent, request.result):
                if not future.done():
                    future.set_exception(self._error)


def _keep(task: asyncio.Task[None]) -> asyncio.Task[None]:
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task


class AssemblyAISTT:
    """Speech recognition by AssemblyAI while you talk, with this computer's engine as backup.

    A ``LoadableSTT`` that is also a ``listen.AudioTap``: ``listen`` streams each utterance
    to it as it is spoken, and ``transcribe`` then returns the cloud transcript of that
    audio. Audio that was not streamed (such as the warm-up) goes to the backup engine,
    as does every turn the cloud cannot handle.
    """

    model_name: Final = MODEL
    device: Final = "cloud"

    def __init__(
        self,
        key: SecretStr,
        backup: LoadableSTT,
        *,
        region: CloudRegion = "eu",
        terms: Iterable[str] = (),
        connect: OpenConnection = open_connection,
        pace: Pace = Pace(),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.region: CloudRegion = region
        self._key = key
        self._backup = backup
        self._url = stream_url(region, terms)
        self._open = connect
        self._clock = clock
        self._sleep = sleep
        self._pace = pace
        self._stream: LiveStream | None = None
        self._enabled = True
        self._reporter: Reporter | None = None

    def report_to(self, reporter: Reporter) -> None:
        """Report each session's length, and how AssemblyAI is doing, to ``reporter``."""
        self._reporter = reporter

    @property
    def backup(self) -> LoadableSTT:
        """The engine on this computer."""
        return self._backup

    @property
    def enabled(self) -> bool:
        """False once AssemblyAI refused the key (until Yarvis restarts)."""
        return self._enabled

    def load(self) -> None:
        """Load the backup engine (blocking): it must be ready the moment it is needed."""
        self._backup.load()

    def begin(self, audio: AudioArray) -> None:
        """The user started talking: open a session and stream what was heard so far."""
        if self._stream is not None:
            self._stream.end(None)  # a leftover that was never ended
        self._stream = None
        if self._enabled:
            self._stream = LiveStream(
                self._connect,
                pace=self._pace,
                clock=self._clock,
                sleep=self._sleep,
                on_closed=self._session_closed,
            )
            self._stream.feed(audio)

    def feed(self, frame: AudioArray) -> None:
        """Stream the next frame of the same utterance."""
        if self._stream is not None:
            self._stream.feed(frame)

    def end(self, audio: AudioArray | None) -> None:
        """The utterance is complete (``audio``) or abandoned (None)."""
        if self._stream is not None:
            self._stream.end(audio)

    async def transcribe(self, audio: AudioArray) -> str:
        """The cloud transcript if ``audio`` was streamed, else the backup engine's."""
        stream = self._stream
        if stream is not None and stream.holds(audio):
            started = self._clock()
            try:
                text = await stream.transcript(audio)
            except RejectedKeyError as exc:
                self._enabled = False
                logger.warning(
                    "AssemblyAI refused the key or the account (%s). Speech is recognized "
                    "on this computer until Yarvis restarts. To fix the key: yarvis key set "
                    "assemblyai",
                    exc,
                )
                self._health(
                    "error", "AssemblyAI refused the key: this computer listens until restart"
                )
            except TooFastError as exc:
                self._pace = CAREFUL_PACE
                logger.warning("AssemblyAI: %s. Sending audio more slowly from now on.", exc)
                self._health("warn", f"AssemblyAI missed a turn ({exc}): this computer took over")
            except CloudSttError as exc:
                logger.warning(
                    "AssemblyAI could not transcribe this (%s): using this computer", exc
                )
                self._health("warn", f"AssemblyAI missed a turn ({exc}): this computer took over")
            else:
                logger.info("AssemblyAI transcript after %d ms", (self._clock() - started) * 1000)
                self._health("ok", cloud_hearing(self.region))
                return text
        return await self._backup.transcribe(audio)

    def _session_closed(self, seconds: float) -> None:
        if self._reporter is not None:
            self._reporter.speech(MODEL, seconds)

    def _health(self, state: Literal["ok", "warn", "error"], detail: str) -> None:
        if self._reporter is not None:
            self._reporter.health(SPEECH, state, detail)

    def _connect(self) -> Awaitable[Connection]:
        headers = {
            "Authorization": self._key.get_secret_value(),
            "AssemblyAI-Version": API_VERSION,
        }
        return self._open(self._url, headers)
