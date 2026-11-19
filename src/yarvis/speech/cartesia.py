"""Yarvis's voice from Cartesia (Sonic 3.6), with Kokoro on this computer as backup.

Why it exists (ADR 0020): Victor prefers Cartesia's voice to Kokoro's. Each sentence Claude
writes is sent to Cartesia as soon as it is complete and comes back as 24 kHz audio, the
same format as Kokoro's, over one WebSocket that stays open between sentences.

Privacy: only the sentence being spoken leaves this computer; never the profile, memory,
history or anything the user said. Cartesia is in the US and may train on the text unless
the user opted out (``yarvis key set cartesia`` says how). Nothing is written to disk, and
the text is never logged.

Reliability: if Cartesia can't speak a sentence (no internet, an error, no audio within
``first_audio_timeout_s``), Kokoro speaks it instead, so Yarvis never goes silent. A refused
key or used-up credits switch Cartesia off until Yarvis restarts.

What it must never do: put the API key anywhere but the ``Authorization`` header (not in
the address, not in a log); or let audio from an interrupted sentence reach the next one.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import SecretStr

    from yarvis.config.models import VoiceConfig
    from yarvis.security.secrets import SecretStore
    from yarvis.speech.base import AudioArray
    from yarvis.telemetry import LinkState, Reporter

logger = logging.getLogger(__name__)

URL: Final = "wss://api.cartesia.ai/tts/websocket"
API_VERSION: Final = "2026-08-14"
"""Checked against Cartesia's docs on 19 November 2026 (ADR 0020)."""
MODEL: Final = "sonic-3.6"
ARCHIE: Final = "ef191366-f52f-447a-a398-ed8c0f2943a1"
"""Archie: British English, male; one of Cartesia's recommended voices for Sonic 3.6."""
SAMPLE_RATE: Final = 24_000
MIN_SPEED: Final = 0.6
MAX_SPEED: Final = 1.5
FIRST_AUDIO_TIMEOUT_S: Final = 2.0
"""No audio after this long: Kokoro speaks the sentence instead."""
SENTENCE_TIMEOUT_S: Final = 20.0
OPEN_TIMEOUT_S: Final = 5.0
CLOSE_TIMEOUT_S: Final = 1.0
"""Closing after an interruption: the next sentence mustn't wait for the handshake."""
MAX_SENTENCE_BYTES: Final = 16 * 2**20
"""About three minutes of 24 kHz float audio: far more than any spoken sentence."""
VOICE: Final = "voice"
"""The dashboard's key for the voice's connection."""
_REFUSED_STATUS: Final = frozenset({401, 402, 403})
_REFUSED_CODES: Final = frozenset({"quota_exceeded"})


class CartesiaError(Exception):
    """Cartesia could not speak a sentence (the message says why, without the text)."""


class RejectedError(CartesiaError):
    """Cartesia refused the key or the account has no credits left: don't ask again."""


class Connection(Protocol):
    """The part of a WebSocket client connection the voice uses."""

    async def send(self, message: str | bytes, /) -> None:
        """Send a text (JSON) message."""
        ...

    async def recv(self) -> str | bytes:
        """Receive the next message; raise ``ConnectionClosed`` at the end."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...


type OpenConnection = Callable[[str, Mapping[str, str]], Awaitable[Connection]]


class Backup(Protocol):
    """The voice on this computer (Kokoro) that speaks when Cartesia can't."""

    @property
    def sample_rate(self) -> int:
        """Must match Cartesia's: one speaker plays both."""
        ...

    def load(self) -> None:
        """Load the model (blocking)."""
        ...

    async def synthesize(self, text: str) -> AudioArray:
        """Speak ``text`` on this computer."""
        ...


async def open_connection(url: str, headers: Mapping[str, str]) -> Connection:
    """Open the WebSocket to Cartesia (TLS, with the certificate verified).

    Raises:
        RejectedError: If Cartesia refuses the key outright.
        CartesiaError: If the connection cannot be made.
    """
    from websockets.asyncio.client import connect  # noqa: PLC0415 - only when it is used

    try:
        return await connect(
            url,
            additional_headers=dict(headers),
            open_timeout=OPEN_TIMEOUT_S,
            compression=None,
            max_size=2**22,
            close_timeout=CLOSE_TIMEOUT_S,
            proxy=None,  # straight to Cartesia: no system proxy sees the traffic
            user_agent_header="Yarvis",
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        error = RejectedError if status in _REFUSED_STATUS else CartesiaError
        raise error(f"the server answered HTTP {status}") from exc
    except (OSError, TimeoutError, WebSocketException) as exc:
        msg = f"could not connect ({type(exc).__name__})"
        raise CartesiaError(msg) from exc


class CartesiaTTS:
    """``TextToSpeech`` through Cartesia, with ``backup`` speaking whenever it can't."""

    def __init__(
        self,
        api_key: SecretStr,
        backup: Backup,
        *,
        voice: str = ARCHIE,
        model: str = MODEL,
        speed: float = 1.0,
        connect: OpenConnection = open_connection,
        first_audio_timeout_s: float = FIRST_AUDIO_TIMEOUT_S,
        sentence_timeout_s: float = SENTENCE_TIMEOUT_S,
        new_context: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if backup.sample_rate != SAMPLE_RATE:
            msg = f"the backup voice must produce {SAMPLE_RATE} Hz audio"
            raise ValueError(msg)
        self._key = api_key
        self._backup = backup
        self.voice = voice
        self.model = model
        self.speed = min(max(speed, MIN_SPEED), MAX_SPEED)
        self._connect = connect
        self._first_wait_s = first_audio_timeout_s
        self._sentence_timeout_s = sentence_timeout_s
        self._new_context = new_context
        self._connection: Connection | None = None
        self._lock = asyncio.Lock()
        self._off: str | None = None
        """Why Cartesia is switched off until restart (None: it is on)."""
        self._reporter: Reporter | None = None

    @property
    def sample_rate(self) -> int:
        """24 kHz, like Kokoro."""
        return SAMPLE_RATE

    def load(self) -> None:
        """Load the backup voice, so taking over costs no time."""
        self._backup.load()

    def report_to(self, reporter: Reporter) -> None:
        """Report characters spoken, and how the connection is doing, to ``reporter``."""
        self._reporter = reporter

    async def synthesize(self, text: str) -> AudioArray:
        """Speak ``text`` with Cartesia, or with the backup if that fails."""
        if self._off is None:
            try:
                async with self._lock:
                    audio = await self._speak(text)
            except RejectedError as exc:
                self._off = str(exc)
                logger.warning("Cartesia is off until Yarvis restarts: %s", exc)
                self._health("error", f"Kokoro speaks: Cartesia {exc}")
            except CartesiaError as exc:
                logger.warning("Cartesia could not speak a sentence: %s", exc)
                self._health("warn", f"Kokoro spoke a sentence: Cartesia {exc}")
            else:
                if self._reporter is not None:
                    self._reporter.spoken(self.model, len(text))
                    self._health("ok", f"Cartesia {self.model}")
                return audio
        return await self._backup.synthesize(text)

    async def _speak(self, text: str) -> AudioArray:
        """One sentence; a connection Cartesia closed while idle is opened once more.

        Any failure becomes a ``CartesiaError`` (so Kokoro speaks), and drops the connection.
        """
        fresh = self._connection is None
        try:
            connection = await self._open()
            try:
                return await self._sentence(connection, text)
            except ConnectionClosed:
                self._connection = None
                if fresh:
                    raise
                logger.info("Cartesia closed the idle connection: opening it again")
                return await self._sentence(await self._open(), text)
        except CartesiaError:
            self._connection = None
            raise
        except (OSError, WebSocketException) as exc:  # ConnectionClosed included
            self._connection = None
            msg = f"lost the connection ({type(exc).__name__})"
            raise CartesiaError(msg) from exc

    async def _open(self) -> Connection:
        if self._connection is None:
            headers = {
                "Authorization": f"Bearer {self._key.get_secret_value()}",
                "Cartesia-Version": API_VERSION,
            }
            self._connection = await self._connect(URL, headers)
        return self._connection

    async def _sentence(self, connection: Connection, text: str) -> AudioArray:
        context = self._new_context()
        request = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"id": self.voice},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": SAMPLE_RATE,
            },
            "language": "en",
            "context_id": context,
            "continue": False,
            "max_buffer_delay_ms": 0,  # whole sentences: Cartesia mustn't wait for more
            "generation_config": {"speed": self.speed},
        }
        await connection.send(json.dumps(request))
        chunks: list[bytes] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._sentence_timeout_s  # the whole sentence, not a message
        try:
            while True:
                wait = deadline if chunks else min(deadline, loop.time() + self._first_wait_s)
                try:
                    async with asyncio.timeout_at(wait):
                        raw = await connection.recv()
                except TimeoutError as exc:
                    await self._drop(connection, context)
                    msg = "sent no audio in time" if not chunks else "took too long"
                    raise CartesiaError(msg) from exc
                done = self._take(raw, context, chunks)
                size = sum(len(chunk) for chunk in chunks)
                if size > MAX_SENTENCE_BYTES:
                    await self._drop(connection, context)
                    msg = "sent more audio than a sentence can have"
                    raise CartesiaError(msg)
                if done:
                    break
        except asyncio.CancelledError:
            await self._drop(connection, context)  # interrupted: nothing late may leak
            raise
        audio = b"".join(chunks)
        if not audio or len(audio) % 4:  # float32 samples: whole ones, and at least one
            msg = "sent incomplete audio"
            raise CartesiaError(msg)
        return np.frombuffer(audio, dtype=np.float32).copy()

    def _take(self, raw: str | bytes, context: str, chunks: list[bytes]) -> bool:
        """Handle one message for ``context``; True when the sentence is complete."""
        try:
            message = _Message.model_validate_json(raw)
        except ValidationError as exc:
            msg = "sent a message Yarvis can't read"
            raise CartesiaError(msg) from exc
        if message.type == "error" and (
            message.context_id is None or _refused(message) or message.context_id == context
        ):
            raise _error(message)  # also errors for the whole connection (a refused key)
        if message.context_id != context:
            return False  # another (cancelled) sentence's late message
        if message.type == "chunk":
            try:
                chunks.append(base64.b64decode(message.data, validate=True))
            except (binascii.Error, ValueError) as exc:
                msg = "sent audio that can't be decoded"
                raise CartesiaError(msg) from exc
            return message.done
        return message.type == "done"  # anything else: timestamps and the like

    async def _drop(self, connection: Connection, context: str) -> None:
        """Stop a sentence and close the connection, so its late audio can't leak."""
        try:
            with contextlib.suppress(*_GONE):  # best effort: it may be gone already
                await connection.send(json.dumps({"context_id": context, "cancel": True}))
            with contextlib.suppress(*_GONE, TimeoutError):
                async with asyncio.timeout(CLOSE_TIMEOUT_S):  # an interruption can't wait
                    await connection.close()
        finally:
            self._connection = None  # even if this, too, is interrupted
        self._connection = None

    def _health(self, state: LinkState, detail: str) -> None:
        if self._reporter is not None:
            self._reporter.health(VOICE, state, detail)


class _Message(BaseModel):
    """One message from Cartesia: only the fields Yarvis reads (outside data, validated)."""

    model_config = ConfigDict(extra="ignore")

    type: str = ""
    context_id: str | None = None
    data: str = ""
    done: bool = False
    status_code: int | None = None
    error_code: str = Field(default="", max_length=64)  # it reaches the log


_GONE: Final = (ConnectionClosed, OSError, WebSocketException, RuntimeError)
"""What sending to or closing a connection that is already gone can raise."""


def _refused(message: _Message) -> bool:
    """A refused key or used-up credits: don't ask Cartesia again."""
    return message.status_code in _REFUSED_STATUS or message.error_code in _REFUSED_CODES


def _error(message: _Message) -> CartesiaError:
    """Cartesia's error message, as an exception (refusals switch it off)."""
    status, code = message.status_code, message.error_code
    detail = f"answered {status} ({code})" if code else f"answered {status}"
    return RejectedError(detail) if _refused(message) else CartesiaError(detail)


def cloud_voice[T: Backup](config: VoiceConfig, kokoro: T, secrets: SecretStore) -> CartesiaTTS | T:
    """Cartesia when chosen and a key is stored, with ``kokoro`` behind it; else Kokoro."""
    if config.tts != "cartesia":
        return kokoro
    from yarvis.security.secrets import SecretName  # noqa: PLC0415 - avoids an import cycle

    key = secrets.get(SecretName.CARTESIA)
    if key is None:
        logger.warning(
            "voice.tts is cartesia, but no Cartesia key is stored: Kokoro speaks. "
            "Store one with: yarvis key set cartesia"
        )
        return kokoro
    return CartesiaTTS(
        key.value,
        kokoro,
        voice=config.cartesia_voice,
        model=config.cartesia_model,
        speed=config.speech_rate,
    )
