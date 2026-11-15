"""Spotify: play anything by name, and control the player.

Finding music uses Spotify's Web API with an app's client ID and secret ("client
credentials": no user login, search only). Playing and controlling uses the Spotify Mac
app's AppleScript interface, with values passed as data (``argv``), never as script text.

Without credentials, "play X" opens a Spotify search for X instead.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast
from urllib.parse import quote

import httpx2
from pydantic import Field, SecretStr, model_validator

from yarvis.security.redact import default_redactor
from yarvis.tools.base import Arguments, Tier, Tool, ToolError
from yarvis.tools.computer import OPEN, osascript

if TYPE_CHECKING:
    from yarvis.tools.computer import Runner

logger = logging.getLogger(__name__)

TOKEN_URL: Final = "https://accounts.spotify.com/api/token"  # noqa: S105 - a URL, not a secret
SEARCH_URL: Final = "https://api.spotify.com/v1/search"
_URI: Final = re.compile(r"^spotify:(track|album|artist|playlist):[A-Za-z0-9]{22}$")
_FIELD: Final = "\x1f"

Kind = Literal["track", "album", "artist", "playlist"]


@dataclass(frozen=True, slots=True)
class Found:
    """A search result that can be played."""

    uri: str
    name: str
    by: str


class SpotifyCatalog:
    """Searches Spotify's catalogue (Web API, client-credentials flow)."""

    def __init__(
        self,
        client_id: str,
        client_secret: SecretStr,
        *,
        http: httpx2.AsyncClient | None = None,
        clock: Any = time.monotonic,  # noqa: ANN401 - a () -> float callable
    ) -> None:
        self._client_id = client_id
        self._secret = client_secret
        self._http = http or httpx2.AsyncClient(timeout=10.0, follow_redirects=False)
        self._clock = clock
        self._token: str | None = None
        self._expires = 0.0
        default_redactor.register(client_secret.get_secret_value())

    @classmethod
    def from_secret(cls, credentials: SecretStr) -> Self:
        """From the keychain value ``client_id:client_secret``."""
        client_id, _, secret = credentials.get_secret_value().partition(":")
        return cls(client_id, SecretStr(secret))

    async def find(self, kind: Kind, title: str, artist: str = "") -> Found | None:
        """The best match, or None."""
        strict = _query(kind, title, artist)
        found = await self._search(kind, strict)
        if found is None and artist:
            found = await self._search(kind, f"{title} {artist}")  # looser, as a fallback
        return found

    async def _search(self, kind: Kind, query: str) -> Found | None:
        token = await self._bearer()
        response = await self._get(
            SEARCH_URL,
            params={"q": query, "type": kind, "limit": "5"},
            headers={"Authorization": f"Bearer {token}"},
        )
        for item in _items(response.json(), f"{kind}s"):
            uri = str(item.get("uri", ""))
            if _URI.match(uri):
                return Found(uri, str(item.get("name", "")), _by(item))
        return None

    async def _bearer(self) -> str:
        if self._token is not None and self._clock() < self._expires:
            return self._token
        try:
            response = await self._http.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._secret.get_secret_value()),
            )
        except httpx2.HTTPError as exc:
            raise ToolError(_UNREACHABLE) from exc
        if response.status_code in {400, 401}:
            msg = (
                "Spotify rejected the client ID or secret. Store them again: yarvis key set spotify"
            )
            raise ToolError(msg)
        _check(response)
        payload = response.json()
        self._token = str(payload["access_token"])
        default_redactor.register(self._token)
        self._expires = self._clock() + float(payload.get("expires_in", 3600)) - 60
        return self._token

    async def _get(self, url: str, **kwargs: Any) -> httpx2.Response:  # noqa: ANN401
        try:
            response = await self._http.get(url, **kwargs)
        except httpx2.HTTPError as exc:
            raise ToolError(_UNREACHABLE) from exc
        _check(response)
        return response


_UNREACHABLE: Final = "Spotify's search could not be reached. Check the internet connection."


def _check(response: httpx2.Response) -> None:
    if response.status_code == 403:  # noqa: PLR2004
        msg = (
            "Spotify refused the search. Since February 2026 the owner of the Spotify "
            "developer app needs Premium."
        )
        raise ToolError(msg)
    if response.status_code == 429:  # noqa: PLR2004
        msg = "Spotify is rate-limiting searches. Try again in a minute."
        raise ToolError(msg)
    if response.status_code >= 400:  # noqa: PLR2004
        msg = f"Spotify search failed (HTTP {response.status_code})."
        raise ToolError(msg)


def _query(kind: Kind, title: str, artist: str) -> str:
    def clean(text: str) -> str:
        return text.replace('"', "").strip()

    if kind == "artist":
        return f'artist:"{clean(title)}"'
    query = f'{kind}:"{clean(title)}"' if kind in {"track", "album"} else clean(title)
    return f'{query} artist:"{clean(artist)}"' if artist and kind != "playlist" else query


def _items(payload: object, key: str) -> list[dict[str, object]]:
    """``payload[key]["items"]``, keeping only objects (the API may return nulls)."""
    section = _dict(payload).get(key)
    items: object = _dict(section).get("items")
    if not isinstance(items, list):
        return []
    return [found for found in map(_dict, cast("list[object]", items)) if found]


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _by(item: dict[str, object]) -> str:
    artists = item.get("artists")
    if isinstance(artists, list):
        names = [str(_dict(a).get("name", "")) for a in cast("list[object]", artists)[:3]]
        return ", ".join(name for name in names if name)
    return str(_dict(item.get("owner")).get("display_name", ""))


# ----------------------------------------------------------------------------- tools


class PlayArgs(Arguments):
    """Arguments of ``spotify_play``."""

    title: str = Field(min_length=1, max_length=200, description="Song, album, artist or playlist.")
    artist: str = Field(default="", max_length=100, description="Artist, if known.")
    kind: Kind = "track"


class AdjustArgs(Arguments):
    """Arguments of ``spotify_adjust``: set any of them."""

    volume: int | None = Field(default=None, ge=0, le=100, description="Spotify's own volume.")
    shuffle: bool | None = None
    repeat: bool | None = None
    position_seconds: int | None = Field(default=None, ge=0, description="Jump to this point.")
    skip_seconds: int | None = Field(
        default=None, ge=-600, le=600, description="Jump forward (or back, if negative)."
    )

    @model_validator(mode="after")
    def _something(self) -> Self:
        if all(value is None for value in self.model_dump().values()):
            msg = "set at least one setting"
            raise ValueError(msg)
        return self


class NoArgs(Arguments):
    """No arguments."""


_PLAY: Final = (
    "on run argv",
    "set theURI to item 1 of argv",
    'tell application "Spotify"',
    "activate",
    "play track theURI",
    "end tell",
    "end run",
)


class SpotifyPlay(Tool[PlayArgs]):
    """Play something by name."""

    name = "spotify_play"
    description = (
        "Play a song (default), album, artist or playlist on Spotify by name, e.g. title "
        "'Feel What You Want', artist 'Kristine W'."
    )
    tier = Tier.REVERSIBLE
    Args = PlayArgs
    untrusted_output = True  # names come from Spotify's catalogue

    def __init__(self, run: Runner, catalog: SpotifyCatalog | None) -> None:
        self._run = run
        self._catalog = catalog

    async def run(self, args: PlayArgs) -> str:
        """Find it and play it."""
        if self._catalog is None:
            text = f"{args.title} {args.artist}".strip()
            await self._run([OPEN, f"spotify:search:{quote(text)}"])
            return (
                f"Opened a Spotify search for '{text}'. To play songs directly, the user can "
                "add Spotify credentials: yarvis key set spotify."
            )
        found = await self._catalog.find(args.kind, args.title, args.artist)
        if found is None:
            msg = f"Spotify found no {args.kind} called {args.title!r}."
            raise ToolError(msg)
        code, output = await self._run(osascript(_PLAY, found.uri))
        if code != 0:
            logger.warning("Spotify did not play: %s", output[:200])
            msg = "Spotify did not start playing. Is the Spotify app installed?"
            raise ToolError(msg)
        by = f" by {found.by}" if found.by else ""
        return f"Playing the {args.kind} '{found.name}'{by}."

    def describe(self, args: PlayArgs) -> str:
        """'play Feel What You Want by Kristine W on Spotify'."""
        by = f" by {args.artist}" if args.artist else ""
        return f"play {args.title}{by} on Spotify"


_NOW: Final = (
    'if application "Spotify" is not running then return "not running"',
    'tell application "Spotify"',
    "set s to player state as text",
    'if s is "stopped" then return "stopped"',
    "set t to current track",
    "set f to character id 31",
    "return s & f & (name of t) & f & (artist of t) & f & (album of t) & f & "
    "((player position) as integer) & f & ((duration of t) div 1000) & f & sound volume & f & "
    "shuffling & f & repeating",
    "end tell",
)


class SpotifyNowPlaying(Tool[NoArgs]):
    """What is playing."""

    name = "spotify_now_playing"
    description = "What Spotify is playing, with position, volume, shuffle and repeat."
    tier = Tier.READ
    Args = NoArgs
    untrusted_output = True

    def __init__(self, run: Runner) -> None:
        self._run = run

    async def run(self, args: NoArgs) -> str:
        """Ask Spotify."""
        code, output = await self._run(osascript(_NOW))
        if code != 0:
            msg = "Spotify did not answer."
            raise ToolError(msg)
        if output in {"not running", "stopped"}:
            return "Spotify is not playing anything."
        fields = output.split(_FIELD)
        if len(fields) != 9:  # noqa: PLR2004
            msg = "Spotify gave an unexpected answer."
            raise ToolError(msg)
        state, name, artist, album, position, length, volume, shuffle, repeat = fields
        return (
            f"{state}: '{name}' by {artist} (album '{album}'), {_clock(position)} of "
            f"{_clock(length)}. Spotify volume {volume}%, shuffle {_on(shuffle)}, "
            f"repeat {_on(repeat)}."
        )

    def describe(self, args: NoArgs) -> str:
        """'check what's playing'."""
        return "check what's playing"


_ADJUST: Final = (
    "on run argv",
    'if application "Spotify" is not running then return "not running"',
    "set {theVolume, theShuffle, theRepeat, thePosition, theSkip} to argv",
    'tell application "Spotify"',
    'if theVolume is not "" then set sound volume to (theVolume as integer)',
    'if theShuffle is not "" then set shuffling to (theShuffle is "true")',
    'if theRepeat is not "" then set repeating to (theRepeat is "true")',
    'if thePosition is not "" then set player position to (thePosition as integer)',
    'if theSkip is not "" then set player position to (player position + (theSkip as integer))',
    "end tell",
    'return "ok"',
    "end run",
)


class SpotifyAdjust(Tool[AdjustArgs]):
    """Spotify's volume, shuffle, repeat and position."""

    name = "spotify_adjust"
    description = (
        "Change Spotify's own volume (0-100), shuffle, repeat, or the position in the song "
        "(jump to a time, or skip forward/back by seconds)."
    )
    tier = Tier.REVERSIBLE
    Args = AdjustArgs

    def __init__(self, run: Runner) -> None:
        self._run = run

    async def run(self, args: AdjustArgs) -> str:
        """Apply the settings."""
        values = [
            "" if args.volume is None else str(args.volume),
            "" if args.shuffle is None else str(args.shuffle).lower(),
            "" if args.repeat is None else str(args.repeat).lower(),
            "" if args.position_seconds is None else str(args.position_seconds),
            "" if args.skip_seconds is None else str(args.skip_seconds),
        ]
        code, output = await self._run(osascript(_ADJUST, *values))
        if code != 0:
            msg = "Spotify did not accept that."
            raise ToolError(msg)
        if output == "not running":
            msg = "Spotify is not running."
            raise ToolError(msg)
        return "Done: " + self.describe(args) + "."

    def describe(self, args: AdjustArgs) -> str:
        """'set Spotify's volume to 30%, turn shuffle on'."""
        parts: list[str] = []
        if args.volume is not None:
            parts.append(f"set Spotify's volume to {args.volume}%")
        if args.shuffle is not None:
            parts.append(f"turn shuffle {_on(str(args.shuffle).lower())}")
        if args.repeat is not None:
            parts.append(f"turn repeat {_on(str(args.repeat).lower())}")
        if args.position_seconds is not None:
            parts.append(f"jump to {_clock(str(args.position_seconds))}")
        if args.skip_seconds is not None:
            way = "forward" if args.skip_seconds >= 0 else "back"
            parts.append(f"skip {way} {abs(args.skip_seconds)} seconds")
        return ", ".join(parts)


def _on(flag: str) -> str:
    return "on" if flag == "true" else "off"


def _clock(seconds: str) -> str:
    try:
        total = int(float(seconds))
    except ValueError:
        return seconds
    return f"{total // 60}:{total % 60:02d}"


def spotify_tools(run: Runner, catalog: SpotifyCatalog | None) -> list[Tool[Any]]:
    """All Spotify tools."""
    return [SpotifyPlay(run, catalog), SpotifyNowPlaying(run), SpotifyAdjust(run)]
