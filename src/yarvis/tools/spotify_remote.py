"""Spotify's player, through its Web API: in the background, on any computer.

Telling the Spotify app what to do (AppleScript on a Mac) brings it to the front, and
exists only on macOS. Spotify's own Web API plays on a "device" (this computer's Spotify
app, a phone, a speaker) without touching any window, the same way on macOS and Windows.

It needs the user's consent for playback (``yarvis spotify login``, scopes
``user-read-playback-state`` and ``user-modify-playback-state``) and, since February 2026,
a developer app whose owner has Premium. Without that consent, callers fall back to what
the platform offers (AppleScript on a Mac), and tell the user how to allow it.

Yarvis plays on this computer: its Spotify app is found among the user's devices by the
computer's name, and started (hidden, where the platform allows it) when it is not open.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from yarvis.tools.base import ToolError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx2

    from yarvis.tools.spotify import SpotifyAccount

logger = logging.getLogger(__name__)

PLAYBACK_SCOPE: Final = "user-modify-playback-state"
_START_TRIES: Final = 12
_START_PAUSE_S: Final = 1.0
_PLAY_TRIES: Final = 3
_COMMANDS: Final[dict[str, tuple[Literal["PUT", "POST"], str]]] = {
    "pause": ("PUT", "me/player/pause"),
    "play": ("PUT", "me/player/play"),
    "next": ("POST", "me/player/next"),
    "previous": ("POST", "me/player/previous"),
}


class LoginNeededError(ToolError):
    """The user's Spotify login does not (yet) allow controlling playback."""


class NoDeviceError(ToolError):
    """No Spotify app (or speaker) is open to play on."""


_LOGIN_AGAIN: Final = (
    "Yarvis needs the user's permission to control Spotify playback. The user can give it "
    "once by running this in the terminal: yarvis spotify login"
)


@dataclass(frozen=True, slots=True)
class Device:
    """Somewhere Spotify can play: this computer's app, a phone, a speaker."""

    id: str
    name: str
    kind: str
    active: bool


@dataclass(frozen=True, slots=True)
class Playback:
    """What Spotify is doing."""

    playing: bool
    track: str
    by: str
    album: str
    position_s: int
    length_s: int
    volume: int | None
    shuffle: bool
    repeat: bool
    device: str


class SpotifyRemote:
    """Play, pause, skip and adjust Spotify through its Web API."""

    def __init__(
        self,
        account: SpotifyAccount,
        *,
        computer: str,
        start_app: Callable[[], Awaitable[None]],
        pause: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._account = account
        self._computer = _simple(computer)
        self._start_app = start_app
        self._pause = pause

    async def allowed(self) -> bool:
        """Whether the user's login allows controlling playback."""
        return await self._account.allows(PLAYBACK_SCOPE)

    async def devices(self) -> list[Device]:
        """Every device the user's Spotify can play on right now."""
        response = await self._account.call("GET", "me/player/devices")
        _check(response)
        found: list[Device] = []
        for item in _list(_dict(response.json()).get("devices")):
            device_id = item.get("id")
            if isinstance(device_id, str) and device_id:
                found.append(
                    Device(
                        device_id,
                        str(item.get("name", "")),
                        str(item.get("type", "")),
                        bool(item.get("is_active")),
                    )
                )
        return found

    async def play(self, uri: str) -> Device:
        """Play a track, album, artist or playlist on this computer."""
        device = await self.this_computer()
        body: dict[str, object] = (
            {"uris": [uri]} if uri.startswith("spotify:track:") else {"context_uri": uri}
        )
        tries = 0
        while True:
            response = await self._account.call(
                "PUT", "me/player/play", params={"device_id": device.id}, json=body
            )
            try:
                _check(response)
            except NoDeviceError:  # an app that just opened is listed a moment before it plays
                tries += 1
                if tries == _PLAY_TRIES:
                    raise
                await self._pause(_START_PAUSE_S)
            else:
                return device

    async def this_computer(self) -> Device:
        """This computer's Spotify app, started if it is not open yet."""
        found = self._pick(await self.devices())
        if found is not None:
            return found
        logger.info("Spotify is not open here: starting it")
        await self._start_app()
        for _ in range(_START_TRIES):
            await self._pause(_START_PAUSE_S)
            found = self._pick(await self.devices())
            if found is not None:
                return found
        msg = "Spotify did not open on this computer. Is the Spotify app installed?"
        raise NoDeviceError(msg)

    async def command(self, action: str) -> None:
        """``pause``, ``play`` (resume), ``next`` or ``previous``, on the device playing."""
        method, path = _COMMANDS[action]
        _check(await self._account.call(method, path))

    async def adjust(
        self,
        *,
        volume: int | None = None,
        shuffle: bool | None = None,
        repeat: bool | None = None,
        position_s: int | None = None,
        skip_s: int | None = None,
    ) -> None:
        """Change the volume, shuffle, repeat or position of what is playing."""
        if volume is not None:
            await self._put("me/player/volume", volume_percent=str(volume))
        if shuffle is not None:
            await self._put("me/player/shuffle", state=str(shuffle).lower())
        if repeat is not None:
            await self._put("me/player/repeat", state="context" if repeat else "off")
        if skip_s is not None:
            now = await self.state()
            position_s = max(0, (now.position_s if now else 0) + skip_s)
        if position_s is not None:
            await self._put("me/player/seek", position_ms=str(position_s * 1000))

    async def state(self) -> Playback | None:
        """What is playing, or None when Spotify is not playing anywhere."""
        response = await self._account.call("GET", "me/player")
        if response.status_code == 204:  # noqa: PLR2004 - "nothing is playing"
            return None
        _check(response)
        payload = _dict(response.json())
        item = _dict(payload.get("item"))
        if not item:
            return None
        artists = _list(item.get("artists"))
        device = _dict(payload.get("device"))
        volume = device.get("volume_percent")
        return Playback(
            playing=bool(payload.get("is_playing")),
            track=str(item.get("name", "")),
            by=", ".join(str(a.get("name", "")) for a in artists[:3]),
            album=str(_dict(item.get("album")).get("name", "")),
            position_s=_int(payload.get("progress_ms")) // 1000,
            length_s=_int(item.get("duration_ms")) // 1000,
            volume=volume if isinstance(volume, int) else None,
            shuffle=bool(payload.get("shuffle_state")),
            repeat=payload.get("repeat_state") in {"context", "track"},
            device=str(device.get("name", "")),
        )

    async def _put(self, path: str, **params: str) -> None:
        _check(await self._account.call("PUT", path, params=params))

    def _pick(self, devices: list[Device]) -> Device | None:
        """This computer, by name; or the only computer there is."""
        computers = [d for d in devices if d.kind.casefold() == "computer"]
        named = [d for d in computers if _simple(d.name) == self._computer]
        if named:
            return named[0]
        return computers[0] if len(computers) == 1 else None


def computer_name(node: str) -> str:
    """The name Spotify gives this computer ("Victors-MacBook-Air.local" → "...-Air")."""
    return node.removesuffix(".local").removesuffix(".lan")


def _simple(name: str) -> str:
    """Letters and digits only: "Victor's MacBook Air" and "Victors-MacBook-Air" match."""
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def _check(response: httpx2.Response) -> None:
    status = response.status_code
    if status < 400:  # noqa: PLR2004
        return
    error = _dict(_dict(_json(response)).get("error"))
    message = str(error.get("message", "")).casefold()
    reason = str(error.get("reason", ""))
    if status == 401 or (status == 403 and "scope" in message):  # noqa: PLR2004
        raise LoginNeededError(_LOGIN_AGAIN)
    if reason == "PREMIUM_REQUIRED":
        msg = "Controlling playback through Spotify's service needs Spotify Premium."
        raise ToolError(msg)
    if status == 404:  # noqa: PLR2004 - "no active device"
        msg = "Spotify is not playing on any device right now."
        raise NoDeviceError(msg)
    if status == 429:  # noqa: PLR2004
        msg = "Spotify is rate-limiting requests. Try again in a minute."
        raise ToolError(msg)
    logger.info("Spotify's player answered HTTP %d (%s)", status, reason or "no reason")
    msg = f"Spotify did not accept that (HTTP {status})."
    raise ToolError(msg)


def _json(response: httpx2.Response) -> object:
    try:
        return cast("object", response.json())
    except ValueError:
        return None


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in map(_dict, cast("list[object]", value)) if item]


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0
