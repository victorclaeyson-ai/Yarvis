"""Spotify's player through its Web API, against a scripted fake of the API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from pydantic import SecretStr

from yarvis.tools.base import ToolError
from yarvis.tools.spotify import TOKEN_URL, SpotifyAccount
from yarvis.tools.spotify_remote import (
    Device,
    LoginNeededError,
    NoDeviceError,
    Playback,
    SpotifyRemote,
    computer_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable

API = "https://api.spotify.com/v1/"
TRACK = "spotify:track:" + "A" * 22
PLAYLIST = "spotify:playlist:" + "B" * 22
FULL = "playlist-read-private user-read-playback-state user-modify-playback-state"

MAC = {"id": "mac", "name": "Victor\u2019s MacBook Air", "type": "Computer", "is_active": False}
PHONE = {"id": "phone", "name": "iPhone", "type": "Smartphone", "is_active": True}
PC = {"id": "pc", "name": "DESKTOP-4070", "type": "Computer", "is_active": False}


class FakePlayer:
    """Spotify's token endpoint and player API, scripted."""

    def __init__(
        self,
        *device_lists: list[dict[str, Any]],
        scope: str | None = FULL,
        answers: dict[str, list[tuple[int, Any]]] | None = None,
    ) -> None:
        self.device_lists = list(device_lists) or [[MAC, PHONE]]
        self.scope = scope
        self.answers = answers or {}
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == TOKEN_URL:
            body: dict[str, Any] = {"access_token": "player-token-abcdefghijk", "expires_in": 3600}
            if self.scope is not None:
                body["scope"] = self.scope
            return httpx2.Response(200, json=body)
        assert request.headers["authorization"] == "Bearer player-token-abcdefghijk"
        path = str(request.url.path).removeprefix("/v1/")
        sent = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, dict(request.url.params), sent))
        scripted = self.answers.get(f"{request.method} {path}")
        if scripted:
            status, payload = scripted.pop(0)
            return (
                httpx2.Response(status, json=payload)
                if payload is not None
                else (httpx2.Response(status))
            )
        if path == "me/player/devices":
            devices = (
                self.device_lists.pop(0) if len(self.device_lists) > 1 else self.device_lists[0]
            )
            return httpx2.Response(200, json={"devices": devices})
        return httpx2.Response(204)

    def sent(self, method: str, path: str) -> list[tuple[dict[str, str], Any]]:
        return [(params, body) for m, p, params, body in self.calls if (m, p) == (method, path)]


def remote(
    fake: FakePlayer, *, computer: str = "Victors-MacBook-Air", started: list[str] | None = None
) -> tuple[SpotifyRemote, list[float]]:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake.handler))
    account = SpotifyAccount(
        "c" * 32, SecretStr("s" * 32), SecretStr("refresh-token-xyz"), http=http
    )
    waited: list[float] = []

    async def start() -> None:
        (started if started is not None else []).append("started")

    async def pause(seconds: float) -> None:
        waited.append(seconds)

    return SpotifyRemote(account, computer=computer, start_app=start, pause=pause), waited


def error(status: int, message: str, reason: str = "") -> tuple[int, Any]:
    body: dict[str, Any] = {"status": status, "message": message}
    if reason:
        body["reason"] = reason
    return status, {"error": body}


# --------------------------------------------------------------------------- permission


@pytest.mark.parametrize(
    ("scope", "allowed"),
    [(FULL, True), ("playlist-read-private", False), (None, True)],  # None: Spotify did not say
)
async def test_playback_needs_the_users_permission(scope: str | None, allowed: bool) -> None:
    player, _ = remote(FakePlayer(scope=scope))
    assert await player.allowed() is allowed


# --------------------------------------------------------------------------- playing


async def test_a_track_plays_on_this_computer_not_the_phone() -> None:
    fake = FakePlayer([PHONE, MAC, PC])
    player, _ = remote(fake)
    device = await player.play(TRACK)
    assert device == Device("mac", "Victor\u2019s MacBook Air", "Computer", active=False)
    assert fake.sent("PUT", "me/player/play") == [({"device_id": "mac"}, {"uris": [TRACK]})]


async def test_albums_artists_and_playlists_play_as_a_context() -> None:
    fake = FakePlayer([PC])
    player, _ = remote(fake, computer=computer_name("desktop-4070"))
    await player.play(PLAYLIST)
    assert fake.sent("PUT", "me/player/play") == [({"device_id": "pc"}, {"context_uri": PLAYLIST})]


async def test_the_only_computer_is_this_one() -> None:
    fake = FakePlayer([PHONE, {**PC, "name": "Gaming rig"}])
    player, _ = remote(fake, computer="DESKTOP-4070")
    assert (await player.play(TRACK)).id == "pc"


async def test_a_closed_spotify_is_opened_and_waited_for() -> None:
    started: list[str] = []
    fake = FakePlayer([PHONE], [PHONE], [PHONE, MAC])
    player, waited = remote(fake, started=started)
    assert (await player.play(TRACK)).id == "mac"
    assert started == ["started"]
    assert waited == [1.0, 1.0]


async def test_spotify_that_never_opens() -> None:
    player, waited = remote(FakePlayer([PHONE]))
    with pytest.raises(NoDeviceError, match="did not open on this computer"):
        await player.play(TRACK)
    assert len(waited) == 12


async def test_an_app_that_just_opened_gets_a_moment() -> None:
    fake = FakePlayer([MAC], answers={"PUT me/player/play": [error(404, "No active device")] * 2})
    player, waited = remote(fake)
    await player.play(TRACK)
    assert len(fake.sent("PUT", "me/player/play")) == 3
    assert waited == [1.0, 1.0]
    stuck = FakePlayer([MAC], answers={"PUT me/player/play": [error(404, "No active device")] * 3})
    with pytest.raises(NoDeviceError):
        await remote(stuck)[0].play(TRACK)


async def test_devices_without_an_id_are_skipped() -> None:
    fake = FakePlayer([{"name": "Restricted", "type": "Computer"}, None, MAC])  # type: ignore[list-item]
    player, _ = remote(fake)
    assert [d.id for d in await player.devices()] == ["mac"]


# --------------------------------------------------------------------------- controls


@pytest.mark.parametrize(
    ("action", "call"),
    [
        ("pause", ("PUT", "me/player/pause")),
        ("play", ("PUT", "me/player/play")),
        ("next", ("POST", "me/player/next")),
        ("previous", ("POST", "me/player/previous")),
    ],
)
async def test_commands(action: str, call: tuple[str, str]) -> None:
    fake = FakePlayer()
    await remote(fake)[0].command(action)
    assert [(m, p) for m, p, _, _ in fake.calls] == [call]


async def test_adjusting_volume_shuffle_repeat_and_position() -> None:
    fake = FakePlayer(
        answers={"GET me/player": [(200, {"progress_ms": 61_500, "item": {"name": "x"}})]}
    )
    player, _ = remote(fake)
    await player.adjust(volume=30, shuffle=True, repeat=False, skip_s=-15)
    assert [(p, params) for m, p, params, _ in fake.calls if m == "PUT"] == [
        ("me/player/volume", {"volume_percent": "30"}),
        ("me/player/shuffle", {"state": "true"}),
        ("me/player/repeat", {"state": "off"}),
        ("me/player/seek", {"position_ms": "46000"}),
    ]
    fake = FakePlayer(answers={"GET me/player": [(204, None)]})
    player, _ = remote(fake)
    await player.adjust(repeat=True, skip_s=-30)  # nothing playing: from the start
    assert [params for m, _, params, _ in fake.calls if m == "PUT"] == [
        {"state": "context"}, {"position_ms": "0"},
    ]  # fmt: skip
    fake = FakePlayer()
    await remote(fake)[0].adjust(position_s=90)
    assert fake.sent("PUT", "me/player/seek") == [({"position_ms": "90000"}, None)]


async def test_what_is_playing() -> None:
    state = {
        "is_playing": True,
        "progress_ms": 83_000,
        "shuffle_state": True,
        "repeat_state": "track",
        "device": {"name": "Victor\u2019s MacBook Air", "volume_percent": 64},
        "item": {
            "name": "Feel What You Want",
            "duration_ms": 245_000,
            "artists": [{"name": "Kristine W"}, None],
            "album": {"name": "Land of the Living"},
        },
    }
    fake = FakePlayer(answers={"GET me/player": [(200, state), (204, None), (200, {"item": None})]})
    player, _ = remote(fake)
    assert await player.state() == Playback(
        playing=True, track="Feel What You Want", by="Kristine W", album="Land of the Living",
        position_s=83, length_s=245, volume=64, shuffle=True, repeat=True,
        device="Victor\u2019s MacBook Air",
    )  # fmt: skip
    assert await player.state() is None  # 204: nothing anywhere
    assert await player.state() is None  # an advert or a podcast between items


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("answer", "problem", "message"),
    [
        (error(401, "The access token expired"), LoginNeededError, "yarvis spotify login"),
        (error(403, "Insufficient client scope"), LoginNeededError, "permission"),
        (error(403, "Player command failed: Premium required", "PREMIUM_REQUIRED"), ToolError,
         "needs Spotify Premium"),
        (error(404, "Player command failed: No active device found", "NO_ACTIVE_DEVICE"),
         NoDeviceError, "not playing on any device"),
        (error(429, "Too many requests"), ToolError, "rate-limiting"),
        (error(502, "Bad gateway"), ToolError, r"HTTP 502"),
        ((500, None), ToolError, r"HTTP 500"),
    ],
)  # fmt: skip
async def test_player_errors_say_what_to_do(
    answer: tuple[int, Any], problem: type[ToolError], message: str
) -> None:
    fake = FakePlayer(answers={"PUT me/player/pause": [answer]})
    with pytest.raises(problem, match=message):
        await remote(fake)[0].command("pause")


def test_this_computers_name() -> None:
    names: list[tuple[str, str]] = [
        ("Victors-MacBook-Air.local", "Victors-MacBook-Air"),
        ("DESKTOP-4070", "DESKTOP-4070"),
        ("pc.lan", "pc"),
    ]
    named: Callable[[str], str] = computer_name
    assert [named(node) for node, _ in names] == [expected for _, expected in names]
