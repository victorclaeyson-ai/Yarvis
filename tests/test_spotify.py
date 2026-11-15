from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from pydantic import SecretStr

from test_computer import FakeRunner
from yarvis.security.redact import default_redactor
from yarvis.tools.base import ToolError
from yarvis.tools.computer import OPEN
from yarvis.tools.spotify import (
    SEARCH_URL,
    TOKEN_URL,
    AdjustArgs,
    NoArgs,
    PlayArgs,
    SpotifyAdjust,
    SpotifyCatalog,
    SpotifyNowPlaying,
    SpotifyPlay,
    spotify_tools,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CLIENT_ID = "0123456789abcdef0123456789abcdef"
SECRET = "fedcba9876543210fedcba9876543210"  # noqa: S105  # gitleaks:allow - a test value
TRACK = "spotify:track:" + "A" * 22


def track(uri: str = TRACK, name: str = "Feel What You Want") -> dict[str, Any]:
    return {"uri": uri, "name": name, "artists": [{"name": "Kristine W"}]}


class FakeSpotify:
    """Spotify's token and search endpoints, scripted."""

    def __init__(self, *searches: dict[str, Any], token_status: int = 200) -> None:
        self.searches = list(searches)
        self.token_status = token_status
        self.tokens = 0
        self.queries: list[str] = []
        self.status = 200

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == TOKEN_URL:
            self.tokens += 1
            assert request.headers["authorization"].startswith("Basic ")
            assert request.content == b"grant_type=client_credentials"
            body = {"access_token": f"token-{self.tokens}-abcdefghijklmnop", "expires_in": 3600}
            return httpx2.Response(self.token_status, json=body)
        assert str(request.url).startswith(SEARCH_URL)
        assert request.headers["authorization"].startswith("Bearer token-")
        self.queries.append(request.url.params["q"])
        assert request.url.params["limit"] == "5"
        payload = self.searches.pop(0) if self.searches else {}
        return httpx2.Response(self.status, json=payload)


def catalog(fake: FakeSpotify, clock: Callable[[], float] = lambda: 0.0) -> SpotifyCatalog:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake.handler))
    return SpotifyCatalog(CLIENT_ID, SecretStr(SECRET), http=http, clock=clock)


async def test_finding_a_song() -> None:
    fake = FakeSpotify({"tracks": {"items": [None, {"uri": "bad"}, track()]}})
    found = await catalog(fake).find("track", 'Feel What "You" Want', "Kristine W")
    assert found is not None
    assert (found.uri, found.name, found.by) == (TRACK, "Feel What You Want", "Kristine W")
    assert fake.queries == ['track:"Feel What You Want" artist:"Kristine W"']
    assert SECRET not in default_redactor.redact(f"secret {SECRET}")


async def test_a_looser_search_when_the_strict_one_finds_nothing() -> None:
    fake = FakeSpotify({"tracks": {"items": []}}, {"tracks": {"items": [track()]}})
    assert await catalog(fake).find("track", "Feel What U Want", "Kristine") is not None
    assert fake.queries[1] == "Feel What U Want Kristine"


@pytest.mark.parametrize(
    ("kind", "title", "artist", "query"),
    [
        ("artist", "Kristine W", "", 'artist:"Kristine W"'),
        ("album", "Stronger", "Kristine W", 'album:"Stronger" artist:"Kristine W"'),
        ("playlist", "90s dance", "anyone", "90s dance"),
    ],
)
async def test_queries_per_kind(kind: Any, title: str, artist: str, query: str) -> None:
    fake = FakeSpotify({f"{kind}s": {"items": []}}, {f"{kind}s": {"items": []}})
    assert await catalog(fake).find(kind, title, artist) is None
    assert fake.queries[0] == query


async def test_playlists_are_by_their_owner() -> None:
    playlist = {
        "uri": "spotify:playlist:" + "B" * 22,
        "name": "Dance",
        "owner": {"display_name": "Spotify"},
    }
    found = await catalog(FakeSpotify({"playlists": {"items": [playlist]}})).find(
        "playlist", "Dance"
    )
    assert found is not None and found.by == "Spotify"


async def test_the_token_is_reused_until_it_expires() -> None:
    now = [0.0]
    fake = FakeSpotify(*[{"tracks": {"items": [track()]}}] * 3)
    spotify = catalog(fake, clock=lambda: now[0])
    await spotify.find("track", "a")
    await spotify.find("track", "b")
    assert fake.tokens == 1
    now[0] = 3600.0
    await spotify.find("track", "c")
    assert fake.tokens == 2


@pytest.mark.parametrize(
    ("token_status", "search_status", "message"),
    [
        (401, 200, "rejected the client ID or secret"),
        (200, 403, "needs Premium"),
        (200, 429, "rate-limiting"),
        (200, 500, "HTTP 500"),
    ],
)
async def test_spotify_errors(token_status: int, search_status: int, message: str) -> None:
    fake = FakeSpotify({}, token_status=token_status)
    fake.status = search_status
    with pytest.raises(ToolError, match=message):
        await catalog(fake).find("track", "x")


async def test_no_network() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline")

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(broken))
    with pytest.raises(ToolError, match="could not be reached"):
        await SpotifyCatalog(CLIENT_ID, SecretStr(SECRET), http=http).find("track", "x")

    calls = 0

    def token_only(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx2.Response(200, json={"access_token": "t" * 20, "expires_in": 60})
        raise httpx2.ConnectError("offline")

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(token_only))
    with pytest.raises(ToolError, match="could not be reached"):
        await SpotifyCatalog(CLIENT_ID, SecretStr(SECRET), http=http).find("track", "x")


def test_credentials_from_the_keychain_value() -> None:
    spotify = SpotifyCatalog.from_secret(SecretStr(f"{CLIENT_ID}:{SECRET}"))
    assert spotify._client_id == CLIENT_ID  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- the tools


async def test_play_finds_and_plays() -> None:
    run = FakeRunner()
    tool = SpotifyPlay(run, catalog(FakeSpotify({"tracks": {"items": [track()]}})))
    args = PlayArgs(title="Feel What You Want", artist="Kristine W")
    assert await tool.run(args) == "Playing the track 'Feel What You Want' by Kristine W."
    assert run.commands[0][-1] == TRACK  # the URI travels as data
    assert all(TRACK not in part for part in run.commands[0][:-1])
    assert tool.describe(args) == "play Feel What You Want by Kristine W on Spotify"
    assert tool.untrusted_output


async def test_play_without_credentials_opens_a_search() -> None:
    run = FakeRunner()
    result = await SpotifyPlay(run, None).run(
        PlayArgs(title="Feel What You Want", artist="Kristine W")
    )
    assert run.commands == [[OPEN, "spotify:search:Feel%20What%20You%20Want%20Kristine%20W"]]
    assert "yarvis key set spotify" in result


async def test_play_failures() -> None:
    nothing = SpotifyPlay(FakeRunner(), catalog(FakeSpotify({"tracks": {"items": []}})))
    with pytest.raises(ToolError, match="found no track"):
        await nothing.run(PlayArgs(title="zzz"))
    broken = SpotifyPlay(
        FakeRunner((1, "error")), catalog(FakeSpotify({"tracks": {"items": [track()]}}))
    )
    with pytest.raises(ToolError, match="did not start playing"):
        await broken.run(PlayArgs(title="x"))


async def test_now_playing() -> None:
    answer = "\x1f".join(
        [
            "playing",
            "Feel What You Want",
            "Kristine W",
            "Stronger",
            "83",
            "401",
            "70",
            "false",
            "true",
        ]
    )
    tool = SpotifyNowPlaying(FakeRunner((0, answer)))
    assert await tool.run(NoArgs()) == (
        "playing: 'Feel What You Want' by Kristine W (album 'Stronger'), 1:23 of 6:41. "
        "Spotify volume 70%, shuffle off, repeat on."
    )
    assert (
        await SpotifyNowPlaying(FakeRunner((0, "stopped"))).run(NoArgs())
        == "Spotify is not playing anything."
    )
    with pytest.raises(ToolError, match="unexpected"):
        await SpotifyNowPlaying(FakeRunner((0, "a\x1fb"))).run(NoArgs())
    with pytest.raises(ToolError, match="did not answer"):
        await SpotifyNowPlaying(FakeRunner((1, ""))).run(NoArgs())
    assert tool.describe(NoArgs()) == "check what's playing"


async def test_adjust() -> None:
    run = FakeRunner((0, "ok"))
    tool = SpotifyAdjust(run)
    args = AdjustArgs(volume=30, shuffle=True, skip_seconds=-15)
    assert await tool.run(args) == (
        "Done: set Spotify's volume to 30%, turn shuffle on, skip back 15 seconds."
    )
    assert run.commands[0][-5:] == ["30", "true", "", "", "-15"]
    both = AdjustArgs(repeat=False, position_seconds=90)
    assert tool.describe(both) == "turn repeat off, jump to 1:30"
    with pytest.raises(ToolError, match="not running"):
        await SpotifyAdjust(FakeRunner((0, "not running"))).run(args)
    with pytest.raises(ToolError, match="did not accept"):
        await SpotifyAdjust(FakeRunner((1, ""))).run(args)
    with pytest.raises(ValueError, match="at least one"):
        AdjustArgs()
    with pytest.raises(ValueError, match="validation error"):
        AdjustArgs(volume=101)


def test_spotify_tools() -> None:
    assert [tool.name for tool in spotify_tools(FakeRunner(), None)] == [
        "spotify_play", "spotify_now_playing", "spotify_adjust",
    ]  # fmt: skip
    assert json.dumps(SpotifyAdjust(FakeRunner()).spec().input_schema)  # serialisable for the API
