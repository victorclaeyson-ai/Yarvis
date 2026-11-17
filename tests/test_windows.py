"""Windows: Start menu apps, windows to close, volume and media keys, websites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from test_computer import FakeRunner
from test_spotify import FakeAccount, FakeSpotify, account, catalog
from yarvis.tools.base import Tier, ToolError
from yarvis.tools.computer import AppArgs, MediaArgs, UrlArgs, VolumeArgs, environment
from yarvis.tools.desktop import computer_tools
from yarvis.tools.spotify import SpotifyPlay
from yarvis.tools.windows import (
    POWERSHELL,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    VK_MEDIA_PREV_TRACK,
    VK_VOLUME_DOWN,
    VK_VOLUME_UP,
    StartMenu,
    WindowsMedia,
    WindowsOpenApp,
    WindowsOpenWebsite,
    WindowsQuitApp,
    WindowsSetVolume,
    powershell,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

START_APPS = [
    {"Name": "Spotify", "AppID": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"},
    {"Name": "Google Chrome", "AppID": "Chrome"},
    {"Name": "Calculator", "AppID": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"},
    {"Name": "Notepad", "AppID": "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\notepad.exe"},
    {"Name": "Notepad", "AppID": "duplicate"},
    {"Name": "", "AppID": "nameless"},
    {"Name": "No ID"},
]


class Hands:
    """Records what would be opened and which keys pressed."""

    def __init__(self, *, broken: bool = False) -> None:
        self.opened: list[str] = []
        self.pressed: list[int] = []
        self.broken = broken

    async def launch(self, target: str) -> None:
        if self.broken:
            raise OSError("nope")
        self.opened.append(target)

    async def press(self, codes: Sequence[int]) -> None:
        self.pressed += codes


def answer(payload: Any) -> tuple[int, str]:
    return 0, json.dumps(payload)


# --------------------------------------------------------------------------- commands


def test_commands_see_what_they_need_and_no_secrets() -> None:
    environ = {
        "SystemRoot": "C:\\Windows",
        "TEMP": "C:\\Temp",
        "USERPROFILE": "C:\\Users\\victo",
        "ANTHROPIC_API_KEY": "sk-ant-secret",  # gitleaks:allow
        "APPDATA": "",
    }
    env = environment("win32", environ)
    assert env == {
        "SystemRoot": "C:\\Windows",
        "TEMP": "C:\\Temp",
        "USERPROFILE": "C:\\Users\\victo",
        "PATH": str(Path("C:\\Windows") / "System32"),
    }
    assert "ANTHROPIC_API_KEY" not in environment("darwin", environ)
    assert powershell("Get-Date") == [
        POWERSHELL,
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-Date",
    ]


# --------------------------------------------------------------------------- apps


async def test_apps_are_opened_from_the_start_menu() -> None:
    run = FakeRunner(answer(START_APPS))
    hands = Hands()
    tool = WindowsOpenApp(run, StartMenu(run), launch=hands.launch, press=hands.press)
    assert await tool.run(AppArgs(name="spotify")) == "Spotify is open."
    assert hands.opened == ["shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"]
    assert await tool.run(AppArgs(name="chrome")) == "Google Chrome is open."
    assert len(run.commands) == 1  # the list is kept for a while
    assert "Get-StartApps" in run.commands[0][-1]
    await tool.run(AppArgs(name="notepad"))
    assert hands.opened[-1].endswith("notepad.exe")  # the first one wins
    assert tool.describe(AppArgs(name="Spotify")) == "open Spotify"
    assert tool.tier is Tier.REVERSIBLE


@pytest.mark.parametrize(
    ("name", "message"),
    [("Calculater", r"No app called 'Calculater' is in the Start menu\. Did you mean Calculator\?"),
     ("o", "Several apps match 'o'")],
)  # fmt: skip
async def test_unknown_or_unclear_apps(name: str, message: str) -> None:
    run = FakeRunner(answer(START_APPS))
    hands = Hands()
    tool = WindowsOpenApp(run, StartMenu(run), launch=hands.launch, press=hands.press)
    with pytest.raises(ToolError, match=message):
        await tool.run(AppArgs(name=name))
    assert hands.opened == []


async def test_odd_answers_from_windows() -> None:
    hands = Hands()
    one = FakeRunner(answer(START_APPS[0]))  # one app: an object, not a list
    tool = WindowsOpenApp(one, StartMenu(one), launch=hands.launch, press=hands.press)
    await tool.run(AppArgs(name="Spotify"))
    for broken, message in (((1, "error"), "did not list its apps"), ((0, "<html>"), "No app")):
        run = FakeRunner(broken)
        tool = WindowsOpenApp(run, StartMenu(run), launch=hands.launch, press=hands.press)
        with pytest.raises(ToolError, match=message):
            await tool.run(AppArgs(name="Spotify"))
    run = FakeRunner(answer(START_APPS))
    stuck = Hands(broken=True)
    tool = WindowsOpenApp(run, StartMenu(run), launch=stuck.launch, press=stuck.press)
    with pytest.raises(ToolError, match="would not open"):
        await tool.run(AppArgs(name="Spotify"))


async def test_the_app_list_is_refreshed_after_a_while() -> None:
    now = [0.0]
    run = FakeRunner(answer(START_APPS), answer([{"Name": "Slack", "AppID": "Slack"}]))
    menu = StartMenu(run, clock=lambda: now[0])
    assert "Spotify" in await menu.names()
    now[0] = 301.0
    assert list(await menu.names()) == ["Slack"]


# --------------------------------------------------------------------------- quitting


WINDOWS = [
    {"Id": 4242, "ProcessName": "chrome", "Description": "Google Chrome"},
    {"Id": 77, "ProcessName": "Spotify", "Description": "Spotify"},
    {"Id": 0, "ProcessName": "Idle"},
    {"Id": "x", "ProcessName": "odd"},
]


async def test_quitting_closes_the_window_after_a_yes() -> None:
    run = FakeRunner(answer(WINDOWS))
    hands = Hands()
    tool = WindowsQuitApp(run, launch=hands.launch, press=hands.press)
    assert tool.tier is Tier.EXTERNAL
    assert await tool.run(AppArgs(name="chrome")) == "chrome was asked to close."
    assert run.commands[-1][-1] == "[void](Get-Process -Id 4242).CloseMainWindow()"
    assert tool.describe(AppArgs(name="Chrome")) == "quit Chrome"
    run = FakeRunner(answer(WINDOWS))
    await WindowsQuitApp(run, launch=hands.launch, press=hands.press).run(
        AppArgs(name="Google Chrome")
    )
    assert run.commands[-1][-1].endswith("-Id 4242).CloseMainWindow()")


@pytest.mark.parametrize(
    ("answers", "name", "message"),
    [
        ([(1, "error")], "chrome", "did not list the open apps"),
        ([answer(WINDOWS)], "Word", "No app called 'Word' has a window open"),
        ([answer(WINDOWS), (1, "denied")], "Spotify", "Spotify would not quit"),
    ],
)
async def test_quitting_problems(answers: list[tuple[int, str]], name: str, message: str) -> None:
    hands = Hands()
    tool = WindowsQuitApp(FakeRunner(*answers), launch=hands.launch, press=hands.press)
    with pytest.raises(ToolError, match=message):
        await tool.run(AppArgs(name=name))


# --------------------------------------------------------------------------- keys


async def test_volume_with_the_volume_keys() -> None:
    hands = Hands()
    tool = WindowsSetVolume(FakeRunner(), launch=hands.launch, press=hands.press)
    assert await tool.run(VolumeArgs(percent=40)) == "Volume is 40%."
    assert hands.pressed == [VK_VOLUME_DOWN] * 50 + [VK_VOLUME_UP] * 20
    assert tool.describe(VolumeArgs(percent=40)) == "set the volume to 40%"


@pytest.mark.parametrize(
    ("action", "key", "said"),
    [
        ("play", VK_MEDIA_PLAY_PAUSE, "play or pause the music"),
        ("pause", VK_MEDIA_PLAY_PAUSE, "play or pause the music"),
        ("next", VK_MEDIA_NEXT_TRACK, "skip to the next track"),
        ("previous", VK_MEDIA_PREV_TRACK, "go back a track"),
    ],
)
async def test_media_keys(action: str, key: int, said: str) -> None:
    hands = Hands()
    tool = WindowsMedia(FakeRunner(), launch=hands.launch, press=hands.press)
    args = MediaArgs.model_validate({"action": action})
    assert await tool.run(args) == "Done."
    assert hands.pressed == [key]
    assert tool.describe(args) == said
    assert tool.starts_playback(args) is (action != "pause")


# --------------------------------------------------------------------------- websites


async def test_websites_open_in_the_default_browser() -> None:
    hands = Hands()
    tool = WindowsOpenWebsite(FakeRunner(), launch=hands.launch)
    assert await tool.run(UrlArgs(url="https://www.kuleuven.be/en")) == "Opened www.kuleuven.be."
    assert hands.opened == ["https://www.kuleuven.be/en"]
    with pytest.raises(ToolError, match="Only https"):
        await tool.run(UrlArgs(url="file:///C:/Windows/System32/calc.exe"))
    stuck = WindowsOpenWebsite(FakeRunner(), launch=Hands(broken=True).launch)
    with pytest.raises(ToolError, match="would not open"):
        await stuck.run(UrlArgs(url="https://example.com"))
    assert tool.sends_data_out(UrlArgs(url="https://example.com"))


# --------------------------------------------------------------------------- Spotify


def test_spotify_on_windows_plays_through_its_service() -> None:
    keys = catalog(FakeSpotify())
    tools = computer_tools(FakeRunner(), platform="win32", spotify=keys, node="DESKTOP-4070")
    names = [tool.name for tool in tools]
    assert names[-3:] == ["spotify_play", "spotify_now_playing", "spotify_adjust"]
    play = next(t for t in tools if isinstance(t, SpotifyPlay))
    assert play._run is None  # pyright: ignore[reportPrivateUsage] - no AppleScript here
    assert play._remote is None  # pyright: ignore[reportPrivateUsage] - not logged in yet
    user = account(FakeAccount())
    tools = computer_tools(
        FakeRunner(), platform="win32", spotify=keys, spotify_account=user, node="DESKTOP-4070"
    )
    play = next(t for t in tools if isinstance(t, SpotifyPlay))
    assert play._remote is not None  # pyright: ignore[reportPrivateUsage]
    assert [t.name for t in tools][-1] == "spotify_my_playlists"
