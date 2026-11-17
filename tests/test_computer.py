from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

from yarvis.tools.base import ToolError
from yarvis.tools.computer import (
    OPEN,
    OSASCRIPT,
    AppArgs,
    Apps,
    Media,
    MediaArgs,
    OpenApp,
    OpenWebsite,
    QuitApp,
    SetVolume,
    UrlArgs,
    VolumeArgs,
    environment,
    run_command,
    safe_url,
    short_url,
)
from yarvis.tools.desktop import computer_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class FakeRunner:
    """Records commands; answers from a script (default: success, no output)."""

    def __init__(self, *answers: tuple[int, str]) -> None:
        self.answers = list(answers)
        self.commands: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.commands.append(list(argv))
        return self.answers.pop(0) if self.answers else (0, "")


@pytest.fixture
def apps(tmp_path: Path) -> Apps:
    applications = tmp_path / "Applications"
    for name in ("Safari", "Google Chrome", "Spotify", "Notes", "Numbers"):
        (applications / f"{name}.app").mkdir(parents=True)
    (applications / "Readme.app").write_text("not a bundle")
    return Apps([applications, tmp_path / "missing"])


# --------------------------------------------------------------------------- apps


def test_apps_are_found_by_name(apps: Apps, tmp_path: Path) -> None:
    assert apps.find("safari") == ("Safari", tmp_path / "Applications" / "Safari.app")
    assert apps.find("chrome")[0] == "Google Chrome"
    with pytest.raises(ToolError, match="Several apps match 'n': Notes, Numbers"):
        apps.find("n")
    with pytest.raises(ToolError, match=r"Did you mean Safari\?"):
        apps.find("Sfari")
    with pytest.raises(ToolError, match="No app called 'Readme'"):
        apps.find("Readme")


@pytest.mark.parametrize(
    "name", ["/Applications/Evil.app", "../../bin/sh", "Safari; rm -rf ~", "$(reboot)", ""]
)
def test_app_names_cannot_be_paths_or_commands(name: str) -> None:
    with pytest.raises(ValueError, match="validation error"):
        AppArgs(name=name)


async def test_open_app(apps: Apps, tmp_path: Path) -> None:
    run = FakeRunner()
    tool = OpenApp(run, apps)
    assert await tool.run(AppArgs(name="chrome")) == "Google Chrome is open."
    assert run.commands == [[OPEN, "-a", str(tmp_path / "Applications" / "Google Chrome.app")]]
    assert tool.describe(AppArgs(name="chrome")) == "open chrome"
    with pytest.raises(ToolError, match="would not open"):
        await OpenApp(FakeRunner((1, "error")), apps).run(AppArgs(name="Safari"))


async def test_quit_app_passes_the_name_as_data(apps: Apps) -> None:
    run = FakeRunner((0, "quit"), (0, "not running"))
    tool = QuitApp(run, apps)
    assert await tool.run(AppArgs(name="safari")) == "Safari has quit."
    assert await tool.run(AppArgs(name="Notes")) == "Notes was not running."
    argv = run.commands[0]
    assert argv[0] == OSASCRIPT and argv[-1] == "Safari"
    assert all("Safari" not in part for part in argv[:-1])  # never spliced into the script
    assert tool.describe(AppArgs(name="Safari")) == "quit Safari"


async def test_volume() -> None:
    run = FakeRunner()
    tool = SetVolume(run)
    assert await tool.run(VolumeArgs(percent=40)) == "Volume is 40%."
    assert run.commands[0][-1] == "40"
    assert tool.describe(VolumeArgs(percent=40)) == "set the volume to 40%"
    for bad in (-1, 101):
        with pytest.raises(ValueError, match="validation error"):
            VolumeArgs(percent=bad)


async def test_media_controls_the_active_player(apps: Apps) -> None:
    run = FakeRunner((0, "false"), (0, "true"), (0, ""))  # Spotify no, Music yes
    tool = Media(run, apps)
    assert await tool.run(MediaArgs(action="next")) == "Music: next."
    assert run.commands[-1] == [OSASCRIPT, "-e", 'tell application "Music" to next track']
    run = FakeRunner()
    await Media(run, apps).run(MediaArgs(action="toggle", player="Spotify"))
    assert run.commands == [[OSASCRIPT, "-e", 'tell application "Spotify" to playpause']]
    assert tool.describe(MediaArgs(action="pause")) == "pause the music"
    assert tool.describe(MediaArgs(action="next")) == "skip to the next track"
    assert tool.describe(MediaArgs(action="toggle")) == "play or pause the music"
    assert tool.starts_playback(MediaArgs(action="next"))
    assert not tool.starts_playback(MediaArgs(action="pause"))


async def test_media_falls_back_to_an_installed_player(apps: Apps, tmp_path: Path) -> None:
    run = FakeRunner((0, "false"), (0, "false"), (0, ""))
    assert await Media(run, apps).run(MediaArgs(action="play")) == "Spotify: play."
    run = FakeRunner((0, "false"), (0, "false"), (0, ""))
    no_spotify = Apps([tmp_path / "nothing"])
    assert await Media(run, no_spotify).run(MediaArgs(action="play")) == "Music: play."


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:pass@example.com",
        "https:///nohost",
        "https://example.com/ evil",
    ],
)
def test_only_plain_https_websites(url: str) -> None:
    with pytest.raises(ToolError):
        safe_url(url)


async def test_open_website() -> None:
    run = FakeRunner()
    tool = OpenWebsite(run)
    args = UrlArgs(url="https://www.kuleuven.be/en")
    assert await tool.run(args) == "Opened www.kuleuven.be."
    assert run.commands == [[OPEN, "https://www.kuleuven.be/en"]]
    assert tool.describe(args) == "open www.kuleuven.be/en"
    assert tool.sends_data_out(args)


def test_a_web_address_is_said_with_what_it_carries() -> None:
    """A confirmation must not hide the data in an address behind its site's name."""
    assert short_url("https://www.youtube.com/") == "www.youtube.com"
    assert (
        short_url("https://evil.example/c?notes=meet+Thomas") == "evil.example/c?notes=meet+Thomas"
    )
    long = short_url("https://evil.example/?d=" + "x" * 200)
    assert long.startswith("evil.example/?d=xxx")
    assert len(long) == 60
    assert long.endswith("…")


def test_tools_per_platform(apps: Apps, monkeypatch: pytest.MonkeyPatch) -> None:
    mac = [tool.name for tool in computer_tools(FakeRunner(), apps=apps, platform="darwin")]
    assert mac[:4] == ["open_app", "quit_app", "set_volume", "media"]
    monkeypatch.setattr(os, "access", lambda path, mode: True)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    assert [t.name for t in computer_tools(FakeRunner(), platform="linux")] == ["open_website"]
    monkeypatch.setattr(os, "access", lambda path, mode: False)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    assert computer_tools(FakeRunner(), platform="linux") == []
    windows = [tool.name for tool in computer_tools(FakeRunner(), platform="win32")]
    assert windows == ["open_app", "quit_app", "set_volume", "media", "open_website"]
    assert computer_tools(FakeRunner(), platform="freebsd") == []


SHOW_ENV = "import os; print(chr(10).join(k + '=' + v for k, v in os.environ.items()))"
ECHO = "import sys; print(sys.argv[1])"


async def test_commands_run_without_a_shell_or_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")  # gitleaks:allow
    code, output = await run_command([sys.executable, "-c", SHOW_ENV])
    assert code == 0
    assert f"PATH={environment()['PATH']}" in output
    assert "sk-ant-secret" not in output
    assert await run_command([sys.executable, "-c", ECHO, "$(whoami)"]) == (0, "$(whoami)")


async def test_commands_time_out() -> None:
    with pytest.raises(TimeoutError):
        await run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=0.1)
