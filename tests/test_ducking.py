"""Turning Spotify down while Yarvis listens and talks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from test_computer import FakeRunner
from yarvis.config import Config
from yarvis.tools.browser import Browsers
from yarvis.tools.computer import Apps
from yarvis.tools.desktop import music_ducker
from yarvis.tools.ducking import BrowserDucker, Duckers, SpotifyDucker

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


async def test_music_goes_down_and_back_up() -> None:
    run = FakeRunner((0, "60,15"), (0, ""))
    ducker = SpotifyDucker(run)
    await ducker.duck()
    await ducker.duck()  # already down: nothing more
    assert len(run.commands) == 1
    assert run.commands[0][-1] == "25"  # the share travels as data
    assert 'if player state is not playing then return ""' in run.commands[0]
    await ducker.restore()
    assert run.commands[-1][-3:] == ["60", "15", "2"]  # only if still about 15
    await ducker.restore()  # nothing saved any more
    assert len(run.commands) == 2


async def test_nothing_playing_means_nothing_to_restore() -> None:
    for answer in [(0, ""), (1, "Spotify got confused"), (0, "garbage")]:
        run = FakeRunner(answer)
        ducker = SpotifyDucker(run)
        await ducker.duck()
        await ducker.restore()
        assert len(run.commands) == 1


async def test_the_music_never_breaks_a_conversation() -> None:
    async def broken(argv: Sequence[str]) -> tuple[int, str]:
        raise TimeoutError

    ducker = SpotifyDucker(broken)
    await ducker.duck()  # no exception
    calls: list[int] = []

    async def down_then_broken(argv: Sequence[str]) -> tuple[int, str]:
        calls.append(1)
        if len(calls) == 1:
            return 0, "60,15"
        raise TimeoutError

    ducker = SpotifyDucker(down_then_broken)
    await ducker.duck()
    await ducker.restore()  # no exception either
    assert len(calls) == 2


def test_only_on_a_mac_with_something_to_turn_down(tmp_path: Path) -> None:
    (tmp_path / "Spotify.app").mkdir()
    (tmp_path / "Brave Browser.app").mkdir()
    apps = Apps([tmp_path])
    both = music_ducker(Config(), apps=apps, platform="darwin")
    assert isinstance(both, Duckers)
    assert [type(d) for d in both._duckers] == [SpotifyDucker, BrowserDucker]  # pyright: ignore[reportPrivateUsage]
    assert music_ducker(Config(), apps=Apps([tmp_path / "none"]), platform="darwin") is None
    assert music_ducker(Config(), apps=apps, platform="linux") is None
    off = Config.model_validate({"tools": {"computer": False}})
    assert music_ducker(off, apps=apps, platform="darwin") is None


def brave(tmp_path: Path, run: FakeRunner) -> Browsers:
    (tmp_path / "Brave Browser.app").mkdir(exist_ok=True)
    return Browsers(run, Apps([tmp_path]), "Brave Browser")


async def test_a_video_in_the_browser_goes_down_and_back_up(tmp_path: Path) -> None:
    run = FakeRunner((0, "1"), (0, "ok"))
    ducker = BrowserDucker(run, brave(tmp_path, run))
    await ducker.duck()
    await ducker.duck()  # already down
    assert len(run.commands) == 1
    assert "m.volume = m.volume * 0.25" in run.commands[0][-1]  # fixed JavaScript, as data
    script = run.commands[0]
    assert "set theResult to execute active tab of front window javascript theCode" in script
    await ducker.restore()
    assert "data-yarvis-volume" in run.commands[-1][-1]
    await ducker.restore()
    assert len(run.commands) == 2


async def test_no_video_or_no_permission_means_nothing_to_restore(tmp_path: Path) -> None:
    for answer in [(0, "0"), (1, "JavaScript from Apple Events is turned off"), (0, "<private>")]:
        run = FakeRunner(answer)
        ducker = BrowserDucker(run, brave(tmp_path, run))
        await ducker.duck()
        await ducker.restore()
        assert len(run.commands) == 1
    nothing = BrowserDucker(FakeRunner(), Browsers(FakeRunner(), Apps([]), "Safari"))
    await nothing.duck()  # Safari is not installed: no exception


async def test_a_broken_browser_never_breaks_a_conversation(tmp_path: Path) -> None:
    calls: list[int] = []

    async def down_then_broken(argv: Sequence[str]) -> tuple[int, str]:
        calls.append(1)
        if len(calls) == 1:
            return 0, "2"
        raise TimeoutError

    ducker = BrowserDucker(down_then_broken, brave(tmp_path, FakeRunner()))
    await ducker.duck()
    await ducker.restore()  # no exception
    assert len(calls) == 2


async def test_everything_goes_down_together() -> None:
    log: list[str] = []

    class Recording:
        def __init__(self, name: str) -> None:
            self.name = name

        async def duck(self) -> None:
            log.append(f"{self.name} down")

        async def restore(self) -> None:
            log.append(f"{self.name} up")

    both = Duckers(Recording("spotify"), Recording("video"))
    await both.duck()
    await both.restore()
    assert log == ["spotify down", "video down", "spotify up", "video up"]
