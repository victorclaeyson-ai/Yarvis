from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from test_computer import FakeRunner
from yarvis.tools.base import Tier, ToolError
from yarvis.tools.browser import (
    PS,
    BrowserCloseTab,
    BrowserGo,
    BrowserOpen,
    Browsers,
    BrowserSearch,
    BrowserSwitchTab,
    BrowserTabs,
    CloseArgs,
    GoArgs,
    NoArgs,
    OpenArgs,
    SearchArgs,
    SwitchArgs,
    browser_tools,
)
from yarvis.tools.computer import OPEN, Apps
from yarvis.tools.desktop import abilities, computer_tools
from yarvis.tools.typing import EnterArgs, PressEnter, TypeArgs, TypeText

if TYPE_CHECKING:
    from pathlib import Path


def installed(tmp_path: Path, *names: str) -> Apps:
    folder = tmp_path / "Applications"
    for name in names:
        (folder / f"{name}.app").mkdir(parents=True)
    return Apps([folder])


TABS = "\x1f".join(
    [
        "2",
        "Inbox - Gmail\x1eYarvis on GitHub\x1eBrave Search",
        "https://mail.google.com/\x1ehttps://github.com/x\x1ehttps://search.brave.com/",
    ]
)


# --------------------------------------------------------------------------- which browser


async def test_the_open_browser_is_used(tmp_path: Path) -> None:
    apps = installed(tmp_path, "Brave Browser", "Safari")
    ps = "/usr/sbin/cfprefsd\n/Applications/Safari.app/Contents/MacOS/Safari\n"
    run = FakeRunner((0, ps))
    assert await Browsers(run, apps, "auto").pick() == "Safari"
    assert run.commands == [[PS, "-axo", "comm="]]
    assert await Browsers(FakeRunner((0, "")), apps, "auto").pick() == "Brave Browser"
    assert await Browsers(FakeRunner(), apps, "Safari").pick() == "Safari"
    with pytest.raises(ToolError, match="Google Chrome is not installed"):
        await Browsers(FakeRunner(), apps, "Google Chrome").pick()
    with pytest.raises(ToolError, match="No supported browser"):
        await Browsers(FakeRunner(), installed(tmp_path / "x"), "auto").pick()


# --------------------------------------------------------------------------- tabs


@pytest.fixture
def brave(tmp_path: Path) -> Apps:
    return installed(tmp_path, "Brave Browser")


async def test_listing_tabs(brave: Apps) -> None:
    run = FakeRunner((0, ""), (0, TABS))
    tool = BrowserTabs(run, Browsers(run, brave, "auto"))
    assert await tool.run(NoArgs()) == (
        "Brave Browser tabs:\n1. Inbox - Gmail (mail.google.com)\n"
        "2. Yarvis on GitHub (github.com) [current]\n3. Brave Search (search.brave.com)"
    )
    assert tool.untrusted_output  # titles are written by websites
    run = FakeRunner((0, "none"))
    assert await BrowserTabs(run, Browsers(run, brave, "Brave Browser")).run(NoArgs()) == (
        "Brave Browser has no open windows."
    )
    run = FakeRunner((0, "garbage"))
    with pytest.raises(ToolError, match="unexpected answer"):
        await BrowserTabs(run, Browsers(run, brave, "Brave Browser")).run(NoArgs())


async def test_switching_tabs(brave: Apps) -> None:
    run = FakeRunner((0, TABS), (0, ""))
    tool = BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser"))
    assert await tool.run(SwitchArgs(match="gmail")) == "Switched to tab 1: Inbox - Gmail"
    switch = run.commands[-1]
    assert switch[-1] == "1"  # the number travels as data
    assert "set active tab index of front window to n" in switch
    run = FakeRunner((0, TABS))
    tool = BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser"))
    with pytest.raises(ToolError, match="No tab matches"):
        await tool.run(SwitchArgs(match="youtube"))
    run = FakeRunner((0, TABS))
    with pytest.raises(ToolError, match="only 3 tabs"):
        await BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser")).run(SwitchArgs(number=9))
    assert tool.describe(SwitchArgs(number=2)) == "switch to tab 2"
    assert tool.describe(SwitchArgs(match="Gmail")) == "switch to the Gmail tab"
    for bad in ({}, {"number": 1, "match": "x"}):
        with pytest.raises(ValueError, match="either number or match"):
            SwitchArgs.model_validate(bad)


async def test_safari_has_its_own_script(tmp_path: Path) -> None:
    safari = installed(tmp_path, "Safari")
    run = FakeRunner((0, TABS), (0, ""))
    await BrowserSwitchTab(run, Browsers(run, safari, "Safari")).run(SwitchArgs(number=2))
    assert "set current tab of front window to tab n of front window" in run.commands[-1]
    assert "name of every tab of front window" in " ".join(run.commands[0])


async def test_opening_and_searching(brave: Apps, tmp_path: Path) -> None:
    run = FakeRunner()
    browsers = Browsers(run, brave, "Brave Browser")
    tool = BrowserOpen(run, browsers)
    assert await tool.run(OpenArgs(url="https://www.kuleuven.be/")) == (
        "Opened www.kuleuven.be in Brave Browser."
    )
    assert run.commands[-1][-1] == "https://www.kuleuven.be/"
    with pytest.raises(ToolError, match="Only https"):
        await tool.run(OpenArgs(url="javascript:alert(1)"))
    assert tool.describe(OpenArgs(url="https://www.kuleuven.be/")) == "open www.kuleuven.be"

    search = BrowserSearch(run, browsers)
    assert await search.run(SearchArgs(query="weather Leuven")) == (
        "Searching for 'weather Leuven' in Brave Browser."
    )
    assert run.commands[-1][-1] == "https://search.brave.com/search?q=weather+Leuven"
    assert search.describe(SearchArgs(query="weather")) == "search for “weather”"

    safari = installed(tmp_path / "s", "Safari")
    await BrowserSearch(run, Browsers(run, safari, "Safari")).run(SearchArgs(query="a&b"))
    assert run.commands[-1][-1] == "https://www.google.com/search?q=a%26b"
    assert "make new document with properties {URL:theURL}" in run.commands[-1]
    with pytest.raises(ToolError, match="would not search"):
        await BrowserSearch(FakeRunner((1, "")), browsers).run(SearchArgs(query="x"))


async def test_closing_and_navigating(brave: Apps, tmp_path: Path) -> None:
    run = FakeRunner()
    browsers = Browsers(run, brave, "Brave Browser")
    close = BrowserCloseTab(run, browsers)
    assert await close.run(CloseArgs()) == "Closed the current tab."
    assert run.commands[-1][-1] == "0"
    assert await close.run(CloseArgs(number=3)) == "Closed tab 3."
    assert close.describe(CloseArgs()) == "close this tab"
    assert close.describe(CloseArgs(number=3)) == "close tab 3"

    go = BrowserGo(run, browsers)
    assert await go.run(GoArgs(action="back")) == "Done: back."
    assert "tell active tab of front window to go back" in run.commands[-1]
    assert go.describe(GoArgs(action="reload")) == "reload the page"
    assert go.describe(GoArgs(action="forward")) == "go forward"

    safari = Browsers(run, installed(tmp_path / "s", "Safari"), "Safari")
    with pytest.raises(ToolError, match="Safari cannot go back"):
        await BrowserGo(run, safari).run(GoArgs(action="back"))
    await BrowserGo(run, safari).run(GoArgs(action="reload"))


def test_browser_tools_and_tiers(brave: Apps) -> None:
    tools = browser_tools(FakeRunner(), Browsers(FakeRunner(), brave, "auto"))
    assert {t.name: t.tier for t in tools} == {
        "browser_tabs": Tier.READ, "browser_switch_tab": Tier.REVERSIBLE,
        "browser_open": Tier.REVERSIBLE, "browser_search": Tier.REVERSIBLE,
        "browser_close_tab": Tier.REVERSIBLE, "browser_go": Tier.REVERSIBLE,
    }  # fmt: skip


# --------------------------------------------------------------------------- typing


def front(name: str, pid: int = 1) -> tuple[int, str]:
    return 0, f"{name}\x1f{pid}"


async def test_typing_pastes_into_the_app_in_front(tmp_path: Path) -> None:
    apps = installed(tmp_path, "Notes")
    run = FakeRunner((0, ""), front("Notes"), (0, ""))
    tool = TypeText(run, apps)
    args = TypeArgs(text="Hallo Thomas, tot straks!", app="notes")
    assert await tool.run(args) == "Typed 25 characters into Notes."
    assert run.commands[0] == [OPEN, "-a", str(tmp_path / "Applications" / "Notes.app")]
    paste = run.commands[2]
    assert paste[-1] == "Hallo Thomas, tot straks!"  # the text travels as data
    assert 'tell application "System Events" to keystroke "v" using command down' in paste
    assert tool.describe(args) == "type “Hallo Thomas, tot straks!” into notes"
    long = TypeArgs(text="x" * 100)
    assert tool.describe(long) == f"type “{'x' * 57}…” into the app in front"


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (front("Terminal"), "does not type into Terminal"),
        (front("1Password"), "does not type into 1Password"),
        (front("Python", os.getpid()), "own window is in front"),
        ((1, "osascript is not allowed assistive access. (-1719)"), "Accessibility"),
        ((1, "something else"), "Could not see which app"),
    ],
)
async def test_typing_refuses(tmp_path: Path, answer: tuple[int, str], message: str) -> None:
    run = FakeRunner(answer)
    with pytest.raises(ToolError, match=message):
        await TypeText(run, installed(tmp_path)).run(TypeArgs(text="echo hi"))
    assert len(run.commands) == 1  # nothing was typed


async def test_typing_failures(tmp_path: Path) -> None:
    apps = installed(tmp_path, "Notes")
    with pytest.raises(ToolError, match="would not open"):
        await TypeText(FakeRunner((1, "")), apps).run(TypeArgs(text="x", app="Notes"))
    run = FakeRunner(front("Notes"), (1, "not allowed to send keystrokes (1002)"))
    with pytest.raises(ToolError, match="Accessibility"):
        await TypeText(run, apps).run(TypeArgs(text="x"))
    run = FakeRunner(front("Notes"), (1, "?"))
    with pytest.raises(ToolError, match="did not go through"):
        await TypeText(run, apps).run(TypeArgs(text="x"))


async def test_enter_needs_confirmation_and_the_right_app(tmp_path: Path) -> None:
    apps = installed(tmp_path)
    tool = PressEnter(FakeRunner(front("Messages"), (0, "")), apps)
    assert tool.tier is Tier.EXTERNAL
    assert tool.describe(EnterArgs(app="Messages")) == "press Enter in Messages"
    assert await tool.run(EnterArgs(app="messages")) == "Pressed Enter in Messages."
    wrong = PressEnter(FakeRunner(front("Slack")), apps)
    with pytest.raises(ToolError, match="Slack is in front, not Messages"):
        await wrong.run(EnterArgs(app="Messages"))


# --------------------------------------------------------------------------- what is available


def test_tools_depend_on_what_is_installed(tmp_path: Path) -> None:
    everything = installed(tmp_path, "Brave Browser", "Spotify")
    names = [t.name for t in computer_tools(FakeRunner(), apps=everything, platform="darwin")]
    assert "browser_tabs" in names and "spotify_play" in names and "type_text" in names
    assert "open_website" not in names  # the browser tools replace it
    bare = installed(tmp_path / "bare")
    names = [
        t.name for t in computer_tools(FakeRunner(), apps=bare, platform="darwin", typing=False)
    ]
    assert names == ["open_app", "quit_app", "set_volume", "media", "open_website"]


def test_abilities_are_described(tmp_path: Path) -> None:
    tools = computer_tools(
        FakeRunner(), apps=installed(tmp_path, "Brave Browser", "Spotify"), platform="darwin"
    )
    said = " ".join(abilities(tools))
    for words in ("open and quit apps", "control the web browser", "Spotify", "type text"):
        assert words in said
    bare = computer_tools(FakeRunner(), apps=installed(tmp_path / "b"), platform="darwin")
    assert "open websites" in abilities(bare)
