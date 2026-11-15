from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Literal

import pytest

from test_computer import FakeRunner
from yarvis.tools import browser as browser_module
from yarvis.tools.base import Tier, ToolError
from yarvis.tools.browser import (
    PS,
    BrowserCloseTab,
    BrowserGo,
    BrowserMedia,
    BrowserOpen,
    BrowserOpenLink,
    BrowserPress,
    BrowserReadPage,
    Browsers,
    BrowserSearch,
    BrowserSwitchTab,
    BrowserTabs,
    CloseArgs,
    ElementArgs,
    GoArgs,
    MediaArgs,
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


def tab(window: int, index: int, url: str, title: str, *, active: bool = False) -> str:
    return "\x1f".join([str(window), str(index), str(active).lower(), url, title])


TABS = (
    "\x1e".join(
        [
            tab(1, 1, "https://mail.google.com/", "Inbox - Gmail"),
            tab(1, 2, "https://github.com/x", "Yarvis on GitHub", active=True),
            tab(1, 3, "https://search.brave.com/", "Brave Search"),
        ]
    )
    + "\x1e"
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
        "Brave Browser: 3 tabs in 1 window(s).\n1. Inbox - Gmail (mail.google.com)\n"
        "2. Yarvis on GitHub (github.com) [current]\n3. Brave Search (search.brave.com)"
    )
    assert tool.untrusted_output  # titles are written by websites
    script = " ".join(run.commands[-1])
    assert 'if not (mode of window w is "incognito") then' in script  # private stays private
    assert "repeat with w from 1 to (count of windows)" in script  # every window
    run = FakeRunner((0, ""))
    assert await BrowserTabs(run, Browsers(run, brave, "Brave Browser")).run(NoArgs()) == (
        "Brave Browser has no open tabs (private windows are not listed)."
    )


async def test_tabs_in_every_window_are_counted(brave: Apps) -> None:
    """The bug: six tabs in two windows were reported as one."""
    long_title = "A very long title that goes on and on " * 3
    records = [
        tab(1, 1, "https://www.youtube.com/watch?v=" + "x" * 60, long_title, active=True),
        *(tab(1, i, f"https://example.com/{'p' * 80}/{i}", long_title) for i in range(2, 5)),
        tab(2, 1, "https://mail.google.com/", "Inbox"),
        tab(2, 2, "https://github.com/", "GitHub", active=True),
        "1\x1fbroken record",  # a title with separators in it: skipped, not fatal
    ]
    output = "\x1e".join(records)
    assert len(output) > 500  # longer than command output used to be cut to
    run = FakeRunner((0, output))
    listing = await BrowserTabs(run, Browsers(run, brave, "Brave Browser")).run(NoArgs())
    assert listing.startswith("Brave Browser: 6 tabs in 2 window(s).\nFront window:\n1. ")
    assert "\nAnother window:\n5. Inbox (mail.google.com)\n6. GitHub (github.com)" in listing
    assert listing.count("[current]") == 1  # only the front window's tab is current

    many = "\x1e".join(tab(1, i, "https://a.be/", f"Tab {i}") for i in range(1, 71))
    run = FakeRunner((0, many))
    listing = await BrowserTabs(run, Browsers(run, brave, "Brave Browser")).run(NoArgs())
    assert "60. Tab 60" in listing and "61." not in listing
    assert listing.endswith("…and 10 more.")


async def test_switching_tabs(brave: Apps) -> None:
    two_windows = TABS + tab(2, 1, "https://www.youtube.com/", "YouTube")
    run = FakeRunner((0, two_windows), (0, ""))
    tool = BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser"))
    assert await tool.run(SwitchArgs(match="youtube")) == "Switched to tab 4: YouTube"
    switch = run.commands[-1]
    assert switch[-2:] == ["2", "1"]  # window 2, tab 1: numbers travel as data
    assert "set active tab index of window w to t" in switch
    assert "set index of window w to 1" in switch  # its window comes to the front
    run = FakeRunner((0, TABS), (0, ""))
    tool = BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser"))
    assert await tool.run(SwitchArgs(number=3)) == "Switched to tab 3: Brave Search"
    assert run.commands[-1][-2:] == ["1", "3"]
    run = FakeRunner((0, TABS))
    tool = BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser"))
    with pytest.raises(ToolError, match="No tab matches"):
        await tool.run(SwitchArgs(match="netflix"))
    for misheard, number in [("git hub", 2), ("gee mail", 1), ("brave serch", 3)]:
        run = FakeRunner((0, TABS), (0, ""))
        tool = BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser"))
        assert (await tool.run(SwitchArgs(match=misheard))).startswith(f"Switched to tab {number}")
    run = FakeRunner((0, TABS))
    with pytest.raises(ToolError, match="only 3 tabs"):
        await BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser")).run(SwitchArgs(number=9))
    run = FakeRunner((0, TABS), (1, "error"))
    with pytest.raises(ToolError, match="did not switch"):
        await BrowserSwitchTab(run, Browsers(run, brave, "Brave Browser")).run(SwitchArgs(number=1))
    run = FakeRunner((1, "Brave got confused"))
    with pytest.raises(ToolError, match="did not answer"):
        await BrowserTabs(run, Browsers(run, brave, "Brave Browser")).run(NoArgs())
    assert tool.describe(SwitchArgs(number=2)) == "switch to tab 2"
    assert tool.describe(SwitchArgs(match="Gmail")) == "switch to the Gmail tab"
    for bad in ({}, {"number": 1, "match": "x"}):
        with pytest.raises(ValueError, match="either number or match"):
            SwitchArgs.model_validate(bad)


async def test_safari_has_its_own_script(tmp_path: Path) -> None:
    safari = installed(tmp_path, "Safari")
    run = FakeRunner((0, TABS), (0, ""))
    await BrowserSwitchTab(run, Browsers(run, safari, "Safari")).run(SwitchArgs(number=2))
    assert "set current tab of window w to tab t of window w" in run.commands[-1]
    listing = " ".join(run.commands[0])
    assert "name of every tab of window w" in listing
    assert "incognito" not in listing  # Safari has no such property


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

    # "Open a new tab" needs no website, and "go to YouTube" can then fill that tab.
    assert await tool.run(OpenArgs()) == "Opened a new tab in Brave Browser."
    assert run.commands[-1][-1] == ""
    assert "tell front window to make new tab" in run.commands[-1]
    assert tool.describe(OpenArgs()) == "open a new tab"
    youtube = OpenArgs(url="https://www.youtube.com/", in_current_tab=True)
    assert await tool.run(youtube) == "Opened www.youtube.com in the current tab."
    assert "set URL of active tab of front window to theURL" in run.commands[-1]
    with pytest.raises(ToolError, match="did not open"):
        await BrowserOpen(FakeRunner((1, "")), browsers).run(youtube)

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
    assert run.commands[-1][-2:] == ["0", "0"]
    assert "close active tab of front window" in run.commands[-1]
    run.answers = [(0, TABS + tab(2, 1, "https://www.youtube.com/", "YouTube"))]
    assert await close.run(CloseArgs(number=4)) == "Closed tab 4."
    assert run.commands[-1][-2:] == ["2", "1"]  # the YouTube tab in the other window
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
        "browser_read_page": Tier.READ, "browser_open_link": Tier.REVERSIBLE,
        "browser_press": Tier.EXTERNAL, "browser_media": Tier.REVERSIBLE,
    }  # fmt: skip
    untrusted = {t.name for t in tools if t.untrusted_output}
    assert untrusted == {
        "browser_tabs", "browser_switch_tab", "browser_read_page", "browser_open_link",
        "browser_press",
    }  # fmt: skip


# --------------------------------------------------------------------------- the page

PAGE = json.dumps(
    {
        "title": "Feel What You Want - YouTube",
        "url": "https://www.youtube.com/results?search_query=feel+what+you+want",
        "text": "Feel What You Want Kristine W 2.1M views",
        "items": [[1, "link", "Feel What You Want (Official Video)"], [2, "button", "Subscribe"]],
    }
)


async def test_reading_the_page(brave: Apps, tmp_path: Path) -> None:
    run = FakeRunner((0, PAGE))
    tool = BrowserReadPage(run, Browsers(run, brave, "Brave Browser"))
    assert await tool.run(NoArgs()) == (
        "Page: Feel What You Want - YouTube (www.youtube.com)\n"
        "Text: Feel What You Want Kristine W 2.1M views\n"
        "Links and buttons:\n"
        "1. link: Feel What You Want (Official Video)\n"
        "2. button: Subscribe"
    )
    command = run.commands[-1]
    assert "set theResult to execute active tab of front window javascript theCode" in command
    assert 'if mode of front window is "incognito" then return "<private>"' in command
    assert command[-1].startswith("(() => {")  # the fixed program travels as data
    assert tool.describe(NoArgs()) == "read the current page"

    empty = json.dumps({"title": "Blank", "url": "about:blank", "text": "", "items": []})
    run = FakeRunner((0, empty))
    assert await BrowserReadPage(run, Browsers(run, brave, "Brave Browser")).run(NoArgs()) == (
        "Page: Blank (this page)\nText: "
    )

    safari = installed(tmp_path / "s", "Safari")
    run = FakeRunner((0, PAGE))
    await BrowserReadPage(run, Browsers(run, safari, "Safari")).run(NoArgs())
    assert (
        "set theResult to do JavaScript theCode in current tab of front window"
        in (run.commands[-1])
    )


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ((1, "Executing JavaScript through AppleScript is turned off."), "View > Developer > "
         "Allow JavaScript from Apple Events"),
        ((1, "some other failure"), "did not answer"),
        ((0, "<no window>"), "no open window"),
        ((0, "<private>"), "private window"),
        ((0, "not json"), "could not be read"),
        ((0, "[1, 2]"), "could not be read"),
    ],
)  # fmt: skip
async def test_reading_the_page_can_fail(
    brave: Apps, answer: tuple[int, str], message: str
) -> None:
    run = FakeRunner(answer)
    with pytest.raises(ToolError, match=message):
        await BrowserReadPage(run, Browsers(run, brave, "Brave Browser")).run(NoArgs())


async def test_safari_explains_its_own_menu(tmp_path: Path) -> None:
    run = FakeRunner((1, "You must enable 'Allow JavaScript from Apple Events'"))
    safari = Browsers(run, installed(tmp_path, "Safari"), "Safari")
    with pytest.raises(ToolError, match="turn on Develop > Allow JavaScript"):
        await BrowserReadPage(run, safari).run(NoArgs())


def element(text: str, href: str = "") -> tuple[int, str]:
    return 0, json.dumps({"link": bool(href), "text": text, "href": href})


VIDEO = ElementArgs(number=1, label="Feel What You Want")


async def test_opening_a_link_from_the_page(brave: Apps) -> None:
    video = element("Feel What You Want (Official Video)", "https://www.youtube.com/watch?v=abc")
    run = FakeRunner(video, (0, ""))
    tool = BrowserOpenLink(run, Browsers(run, brave, "Brave Browser"))
    assert await tool.run(VIDEO) == "Opened “Feel What You Want (Official Video)”."
    assert 'data-yarvis-n="1"' in run.commands[0][-1]  # only the number is in the code
    navigate = run.commands[-1]
    assert navigate[-1] == "https://www.youtube.com/watch?v=abc"
    assert "set URL of active tab of front window to theURL" in navigate
    assert tool.describe(VIDEO) == "open the link “Feel What You Want”"


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ((0, ""), "page has changed"),
        (element("Subscribe"), "now 'Subscribe'"),
        (element("Feel What You Want"), "a button, not a link"),
        (element("Feel What You Want", "http://example.com/"), "Only https"),
    ],
)
async def test_links_are_checked_before_opening(
    brave: Apps, answer: tuple[int, str], message: str
) -> None:
    run = FakeRunner(answer)
    with pytest.raises(ToolError, match=message):
        await BrowserOpenLink(run, Browsers(run, brave, "Brave Browser")).run(VIDEO)
    assert len(run.commands) == 1  # nothing was opened


async def test_pressing_a_button(brave: Apps) -> None:
    run = FakeRunner(element("Subscribe"), (0, "ok"))
    tool = BrowserPress(run, Browsers(run, brave, "Brave Browser"))
    subscribe = ElementArgs(number=2, label="subscribe")
    assert await tool.run(subscribe) == "Pressed “Subscribe”."
    assert ".click()" in run.commands[-1][-1]
    assert tool.describe(subscribe) == "press “subscribe” on the page"
    assert tool.tier is Tier.EXTERNAL  # a button can send, buy or delete: asked first
    run = FakeRunner(element("Subscribe"), (0, ""))
    with pytest.raises(ToolError, match="page has changed"):
        await BrowserPress(run, Browsers(run, brave, "Brave Browser")).run(subscribe)


@pytest.mark.parametrize(
    ("action", "answer", "said"),
    [
        ("pause", "paused", "Paused."),
        ("play", "playing", "Playing."),
        ("toggle", "paused", "Paused."),
    ],
)
async def test_playing_and_pausing_videos(
    brave: Apps, action: Literal["play", "pause", "toggle"], answer: str, said: str
) -> None:
    run = FakeRunner((0, answer))
    tool = BrowserMedia(run, Browsers(run, brave, "Brave Browser"))
    assert await tool.run(MediaArgs(action=action)) == said
    assert f"const want = '{action}';" in run.commands[-1][-1]
    run = FakeRunner((0, "none"))
    with pytest.raises(ToolError, match="no video"):
        await BrowserMedia(run, Browsers(run, brave, "Brave Browser")).run(MediaArgs(action=action))
    assert tool.describe(MediaArgs(action="toggle")) == "play or pause the video"


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
    assert "first run 'yarvis spotify login'" in said  # so Yarvis can explain it
    bare = computer_tools(FakeRunner(), apps=installed(tmp_path / "b"), platform="darwin")
    assert "open websites" in abilities(bare)


@pytest.mark.parametrize(
    ("wanted", "actual", "same"),
    [
        ("Feel What You Want", "Kristine W - Feel What You Want (Official Video)", True),
        ("feel what you want official video", "Feel What You Want (Official Video)", True),
        ("Kristine W Feel What You Want video", "Kristine W - Feel What You Want (HD)", True),
        ("Subscribe", "Unsubscribe from channel", False),
        ("Subscribe", "Unsubscribe", False),
        ("Delete account", "Play video", False),
        ("", "Play", False),
    ],
)
def test_labels_may_be_shortened_but_not_swapped(wanted: str, actual: str, same: bool) -> None:
    assert browser_module._same_label(wanted, actual) is same  # pyright: ignore[reportPrivateUsage]


def test_what_starts_playback() -> None:
    run = FakeRunner()
    browsers = Browsers(run, Apps([]), "auto")
    assert BrowserOpenLink(run, browsers).starts_playback(VIDEO)  # a video may start
    media = BrowserMedia(run, browsers)
    assert media.starts_playback(MediaArgs(action="play"))
    assert not media.starts_playback(MediaArgs(action="pause"))
    assert not BrowserReadPage(run, browsers).starts_playback(NoArgs())
