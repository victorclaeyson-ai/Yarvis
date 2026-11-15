"""Control the web browser: tabs, websites, searching, back and forward.

Brave, Chrome, Edge, Vivaldi and Chromium share one AppleScript dictionary; Safari has its
own. The browser's name comes from a fixed list and is the only thing ever written into a
script; URLs, searches and tab numbers travel as data (``argv``).

Tab titles are written by websites, so listing tabs returns untrusted content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, Self, get_args
from urllib.parse import quote_plus, urlsplit

from pydantic import Field, model_validator

from yarvis.tools.base import Arguments, Tier, Tool, ToolError
from yarvis.tools.computer import osascript, safe_url

if TYPE_CHECKING:
    from yarvis.tools.computer import Apps, Runner

BrowserName = Literal[
    "Brave Browser", "Google Chrome", "Microsoft Edge", "Vivaldi", "Chromium", "Safari"
]
BROWSERS: Final[tuple[BrowserName, ...]] = get_args(BrowserName)
PS: Final = "/bin/ps"
_FIELD: Final = "\x1f"
_ITEM: Final = "\x1e"
_MAX_TITLE: Final = 80


class Browsers:
    """Picks the browser to control: the configured one, else the one in use."""

    def __init__(self, run: Runner, apps: Apps, preferred: BrowserName | Literal["auto"]) -> None:
        self._run = run
        self._apps = apps
        self._preferred: BrowserName | Literal["auto"] = preferred

    def available(self) -> bool:
        """Whether any supported browser is installed."""
        installed = self._apps.names()
        return any(browser in installed for browser in BROWSERS)

    async def pick(self) -> BrowserName:
        """The browser to act on."""
        installed = self._apps.names()
        preferred = self._preferred
        if preferred != "auto":
            if preferred not in installed:
                msg = f"{preferred} is not installed."
                raise ToolError(msg)
            return preferred
        _, processes = await self._run([PS, "-axo", "comm="])
        for browser in BROWSERS:
            if f"/{browser}.app/" in processes:
                return browser
        for browser in BROWSERS:
            if browser in installed:
                return browser
        msg = "No supported browser is installed."
        raise ToolError(msg)


def _tell(browser: BrowserName, *lines: str) -> tuple[str, ...]:
    return (f'tell application "{browser}"', *lines, "end tell")


def _tabs_script(browser: BrowserName) -> tuple[str, ...]:
    title, active = (
        ("name", "index of current tab") if browser == "Safari" else ("title", "active tab index")
    )
    return (
        f'if application "{browser}" is not running then return "none"',
        *_tell(
            browser,
            'if (count of windows) is 0 then return "none"',
            f"set theTitles to {title} of every tab of front window",
            "set theURLs to URL of every tab of front window",
            f"set activeIndex to {active} of front window",
        ),
        "set AppleScript's text item delimiters to (character id 30)",
        "set f to character id 31",
        "return (activeIndex as text) & f & (theTitles as text) & f & (theURLs as text)",
    )


def _switch_script(browser: BrowserName) -> tuple[str, ...]:
    select = (
        "set current tab of front window to tab n of front window"
        if browser == "Safari"
        else "set active tab index of front window to n"
    )
    return (
        "on run argv",
        "set n to (item 1 of argv) as integer",
        *_tell(browser, select, "activate"),
        "end run",
    )


def _open_script(browser: BrowserName) -> tuple[str, ...]:
    if browser == "Safari":
        body = (
            "if (count of windows) is 0 then",
            "make new document with properties {URL:theURL}",
            "else",
            "tell front window to set current tab to (make new tab with properties {URL:theURL})",
            "end if",
            "activate",
        )
    else:
        body = (
            "if (count of windows) is 0 then",
            "make new window",
            "set URL of active tab of front window to theURL",
            "else",
            "tell front window to make new tab with properties {URL:theURL}",
            "end if",
            "activate",
        )
    return ("on run argv", "set theURL to item 1 of argv", *_tell(browser, *body), "end run")


def _close_script(browser: BrowserName) -> tuple[str, ...]:
    current = "index of current tab" if browser == "Safari" else "active tab index"
    return (
        "on run argv",
        "set n to (item 1 of argv) as integer",
        *_tell(
            browser,
            f"if n is 0 then set n to {current} of front window",
            "close tab n of front window",
        ),
        "end run",
    )


_GO_COMMANDS: Final = {"back": "go back", "forward": "go forward", "reload": "reload"}


def _go_script(
    browser: BrowserName, action: Literal["back", "forward", "reload"]
) -> tuple[str, ...]:
    if browser == "Safari":
        if action != "reload":
            msg = "Safari cannot go back or forward by script; use Brave or Chrome for that."
            raise ToolError(msg)
        line = "set URL of current tab of front window to (URL of current tab of front window)"
        return _tell(browser, line)
    return _tell(browser, f"tell active tab of front window to {_GO_COMMANDS[action]}")


# ----------------------------------------------------------------------------- tools


class NoArgs(Arguments):
    """No arguments."""


class SwitchArgs(Arguments):
    """Arguments of ``browser_switch_tab``: a number or some text."""

    number: int | None = Field(default=None, ge=1, le=500, description="Tab number (1 = leftmost).")
    match: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Text in the tab's title or address.",
    )

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.number is None) == (self.match is None):
            msg = "give either number or match"
            raise ValueError(msg)
        return self


class OpenArgs(Arguments):
    """Arguments of ``browser_open``."""

    url: str = Field(min_length=8, max_length=2000, description="A full https:// URL.")


class SearchArgs(Arguments):
    """Arguments of ``browser_search``."""

    query: str = Field(min_length=1, max_length=300)


class CloseArgs(Arguments):
    """Arguments of ``browser_close_tab``."""

    number: int | None = Field(default=None, ge=1, le=500, description="Omit for the current tab.")


class GoArgs(Arguments):
    """Arguments of ``browser_go``."""

    action: Literal["back", "forward", "reload"]


class _BrowserTool[A: Arguments](Tool[A]):
    def __init__(self, run: Runner, browsers: Browsers) -> None:
        self._run = run
        self._browsers = browsers

    async def _script(self, lines: tuple[str, ...], *values: str, failure: str) -> str:
        code, output = await self._run(osascript(lines, *values))
        if code != 0:
            raise ToolError(failure)
        return output

    async def _tabs(self, browser: BrowserName) -> tuple[int, list[tuple[str, str]]]:
        output = await self._script(_tabs_script(browser), failure=f"{browser} did not answer.")
        if output == "none":
            return 0, []
        parts = output.split(_FIELD)
        if len(parts) != 3:  # noqa: PLR2004
            msg = f"{browser} gave an unexpected answer."
            raise ToolError(msg)
        titles, urls = parts[1].split(_ITEM), parts[2].split(_ITEM)
        return int(parts[0] or 0), list(zip(titles, urls, strict=False))


class BrowserTabs(_BrowserTool[NoArgs]):
    """List the open tabs."""

    name = "browser_tabs"
    description = "List the tabs in the browser's front window (number, title, site)."
    tier = Tier.READ
    Args = NoArgs
    untrusted_output = True

    async def run(self, args: NoArgs) -> str:
        """List them."""
        browser = await self._browsers.pick()
        active, tabs = await self._tabs(browser)
        if not tabs:
            return f"{browser} has no open windows."
        lines = [
            f"{i}. {title[:_MAX_TITLE]} ({urlsplit(url).hostname or url[:40]})"
            + (" [current]" if i == active else "")
            for i, (title, url) in enumerate(tabs, start=1)
        ]
        return f"{browser} tabs:\n" + "\n".join(lines)

    def describe(self, args: NoArgs) -> str:
        """'look at your tabs'."""
        return "look at your browser tabs"


class BrowserSwitchTab(_BrowserTool[SwitchArgs]):
    """Go to another tab."""

    name = "browser_switch_tab"
    description = "Switch to a browser tab by number, or by text in its title or address."
    tier = Tier.REVERSIBLE
    Args = SwitchArgs
    untrusted_output = True

    async def run(self, args: SwitchArgs) -> str:
        """Switch."""
        browser = await self._browsers.pick()
        _, tabs = await self._tabs(browser)
        number = args.number
        if args.match is not None:
            wanted = args.match.casefold()
            hits = [
                i
                for i, (t, u) in enumerate(tabs, 1)
                if wanted in t.casefold() or wanted in u.casefold()
            ]
            if not hits:
                msg = f"No tab matches {args.match!r}."
                raise ToolError(msg)
            number = hits[0]
        if number is None or number > len(tabs):
            msg = f"There are only {len(tabs)} tabs."
            raise ToolError(msg)
        await self._script(_switch_script(browser), str(number), failure="The tab did not switch.")
        return f"Switched to tab {number}: {tabs[number - 1][0][:_MAX_TITLE]}"

    def describe(self, args: SwitchArgs) -> str:
        """'switch to tab 3' / 'switch to the Gmail tab'."""
        return f"switch to tab {args.number}" if args.number else f"switch to the {args.match} tab"


class BrowserOpen(_BrowserTool[OpenArgs]):
    """Open a website in a new tab."""

    name = "browser_open"
    description = "Open an https:// website in a new tab of the user's browser."
    tier = Tier.REVERSIBLE
    Args = OpenArgs

    async def run(self, args: OpenArgs) -> str:
        """Open it."""
        url = safe_url(args.url)
        browser = await self._browsers.pick()
        await self._script(_open_script(browser), url, failure=f"{browser} would not open it.")
        return f"Opened {urlsplit(url).hostname} in {browser}."

    def describe(self, args: OpenArgs) -> str:
        """'open kuleuven.be'."""
        return f"open {urlsplit(args.url).hostname or args.url}"


class BrowserSearch(_BrowserTool[SearchArgs]):
    """Search the web in the browser (for the user to read)."""

    name = "browser_search"
    description = "Search the web in a new browser tab, for the user to look at."
    tier = Tier.REVERSIBLE
    Args = SearchArgs

    async def run(self, args: SearchArgs) -> str:
        """Search."""
        browser = await self._browsers.pick()
        engine = (
            "https://search.brave.com/search?q="
            if browser == "Brave Browser"
            else "https://www.google.com/search?q="
        )
        url = engine + quote_plus(args.query)
        await self._script(_open_script(browser), url, failure=f"{browser} would not search.")
        return f"Searching for '{args.query}' in {browser}."

    def describe(self, args: SearchArgs) -> str:
        """'search for "weather Leuven"'."""
        return f"search for “{args.query}”"


class BrowserCloseTab(_BrowserTool[CloseArgs]):
    """Close a tab."""

    name = "browser_close_tab"
    description = "Close a browser tab by number (or the current tab)."
    tier = Tier.REVERSIBLE
    Args = CloseArgs

    async def run(self, args: CloseArgs) -> str:
        """Close it."""
        browser = await self._browsers.pick()
        await self._script(
            _close_script(browser), str(args.number or 0), failure="The tab did not close."
        )
        return f"Closed {'tab ' + str(args.number) if args.number else 'the current tab'}."

    def describe(self, args: CloseArgs) -> str:
        """'close tab 2' / 'close this tab'."""
        return f"close tab {args.number}" if args.number else "close this tab"


class BrowserGo(_BrowserTool[GoArgs]):
    """Back, forward, reload."""

    name = "browser_go"
    description = "Go back, go forward, or reload in the current browser tab."
    tier = Tier.REVERSIBLE
    Args = GoArgs

    async def run(self, args: GoArgs) -> str:
        """Do it."""
        browser = await self._browsers.pick()
        await self._script(_go_script(browser, args.action), failure=f"{browser} did not respond.")
        return f"Done: {args.action}."

    def describe(self, args: GoArgs) -> str:
        """'go back'."""
        return "reload the page" if args.action == "reload" else f"go {args.action}"


def browser_tools(run: Runner, browsers: Browsers) -> list[Tool[Any]]:
    """All browser tools."""
    return [
        BrowserTabs(run, browsers),
        BrowserSwitchTab(run, browsers),
        BrowserOpen(run, browsers),
        BrowserSearch(run, browsers),
        BrowserCloseTab(run, browsers),
        BrowserGo(run, browsers),
    ]
