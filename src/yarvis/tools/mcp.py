"""Tools from MCP servers, behind the same permission gate as Yarvis's own (ADR 0013).

An MCP server is code someone else wrote, and its tool descriptions are text the model
reads: both can be hostile (design doc §5.2, T2). So, following §5.4:

* **Nothing is automatic.** A server runs only if it is in the config, and only the tools
  listed under it are offered, each with the tier the user gave it (0 to 3).
* **Pinned definitions.** ``yarvis mcp approve`` shows each listed tool's description and
  arguments and records a fingerprint (SHA-256) of them. A tool whose definition changes
  later ("rug pull") is left out until the user approves it again.
* **Scrubbed environment.** A server Yarvis starts gets only a few harmless variables
  (PATH, HOME, ...), the ones the config passes on, and its own secrets from the keychain;
  never Yarvis's API keys. It runs in its own folder, and what it prints goes to its log.
* **HTTPS only** for remote servers, with an optional bearer token from the keychain.
* **Outside content.** Results (and error messages) are wrapped as untrusted data and
  taint the turn; every call counts as able to send data out, so after reading outside
  content, each one needs the user's "yes".
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import JsonValue  # noqa: TC002 - pydantic needs it at runtime

from yarvis.llm.base import ToolSpec
from yarvis.paths import write_private_file
from yarvis.telemetry import Link
from yarvis.tools.base import Arguments, Tier, Tool, ToolError, UntrustedToolError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path
    from types import TracebackType
    from typing import Self

    from mcp_types import CallToolResult, ListToolsResult

    from yarvis.config.models import McpConfig, McpServerConfig
    from yarvis.paths import AppPaths

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S: Final = 30.0
"""Starting a server can mean downloading it first (``npx -y``, ``uvx``)."""
STOP_TIMEOUT_S: Final = 5.0
MAX_PAGES: Final = 20
MAX_TOOLS: Final = 200
MAX_DESCRIPTION_CHARS: Final = 1_000
MAX_SCHEMA_CHARS: Final = 20_000
MAX_ERROR_CHARS: Final = 500
_SPOKEN_VALUE_CHARS: Final = 60
_SPOKEN_ARGS_CHARS: Final = 240
_NAME_CHARS: Final = re.compile(r"[^A-Za-z0-9_-]")


class McpServerError(Exception):
    """An MCP server could not be started or listed."""


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """A tool as an MCP server describes it."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the name, description and argument schema (what the user approves)."""
        definition = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class Session(Protocol):
    """The part of an MCP client session Yarvis uses (``mcp.Client`` fits)."""

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        """One page of the server's tools."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> CallToolResult:
        """Call one tool."""
        ...


type Opener = Callable[[McpServerConfig], AbstractAsyncContextManager[Session]]
"""Connects to a server; the context lasts as long as Yarvis uses it."""


async def list_tools(session: Session) -> tuple[RemoteTool, ...]:
    """Every tool the server offers (bounded: a server cannot make Yarvis list forever)."""
    tools: list[RemoteTool] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        page = await session.list_tools(cursor=cursor)
        tools += (
            RemoteTool(tool.name, tool.description or "", dict(tool.input_schema))
            for tool in page.tools
        )
        cursor = page.next_cursor
        if cursor is None or len(tools) >= MAX_TOOLS:
            break
    return tuple(tools[:MAX_TOOLS])


class Connection:
    """One server, kept connected by a task of its own.

    The MCP SDK's connections must be opened and closed by the same task; calls can come
    from any task while it is open.
    """

    def __init__(self, server: McpServerConfig, opener: Opener) -> None:
        self.server = server
        self._opener = opener
        self._session: Session | None = None
        self._ready: asyncio.Future[tuple[RemoteTool, ...]] | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """Whether calls can be made."""
        return self._session is not None

    async def start(self, timeout_s: float = CONNECT_TIMEOUT_S) -> tuple[RemoteTool, ...]:
        """Connect and list the server's tools.

        Raises:
            McpServerError: If it cannot be started, reached or listed in time.
        """
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._own(self._ready))
        try:
            async with asyncio.timeout(timeout_s):
                return await asyncio.shield(self._ready)
        except TimeoutError:
            await self._cancel()  # it never got going: nothing to close politely
            msg = f"it did not answer within {timeout_s:.0f} seconds"
            raise McpServerError(msg) from None

    async def call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call one of the server's tools.

        Raises:
            ToolError: If the server is not running.
        """
        session = self._session
        if session is None:
            msg = f"The {self.server.name} server is not running."
            raise ToolError(msg)
        return await session.call_tool(name, arguments, read_timeout_seconds=self.server.timeout_s)

    async def stop(self) -> None:
        """Disconnect (and end the server's process, if Yarvis started it)."""
        self._stop.set()
        task = self._task
        if task is None or task.done():
            return
        try:
            async with asyncio.timeout(STOP_TIMEOUT_S):
                await asyncio.shield(task)
        except TimeoutError:
            await self._cancel()

    async def _cancel(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _own(self, ready: asyncio.Future[tuple[RemoteTool, ...]]) -> None:
        try:
            async with self._opener(self.server) as session:
                tools = await list_tools(session)
                self._session = session
                ready.set_result(tools)
                await self._stop.wait()
        except Exception as exc:  # noqa: BLE001 - any failure: report it, keep Yarvis running
            if not ready.done():
                ready.set_exception(McpServerError(_reason(exc)))
            else:
                logger.warning("MCP server %s stopped: %s", self.server.name, _reason(exc))
        finally:
            self._session = None


def _reason(exc: BaseException) -> str:
    """A short description of a failure (the first one, for an exception group)."""
    first = _first(exc)
    text = str(first)[:MAX_ERROR_CHARS]
    return f"{type(first).__name__}: {text}" if text else type(first).__name__


def _first(exc: BaseException) -> BaseException:
    if isinstance(exc, BaseExceptionGroup):
        inner = cast("BaseExceptionGroup[BaseException]", exc).exceptions
        return _first(inner[0]) if inner else cast("BaseException", exc)
    return exc


class _Validator(Protocol):
    def iter_errors(self, instance: object) -> Iterator[ValidationError]: ...


class McpArguments(Arguments):
    """The arguments of an MCP tool call, checked against the server's schema."""

    values: dict[str, JsonValue]


class McpTool(Tool[McpArguments]):
    """One allowlisted, approved tool of an MCP server."""

    Args = McpArguments
    untrusted_output = True

    def __init__(self, connection: Connection, remote: RemoteTool, tier: Tier) -> None:
        server = connection.server.name
        self.name = exposed_name(server, remote.name)
        self.description = f"From the {server} MCP server: {remote.description}"[
            :MAX_DESCRIPTION_CHARS
        ]
        self.tier = tier
        self.timeout_s = connection.server.timeout_s + 5.0
        self.server = server
        self.remote = remote
        self._connection = connection
        self._validator = cast("_Validator", Draft202012Validator(remote.input_schema))

    def spec(self) -> ToolSpec:
        """The tool as the model sees it: the schema the user approved."""
        return ToolSpec(self.name, self.description, self.remote.input_schema)

    def parse(self, raw: Mapping[str, Any]) -> McpArguments:
        """Check the arguments against the server's schema."""
        values = dict(raw)
        problems = sorted(self._validator.iter_errors(values), key=lambda error: error.path)
        if problems:
            first = problems[0]
            where = ".".join(str(part) for part in first.path) or "input"
            msg = f"Invalid arguments: {where}: {first.message[:200]}"
            raise ToolError(msg)
        return McpArguments.model_validate({"values": values})

    def sends_data_out(self, args: McpArguments) -> bool:
        """Always: Yarvis cannot know where a server sends what it is given."""
        del args
        return True

    async def run(self, args: McpArguments) -> str:
        """Call the tool; return what it said (the toolbox wraps it as outside content)."""
        result = await self._connection.call(self.remote.name, dict(args.values))
        text = result_text(result)
        if result.is_error:
            summary = f"The {self.server} server could not do it."
            raise UntrustedToolError(summary, text)
        return text

    def describe(self, args: McpArguments) -> str:
        """``use notes' add note with text "buy milk"``: built from the checked arguments."""
        action = f"use {self.server}'s {_words(self.remote.name)}"
        spoken = ", ".join(f"{_words(key)} {_spoken(value)}" for key, value in args.values.items())
        if len(spoken) > _SPOKEN_ARGS_CHARS:
            spoken = spoken[: _SPOKEN_ARGS_CHARS - 1] + "…"
        return f"{action} with {spoken}" if spoken else action


def exposed_name(server: str, tool: str) -> str:
    """The name the model calls: ``server__tool``, within the API's 64 characters."""
    name = _NAME_CHARS.sub("_", f"{server}__{tool}")
    if len(name) <= 64:  # noqa: PLR2004 - the API's limit
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[:55]}_{digest}"


def result_text(result: CallToolResult) -> str:
    """The readable part of a result (text; other content is only named)."""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else f"[{block.type} not shown]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    return "\n".join(parts) or "(the tool returned nothing)"


def _words(name: str) -> str:
    """``add_note`` / ``addNote`` → ``add note`` (said out loud in confirmations)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[_\-.]+", " ", spaced).strip().lower()


def _spoken(value: JsonValue) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        text = value if len(value) <= _SPOKEN_VALUE_CHARS else value[:_SPOKEN_VALUE_CHARS] + "…"
        return f'"{text}"'
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= _SPOKEN_VALUE_CHARS else text[:_SPOKEN_VALUE_CHARS] + "…"


# --------------------------------------------------------------------------- approvals


class Approvals:
    """The tool definitions the user approved (``yarvis mcp approve``), by fingerprint."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def of(cls, paths: AppPaths) -> Approvals:
        """Where this installation keeps them."""
        return cls(paths.data_dir / "mcp-approved.json")

    def load(self) -> dict[str, dict[str, str]]:
        """``{server: {tool: fingerprint}}``; empty if nothing was approved (or unreadable)."""
        try:
            data: object = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.warning("Could not read the approved MCP tools; approve them again")
            return {}
        if not isinstance(data, dict):
            return {}
        approved: dict[str, dict[str, str]] = {}
        for server, tools in cast("dict[object, object]", data).items():
            if isinstance(server, str) and isinstance(tools, dict):
                listed = cast("dict[object, object]", tools)
                approved[server] = {str(name): str(digest) for name, digest in listed.items()}
        return approved

    def approve(self, server: str, tools: Iterable[RemoteTool]) -> None:
        """Record these definitions as the approved ones for ``server`` (replacing its old)."""
        approved = self.load()
        approved[server] = {tool.name: tool.fingerprint for tool in tools}
        content = json.dumps(approved, indent=2, sort_keys=True) + "\n"
        write_private_file(self.path, content, overwrite=True)

    def forget(self, server: str) -> None:
        """Drop every approval for ``server``."""
        approved = self.load()
        if approved.pop(server, None) is not None:
            content = json.dumps(approved, indent=2, sort_keys=True) + "\n"
            write_private_file(self.path, content, overwrite=True)


@dataclass(frozen=True, slots=True)
class Review:
    """How a server's tools compare with the config and the approvals."""

    usable: tuple[tuple[RemoteTool, Tier], ...]
    """Listed in the config and approved as they are now."""
    unapproved: tuple[RemoteTool, ...]
    """Listed, but never approved or changed since (left out until approved)."""
    missing: tuple[str, ...]
    """Listed in the config, but the server has no such tool."""
    unlisted: tuple[str, ...]
    """Offered by the server, not listed in the config (never used)."""
    problems: tuple[str, ...] = ()
    """Listed tools that cannot be offered (an unusable schema, a name clash)."""


def review(
    server: McpServerConfig, tools: Sequence[RemoteTool], approved: Mapping[str, str]
) -> Review:
    """Sort a server's tools by what may be used (see ``Review``)."""
    offered = {tool.name: tool for tool in tools}
    usable: list[tuple[RemoteTool, Tier]] = []
    unapproved: list[RemoteTool] = []
    problems: list[str] = []
    for name, tier in server.tools.items():
        tool = offered.get(name)
        if tool is None:
            continue
        if problem := _unusable(tool):
            problems.append(f"{name}: {problem}")
        elif approved.get(name) == tool.fingerprint:
            usable.append((tool, Tier(tier)))
        else:
            unapproved.append(tool)
    return Review(
        usable=tuple(usable),
        unapproved=tuple(unapproved),
        missing=tuple(name for name in server.tools if name not in offered),
        unlisted=tuple(name for name in offered if name not in server.tools),
        problems=tuple(problems),
    )


def _unusable(tool: RemoteTool) -> str | None:
    if len(json.dumps(tool.input_schema)) > MAX_SCHEMA_CHARS:
        return "its argument schema is too large"
    if tool.input_schema.get("type") != "object":
        return "its arguments are not an object"
    try:
        Draft202012Validator.check_schema(tool.input_schema)
    except SchemaError:
        return "its argument schema is invalid"
    return None


# --------------------------------------------------------------------------- the hub


class McpHub:
    """Starts the configured servers and offers their approved tools (an async context)."""

    def __init__(self, config: McpConfig, approvals: Approvals, opener: Opener) -> None:
        self._config = config
        self._approvals = approvals
        self._opener = opener
        self._connections: list[Connection] = []
        self.tools: list[McpTool] = []
        self.servers: list[str] = []
        """The servers that are running and contribute at least one tool."""
        self.links: list[Link] = []
        """How every configured server is doing, for the dashboard."""

    async def __aenter__(self) -> Self:
        approved = self._approvals.load()
        connections = [Connection(server, self._opener) for server in self._config.servers]
        self._connections = connections
        listings = await asyncio.gather(
            *(connection.start() for connection in connections), return_exceptions=True
        )
        taken: set[str] = set()
        for connection, listing in zip(connections, listings, strict=True):
            name = connection.server.name
            key, label = f"mcp:{name}", f"MCP · {name}"
            if isinstance(listing, BaseException):
                logger.warning("MCP server %s is not available: %s", name, listing)
                self.links.append(Link(key, label, "error", f"not available: {listing}"))
                continue
            tools = self._offer(connection, listing, approved.get(name, {}), taken)
            if tools:
                self.servers.append(name)
                self.tools += tools
                count = f"{len(tools)} tool{'s' if len(tools) != 1 else ''}"
                self.links.append(Link(key, label, "ok", count))
            else:
                await connection.stop()
                self.links.append(
                    Link(key, label, "warn", f"no usable tools (yarvis mcp approve {name})")
                )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await asyncio.gather(*(connection.stop() for connection in self._connections))

    def _offer(
        self,
        connection: Connection,
        listing: Sequence[RemoteTool],
        approved: Mapping[str, str],
        taken: set[str],
    ) -> list[McpTool]:
        server = connection.server
        sorted_out = review(server, listing, approved)
        for tool in sorted_out.unapproved:
            logger.warning(
                "MCP tool %s.%s is new or changed since it was approved: left out. "
                "Run: yarvis mcp approve %s",
                server.name,
                tool.name,
                server.name,
            )
        for name in sorted_out.missing:
            logger.warning("MCP server %s has no tool called %s", server.name, name)
        for problem in sorted_out.problems:
            logger.warning("MCP tool %s.%s left out", server.name, problem)
        tools: list[McpTool] = []
        for remote, tier in sorted_out.usable:
            tool = McpTool(connection, remote, tier)
            if tool.name in taken:
                logger.warning("MCP tool %s clashes with another tool: left out", tool.name)
                continue
            taken.add(tool.name)
            tools.append(tool)
        logger.info("MCP server %s: %d tool(s) in use", server.name, len(tools))
        return tools
