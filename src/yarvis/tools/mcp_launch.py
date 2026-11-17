"""Opening real MCP connections: starting a server's program, or reaching it over HTTPS.

Kept apart from ``tools.mcp`` (which only sees a ``Session``) so the rules of §5.4 that
touch the operating system are in one place: the scrubbed environment, the secrets from
the keychain, the server's own folder and log, and HTTPS with an optional bearer token.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Final, TextIO

from yarvis.paths import PRIVATE_FILE_MODE, ensure_private_dir
from yarvis.tools.mcp import McpServerError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping
    from pathlib import Path

    from mcp import StdioServerParameters
    from pydantic import SecretStr

    from yarvis.config.models import McpServerConfig
    from yarvis.tools.mcp import Session

HTTP_TIMEOUT_S: Final = 30.0
HTTP_READ_TIMEOUT_S: Final = 300.0
"""A remote server may keep a response stream open while it works."""


def secret_key(server: str, name: str) -> str:
    """Where ``yarvis mcp secret <server> <name>`` keeps a server's secret in the keychain."""
    return f"mcp:{server}:{name}"


def _yarvis_version() -> str:
    try:
        return version("yarvis")
    except PackageNotFoundError:  # pragma: no cover - only when run from a bare checkout
        return "0"


@dataclass(frozen=True, slots=True)
class Launcher:
    """Opens connections to configured servers (an ``Opener`` for ``tools.mcp``)."""

    secrets: Callable[[str], SecretStr | None]
    """Looks up a secret by key (``SecretStore.get_extra``)."""
    work_dir: Path
    """Each server Yarvis starts runs in its own private folder below this one."""
    log_dir: Path
    """What a started server prints goes to ``mcp-<name>.log`` here."""
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))

    def stdio_parameters(self, server: McpServerConfig) -> StdioServerParameters:
        """How to start ``server``: its program, a scrubbed environment and its folder.

        The MCP SDK adds only harmless variables (PATH, HOME, ...) to the ``env`` given
        here: Yarvis's own keys are never passed on.

        Raises:
            McpServerError: If a secret the server needs is not in the keychain.
        """
        from mcp import StdioServerParameters  # noqa: PLC0415 - heavy, only when used

        if not server.command:
            msg = "it has no command"
            raise McpServerError(msg)
        env = {name: self.environ[name] for name in server.env if name in self.environ}
        for name in server.secrets:
            value = self.secrets(secret_key(server.name, name))
            if value is None:
                msg = f"its secret {name} is not set (yarvis mcp secret {server.name} {name})"
                raise McpServerError(msg)
            env[name] = value.get_secret_value()
        folder = ensure_private_dir(ensure_private_dir(self.work_dir) / server.name)
        program, *args = server.command
        return StdioServerParameters(command=program, args=args, env=env, cwd=folder)

    def headers(self, server: McpServerConfig) -> dict[str, str]:
        """The HTTP headers for a remote server (its bearer token, if it has one).

        Raises:
            McpServerError: If the token is not in the keychain.
        """
        if server.token is None:
            return {}
        value = self.secrets(secret_key(server.name, server.token))
        if value is None:
            msg = f"its token is not set (yarvis mcp secret {server.name} {server.token})"
            raise McpServerError(msg)
        return {"Authorization": f"Bearer {value.get_secret_value()}"}

    @asynccontextmanager
    async def __call__(self, server: McpServerConfig) -> AsyncGenerator[Session]:
        """Connect to ``server`` for as long as the context lasts."""
        from mcp import Client, Implementation  # noqa: PLC0415 - heavy, only when used
        from mcp.client.stdio import stdio_client  # noqa: PLC0415
        from mcp.client.streamable_http import streamable_http_client  # noqa: PLC0415

        async with AsyncExitStack() as stack:
            if server.url is None:
                parameters = self.stdio_parameters(server)
                log = stack.enter_context(self._open_log(server.name))
                transport = stdio_client(parameters, errlog=log)
            elif not server.url.startswith("https://"):
                msg = "remote servers are only reached over HTTPS"
                raise McpServerError(msg)
            else:
                import httpx2  # noqa: PLC0415

                timeout = httpx2.Timeout(HTTP_TIMEOUT_S, read=HTTP_READ_TIMEOUT_S)
                http = httpx2.AsyncClient(headers=self.headers(server), timeout=timeout)
                await stack.enter_async_context(http)
                transport = streamable_http_client(server.url, http_client=http)
            info = Implementation(name="yarvis", version=_yarvis_version())
            client = Client(transport, client_info=info, cache=None)
            yield await stack.enter_async_context(client)

    def _open_log(self, name: str) -> TextIO:
        """A private log for what the server prints (rewritten at each start)."""
        ensure_private_dir(self.log_dir)
        path = self.log_dir / f"mcp-{name}.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "w", encoding="utf-8")
