"""What every CLI command needs: its dependencies (``Services``) and a few shared helpers."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

import typer
from rich.console import Console
from rich.markup import escape

from yarvis import spotify_login
from yarvis.audio.devices import AudioProbe, SoundDeviceProbe
from yarvis.cli.ui import stdin_lines
from yarvis.config import Config, ConfigError, load_config
from yarvis.models import ModelFile, download
from yarvis.paths import AppPaths
from yarvis.runtime import Backends
from yarvis.security.secrets import SecretStore
from yarvis.speech.choice import parakeet_available, this_build
from yarvis.tools.mcp_launch import Launcher

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from pydantic import SecretStr

    from yarvis.models import ParakeetBuild
    from yarvis.tools.mcp import Opener
    from yarvis.tools.spotify import SpotifyKeys


def run_app(
    config: Config,
    *,
    api_key: SecretStr,
    cache_dir: Path,
    backends: Backends,
    voice_input: bool,
    voice_output: bool,
    spotify: SpotifyKeys | None = None,
) -> None:  # pragma: no cover - opens a real window
    """Open the Yarvis window."""
    from yarvis.app.window import run_app as run_window  # noqa: PLC0415 - GUI imports

    run_window(
        config,
        api_key=api_key,
        cache_dir=cache_dir,
        backends=backends,
        voice_input=voice_input,
        voice_output=voice_output,
        spotify=spotify,
    )


def mcp_launcher(paths: AppPaths, secrets: SecretStore) -> Opener:
    """Starts or reaches the configured MCP servers (their folders and logs in ``paths``)."""
    return Launcher(secrets.get_extra, work_dir=paths.data_dir / "mcp", log_dir=paths.log_dir)


@dataclass(slots=True)
class Services:
    """Dependencies of the CLI. Tests pass their own instance as the Click ``obj``."""

    paths: AppPaths
    secrets: SecretStore
    audio: AudioProbe
    console: Console
    err_console: Console
    backends: Callable[[], Backends] = Backends.real
    lines: Callable[[], AsyncIterator[str]] = stdin_lines
    downloader: Callable[[ModelFile, Path, Callable[[int], None]], Path] = field(
        default=lambda model, directory, progress: download(model, directory, progress=progress)
    )
    app_runner: Callable[..., None] = field(default_factory=lambda: run_app)
    parakeet_ok: Callable[[], bool] = parakeet_available
    """Whether Parakeet speech recognition can run here."""
    parakeet_build: Callable[[], ParakeetBuild] = this_build
    """Which Parakeet build runs here: the Apple GPU (MLX) or the processor (ONNX)."""
    spotify_login: Callable[[str, SecretStr], str] = field(
        default=lambda client_id, secret: spotify_login.login(
            client_id, secret, open_url=webbrowser.open
        )
    )
    """Runs the Spotify consent flow in the browser; returns a refresh token."""
    mcp_opener: Callable[[AppPaths, SecretStore], Opener] = mcp_launcher
    """Connects to MCP servers (``yarvis mcp approve``)."""

    @classmethod
    def default(cls) -> Services:
        """Real implementations for normal use."""
        return cls(
            paths=AppPaths.default(),
            secrets=SecretStore(),
            audio=SoundDeviceProbe(),
            console=Console(),
            err_console=Console(stderr=True),
        )


def services_of(ctx: typer.Context) -> Services:
    """The CLI's ``Services`` (set up by the root command)."""
    obj: object = ctx.find_root().obj
    if not isinstance(obj, Services):  # pragma: no cover - guaranteed by the root callback
        msg = "CLI services were not initialised"
        raise TypeError(msg)
    return obj


def load_config_or_exit(services: Services) -> Config:
    """The configuration, or a readable error and exit code 1."""
    try:
        return load_config(services.paths.config_file)
    except ConfigError as exc:
        services.err_console.print(f"[red]{escape(str(exc))}[/]", highlight=False)
        raise typer.Exit(code=1) from None


def fail(services: Services, message: str) -> NoReturn:
    """Print an error and exit with code 1."""
    services.err_console.print(f"[red]✗ {escape(message)}[/]", highlight=False)
    raise typer.Exit(code=1)
