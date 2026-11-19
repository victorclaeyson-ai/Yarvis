"""The ``yarvis`` command-line interface."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import re
from typing import TYPE_CHECKING, Annotated

import httpx2
import typer
from pydantic import SecretStr
from rich.markup import escape
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.table import Table

from yarvis import __version__, spotify_login
from yarvis.bench import SttCandidate, recommendation, run_bench
from yarvis.cli.extensions import mcp_app, skills_app
from yarvis.cli.memory import memory_app
from yarvis.cli.profile import profile_app
from yarvis.cli.services import Services, fail, load_config_or_exit, services_of
from yarvis.cli.ui import ConsoleUi
from yarvis.cli.usage import usage_app
from yarvis.config import Config, ConfigError, load_config, write_default_config
from yarvis.doctor import CheckResult, DoctorContext, Status, has_failures, run_doctor
from yarvis.log import configure_logging
from yarvis.models import (
    KOKORO_FILES,
    WAKE_WORD_FILES,
    WHISPER_APPROX_MB,
    ModelFile,
    ModelIntegrityError,
    kokoro_dir,
    missing,
    parakeet_dir,
    parakeet_files,
    wake_word_dir,
    whisper_present,
)
from yarvis.runtime import LoadableSTT, run_assistant
from yarvis.security.secrets import (
    InvalidSecretError,
    ResolvedSecret,
    SecretName,
    SecretSource,
)
from yarvis.speech.choice import (
    SttEngine,
    choose_engine,
    cloud_key,
)
from yarvis.tools.spotify import SpotifyKeys

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yarvis.speech.whisper import SttDevice

logger = logging.getLogger(__name__)


# Pretty exceptions render local variables, which could include secrets: keep them off.
app = typer.Typer(
    name="yarvis",
    help="Yarvis — a local-first voice assistant you can trust with your computer.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Manage the configuration file.", no_args_is_help=True)
key_app = typer.Typer(help="Manage API keys in the OS keychain.", no_args_is_help=True)
models_app = typer.Typer(help="Download and inspect speech models.", no_args_is_help=True)
spotify_app = typer.Typer(help="Connect your Spotify account.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(key_app, name="key")
app.add_typer(models_app, name="models")
app.add_typer(spotify_app, name="spotify")
app.add_typer(mcp_app, name="mcp")
app.add_typer(skills_app, name="skills")
app.add_typer(profile_app, name="profile")
app.add_typer(memory_app, name="memory")
app.add_typer(usage_app, name="usage")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yarvis {__version__}")
        raise typer.Exit


@app.callback()
def root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Yarvis command-line interface."""
    del version  # handled by the eager callback
    if ctx.obj is None:
        ctx.obj = Services.default()


# --------------------------------------------------------------------------- doctor

_STATUS_STYLE = {
    Status.OK: "[green]✓ ok[/]",
    Status.WARN: "[yellow]! warn[/]",
    Status.FAIL: "[red]✗ fail[/]",
}


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check that this machine is ready to run Yarvis."""
    services = services_of(ctx)
    context = DoctorContext.current(
        paths=services.paths,
        secrets=services.secrets,
        audio=services.audio,
    )
    results = run_doctor(context)
    services.console.print(_results_table(results))
    if has_failures(results):
        services.console.print(
            "\n[red]Some checks failed.[/] Fix them and run [bold]yarvis doctor[/] again."
        )
        raise typer.Exit(code=1)
    services.console.print("\n[green]All good.[/]")


def _results_table(results: list[CheckResult]) -> Table:
    table = Table(title="Yarvis doctor", title_justify="left", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details", overflow="fold")
    for result in results:
        detail = result.detail if result.hint is None else f"{result.detail}\n[dim]{result.hint}[/]"
        table.add_row(result.name, _STATUS_STYLE[result.status], detail)
    return table


# --------------------------------------------------------------------------- config


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Print where the configuration file lives."""
    services = services_of(ctx)
    services.console.print(str(services.paths.config_file), highlight=False, soft_wrap=True)


@config_app.command("init")
def config_init(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Create a commented default configuration file."""
    services = services_of(ctx)
    path = services.paths.config_file
    try:
        write_default_config(path, overwrite=force)
    except FileExistsError:
        services.err_console.print(
            f"[red]{escape(str(path))} already exists.[/] Use --force to overwrite it."
        )
        raise typer.Exit(code=1) from None
    services.console.print(f"Wrote {path}", highlight=False)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show the effective configuration (defaults merged with your file)."""
    services = services_of(ctx)
    try:
        config = load_config(services.paths.config_file)
    except ConfigError as exc:
        services.err_console.print(f"[red]{escape(str(exc))}[/]", highlight=False)
        raise typer.Exit(code=1) from None
    services.console.print_json(config.model_dump_json())


# --------------------------------------------------------------------------- keys


@key_app.command("set")
def key_set(
    ctx: typer.Context,
    name: Annotated[SecretName, typer.Argument(help="Which key to store.")],
) -> None:
    """Store an API key in the OS keychain (input is hidden)."""
    services = services_of(ctx)
    if name is SecretName.SPOTIFY_LOGIN:
        fail(services, "Log in through Spotify instead: yarvis spotify login")
    if name is SecretName.SPOTIFY:
        value = _spotify_credentials(services)
    elif name is SecretName.ASSEMBLYAI:
        services.console.print(
            "Sign up at https://www.assemblyai.com/dashboard/signup and copy the API key from "
            "your dashboard. Yarvis then streams what you say to it (never the wake word or "
            "anything said before it) to AssemblyAI for transcription, on its EU servers "
            "unless you set voice.cloud_region.",
            highlight=False,
        )
        value = typer.prompt("Paste your AssemblyAI API key", hide_input=True)
    else:
        value = typer.prompt(f"Paste your {name.value} API key", hide_input=True)
    try:
        services.secrets.set(name, value)
    except (InvalidSecretError, RuntimeError) as exc:
        services.err_console.print(f"[red]Not saved:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from None
    services.console.print(f"Saved the {name.value} key to the keychain.")


_SPOTIFY_ID = re.compile(r"^[0-9a-f]{32}$")


def _spotify_credentials(services: Services) -> str:
    """Ask for a Spotify developer app's client ID and secret; store them as one value."""
    services.console.print(
        "Create an app at https://developer.spotify.com/dashboard (any name; redirect URI "
        "http://127.0.0.1:8888/callback; tick 'Web API'), then copy its Client ID and secret.",
        highlight=False,
    )
    client_id = typer.prompt("Client ID").strip()
    secret = typer.prompt("Client secret", hide_input=True).strip()
    if not (_SPOTIFY_ID.match(client_id) and _SPOTIFY_ID.match(secret)):
        fail(services, "Spotify client IDs and secrets are 32 characters of 0-9 and a-f.")
    return f"{client_id}:{secret}"


@key_app.command("delete")
def key_delete(
    ctx: typer.Context,
    name: Annotated[SecretName, typer.Argument(help="Which key to delete.")],
) -> None:
    """Remove an API key from the OS keychain."""
    services = services_of(ctx)
    if services.secrets.delete(name):
        services.console.print(f"Deleted the {name.value} key.")
    else:
        services.console.print(f"No {name.value} key was stored in the keychain.")


@key_app.command("status")
def key_status(ctx: typer.Context) -> None:
    """Show which keys are configured and where. Never prints the keys themselves."""
    services = services_of(ctx)
    for name in SecretName:
        resolved = services.secrets.get(name)
        where = "not set" if resolved is None else f"set ({resolved.source.value})"
        services.console.print(f"{name.value}: {where}", highlight=False)


# --------------------------------------------------------------------------- run


@app.command()
def run(
    ctx: typer.Context,
    text: Annotated[
        bool, typer.Option("--text", help="Type instead of speaking (microphone stays off).")
    ] = False,
    mute: Annotated[
        bool, typer.Option("--mute", help="Show replies without speaking them.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Download missing models without asking.")
    ] = False,
    no_wake_word: Annotated[
        bool, typer.Option("--no-wake-word", help="Don't listen for “Hey Yarvis” this time.")
    ] = False,
) -> None:
    """Start a conversation with Yarvis in the terminal."""
    services = services_of(ctx)
    config, key = _prepare(
        services,
        voice_input=not text,
        voice_output=not mute,
        assume_yes=yes,
        wake_word=not no_wake_word,
    )
    cache_dir = services.paths.cache_dir
    ui = ConsoleUi(services.console, name=config.assistant.name)
    services.console.print(f"[bold]{config.assistant.name}[/] is ready. Say hello!")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            run_assistant(
                config,
                api_key=key.value,
                cache_dir=cache_dir,
                lines=services.lines(),
                ui=ui,
                backends=services.backends(),
                voice_input=not text,
                voice_output=not mute,
                on_loading=ui.notice,
                spotify=_spotify(services),
            )
        )
    services.console.print("\n[dim]Bye.[/]")


@app.command("app")
def app_command(
    ctx: typer.Context,
    text: Annotated[
        bool, typer.Option("--text", help="Type instead of speaking (microphone stays off).")
    ] = False,
    mute: Annotated[
        bool, typer.Option("--mute", help="Show replies without speaking them.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Download missing models without asking.")
    ] = False,
    no_wake_word: Annotated[
        bool, typer.Option("--no-wake-word", help="Don't listen for “Hey Yarvis” this time.")
    ] = False,
) -> None:
    """Open the Yarvis app window."""
    services = services_of(ctx)
    config, key = _prepare(
        services,
        voice_input=not text,
        voice_output=not mute,
        assume_yes=yes,
        wake_word=not no_wake_word,
    )
    services.console.print(f"Opening [bold]{config.assistant.name}[/]… (close the window to quit)")
    try:
        services.app_runner(
            config,
            api_key=key.value,
            cache_dir=services.paths.cache_dir,
            backends=services.backends(),
            voice_input=not text,
            voice_output=not mute,
            spotify=_spotify(services),
        )
    except RuntimeError as exc:
        fail(
            services,
            f"{exc}. On Linux, install a GUI backend first: uv pip install 'pywebview[qt]'. "
            "The terminal version still works: yarvis run",
        )


def _prepare(
    services: Services,
    *,
    voice_input: bool,
    voice_output: bool,
    assume_yes: bool,
    wake_word: bool = True,
) -> tuple[Config, ResolvedSecret]:
    """Checks shared by ``run`` and ``app``: config, logging, API key, models.

    Returns the configuration as it will actually run (e.g. without the wake word if its
    model is unavailable).
    """
    config = load_config_or_exit(services)
    if not (wake_word and voice_input):
        config = _without_wake_word(config)
    services.paths.ensure_all()
    configure_logging(
        level=config.logging.level,
        log_dir=services.paths.log_dir if config.logging.file else None,
        console=services.err_console,
    )
    if config.llm.provider != "anthropic":
        fail(services, 'The Ollama provider arrives in Phase 3. Set llm.provider = "anthropic".')
    key = services.secrets.get(SecretName.ANTHROPIC)
    if key is None:
        fail(services, "No Anthropic API key found. Run: yarvis key set anthropic")
    if voice_output and not _ensure_kokoro(services, assume_yes=assume_yes):
        fail(services, "Yarvis cannot speak without its voice model. Use --mute to skip it.")
    if config.voice.wake_word and not _ensure_wake_word(services, assume_yes=assume_yes):
        services.console.print(
            "[yellow]Continuing without the wake word: press Space (app) or Enter (terminal) "
            "to talk.[/]"
        )
        config = _without_wake_word(config)
    if voice_input:
        config = _with_speech_recognition(services, config, assume_yes=assume_yes)
    return config, key


def _with_speech_recognition(services: Services, config: Config, *, assume_yes: bool) -> Config:
    """Resolve ``stt = "auto"`` and make sure the chosen model is (being) downloaded."""
    engine = choose_engine(config.voice, parakeet_ok=services.parakeet_ok())
    if engine == "parakeet" and not _ensure_parakeet(services, assume_yes=assume_yes):
        services.console.print("[yellow]Using Whisper for speech recognition instead.[/]")
        engine = "faster-whisper"
    config = config.model_copy(update={"voice": config.voice.model_copy(update={"stt": engine})})
    if cloud_key(config.voice, services.secrets) is not None:
        where = "EU" if config.voice.cloud_region == "eu" else "nearest"
        backup = "Parakeet" if engine == "parakeet" else f"Whisper {config.voice.stt_model}"
        services.console.print(
            f"[dim]Speech recognition: AssemblyAI ({where} servers), with {backup} on this "
            "computer as backup.[/]"
        )
    if engine == "faster-whisper" and not whisper_present(
        services.paths.cache_dir, config.voice.stt_model
    ):
        size = WHISPER_APPROX_MB[config.voice.stt_model]
        services.console.print(
            f"[dim]First run: downloading the speech recognition model (~{size} MB)…[/]"
        )
    return config


def _spotify(services: Services) -> SpotifyKeys | None:
    """The Spotify developer app's credentials and the user's login, if stored."""
    app_keys = services.secrets.get(SecretName.SPOTIFY)
    if app_keys is None:
        return None
    login = services.secrets.get(SecretName.SPOTIFY_LOGIN)

    def save_login(token: str) -> None:
        try:
            services.secrets.set(SecretName.SPOTIFY_LOGIN, token)
        except (InvalidSecretError, RuntimeError) as exc:  # pragma: no cover - keychain trouble
            logger.warning("Could not store the renewed Spotify login: %s", exc)

    return SpotifyKeys(
        app_keys.value,
        login.value if login is not None else None,
        save_login if login is not None and login.source is SecretSource.KEYCHAIN else None,
    )


@spotify_app.command("login")
def spotify_login_command(ctx: typer.Context) -> None:
    """Let Yarvis play your playlists and control Spotify in the background (asks once)."""
    services = services_of(ctx)
    app_keys = services.secrets.get(SecretName.SPOTIFY)
    if app_keys is None:
        fail(services, "First store your Spotify app's keys: yarvis key set spotify")
    client_id, _, secret = app_keys.value.get_secret_value().partition(":")
    services.console.print(
        "Your browser will open Spotify. Allow Yarvis to read your playlists and control "
        "playback, then come back here. (Yarvis listens on "
        f"{spotify_login.REDIRECT_URI} until Spotify answers.)",
        highlight=False,
    )
    try:
        token = services.spotify_login(client_id, SecretStr(secret))
        services.secrets.set(SecretName.SPOTIFY_LOGIN, token)
    except (spotify_login.LoginError, InvalidSecretError, RuntimeError) as exc:
        fail(services, f"Spotify login failed: {exc}")
    services.console.print(
        "[green]Connected.[/] Yarvis can now play your own playlists. To disconnect: "
        "yarvis key delete spotify-login"
    )


@app.command()
def bench(
    ctx: typer.Context,
    rounds: Annotated[int, typer.Option(min=1, max=20, help="Measurements per component.")] = 3,
    claude: Annotated[
        bool, typer.Option("--claude/--no-claude", help="Also measure Claude (a few tokens).")
    ] = True,
) -> None:
    """Measure each stage on this machine and recommend the fastest settings."""
    services = services_of(ctx)
    config, key = _prepare(services, voice_input=True, voice_output=True, assume_yes=False)
    backends = services.backends()
    cache = services.paths.cache_dir
    devices: list[SttDevice] = ["cpu"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        devices.append("mps")

    def make_stt(engine: SttEngine, device: SttDevice = "cpu") -> Callable[[], LoadableSTT]:
        update = {"stt": engine, "stt_device": device, "cloud_stt": False}  # local engines only
        voice = config.voice.model_copy(update=update)
        return lambda: backends.stt(config.model_copy(update={"voice": voice}), cache)

    recognizers = [
        SttCandidate(
            f"whisper {config.voice.stt_model} on {device}",
            device,
            make_stt("faster-whisper", device),
        )
        for device in devices
    ]
    if config.voice.stt == "parakeet":  # resolved by _prepare: installed, downloaded
        where = "mps" if services.parakeet_build() == "mlx" else "cpu"
        recognizers.insert(0, SttCandidate(f"parakeet on {where}", where, make_stt("parakeet")))
    results = asyncio.run(
        run_bench(
            tts=backends.tts(config, cache),
            recognizers=recognizers,
            llm=backends.llm(config, key.value) if claude else None,
            rounds=rounds,
            on_progress=lambda message: services.console.print(f"[dim]{message}[/]"),
        )
    )
    table = Table(title="Yarvis bench", title_justify="left")
    table.add_column("Component", style="bold")
    table.add_column("Setting")
    table.add_column("Median", justify="right")
    table.add_column("Runs (ms)")
    table.add_column("Notes", overflow="fold")
    for result in results:
        table.add_row(
            result.component,
            escape(result.setting),
            f"{result.median_ms:.0f} ms",
            " ".join(f"{run:.0f}" for run in result.runs_ms),
            escape(result.note),
        )
    services.console.print(table)
    if advice := recommendation(results):
        services.console.print(f"\n{escape(advice)}")


# --------------------------------------------------------------------------- models


@models_app.command("download")
def models_download(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Download the voice, wake-word and speech recognition models ahead of time."""
    services = services_of(ctx)
    config = load_config_or_exit(services)
    services.paths.ensure_all()
    if not _ensure_kokoro(services, assume_yes=yes):
        raise typer.Exit(code=1)
    if config.voice.wake_word and not _ensure_wake_word(services, assume_yes=yes):
        raise typer.Exit(code=1)
    if choose_engine(config.voice, parakeet_ok=services.parakeet_ok()) == "parakeet":
        if not _ensure_parakeet(services, assume_yes=yes):
            raise typer.Exit(code=1)
    elif not whisper_present(services.paths.cache_dir, config.voice.stt_model):
        services.console.print(f"Downloading Whisper {config.voice.stt_model}…")
        stt = services.backends().stt(config, services.paths.cache_dir)
        stt.load()
    services.console.print("[green]All speech models are ready.[/]")


def _without_wake_word(config: Config) -> Config:
    voice = config.voice.model_copy(update={"wake_word": False})
    return config.model_copy(update={"voice": voice})


def _ensure_kokoro(services: Services, *, assume_yes: bool) -> bool:
    directory = kokoro_dir(services.paths.cache_dir)
    todo = missing(KOKORO_FILES, directory)
    if not todo:
        return True
    megabytes = sum(model.size for model in todo) / 1_000_000
    services.console.print(f"Yarvis needs its voice model (Kokoro, {megabytes:.0f} MB, one time).")
    if not assume_yes and not typer.confirm("Download it now?", default=True):
        return False
    return _download(services, todo, directory)


def _ensure_parakeet(services: Services, *, assume_yes: bool) -> bool:
    build = services.parakeet_build()
    directory = parakeet_dir(services.paths.cache_dir, build)
    todo = missing(parakeet_files(build), directory)
    if not todo:
        return True
    gigabytes = sum(model.size for model in todo) / 1_000_000_000
    where = "your Mac's GPU" if build == "mlx" else "this computer's processor"
    services.console.print(
        "Yarvis can understand you much better with Parakeet, a speech recognition model "
        f"that runs on {where} ({gigabytes:.1f} GB, one time; NVIDIA, CC BY 4.0)."
    )
    if not assume_yes and not typer.confirm("Download it now?", default=True):
        return False
    return _download(services, todo, directory)


def _ensure_wake_word(services: Services, *, assume_yes: bool) -> bool:
    directory = wake_word_dir(services.paths.cache_dir)
    todo = missing(WAKE_WORD_FILES, directory)
    if not todo:
        return True
    megabytes = max(1, round(sum(model.size for model in todo) / 1_000_000))
    services.console.print(
        f"The “Hey Yarvis” wake word needs a small model (openWakeWord, {megabytes} MB, one "
        "time). It is free for personal, non-commercial use (CC BY-NC-SA 4.0)."
    )
    if not assume_yes and not typer.confirm("Download it now?", default=True):
        return False
    return _download(services, todo, directory)


def _download(services: Services, todo: list[ModelFile], directory: Path) -> bool:
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=services.console,
        ) as progress:
            for model in todo:
                task = progress.add_task(model.name, total=model.size)
                services.downloader(model, directory, lambda n, t=task: progress.advance(t, n))
    except (ModelIntegrityError, httpx2.HTTPError) as exc:
        services.err_console.print(f"[red]Download failed:[/] {escape(str(exc))}")
        return False
    return True


def main() -> None:
    """Console-script entry point."""
    app()
