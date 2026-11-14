"""The ``yarvis`` command-line interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from yarvis import __version__
from yarvis.audio.devices import AudioProbe, SoundDeviceProbe
from yarvis.config import ConfigError, load_config, write_default_config
from yarvis.doctor import CheckResult, DoctorContext, Status, has_failures, run_doctor
from yarvis.paths import AppPaths
from yarvis.security.secrets import InvalidSecretError, SecretName, SecretStore


@dataclass(slots=True)
class Services:
    """Dependencies of the CLI. Tests pass their own instance as the Click ``obj``."""

    paths: AppPaths
    secrets: SecretStore
    audio: AudioProbe
    console: Console
    err_console: Console

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
app.add_typer(config_app, name="config")
app.add_typer(key_app, name="key")


def _services(ctx: typer.Context) -> Services:
    obj: object = ctx.find_root().obj
    if not isinstance(obj, Services):  # pragma: no cover - guaranteed by the root callback
        msg = "CLI services were not initialised"
        raise TypeError(msg)
    return obj


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
    services = _services(ctx)
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
    services = _services(ctx)
    services.console.print(str(services.paths.config_file), highlight=False, soft_wrap=True)


@config_app.command("init")
def config_init(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Create a commented default configuration file."""
    services = _services(ctx)
    path = services.paths.config_file
    try:
        write_default_config(path, overwrite=force)
    except FileExistsError:
        services.err_console.print(f"[red]{path} already exists.[/] Use --force to overwrite it.")
        raise typer.Exit(code=1) from None
    services.console.print(f"Wrote {path}", highlight=False)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show the effective configuration (defaults merged with your file)."""
    services = _services(ctx)
    try:
        config = load_config(services.paths.config_file)
    except ConfigError as exc:
        services.err_console.print(f"[red]{exc}[/]", highlight=False)
        raise typer.Exit(code=1) from None
    services.console.print_json(config.model_dump_json())


# --------------------------------------------------------------------------- keys


@key_app.command("set")
def key_set(
    ctx: typer.Context,
    name: Annotated[SecretName, typer.Argument(help="Which key to store.")],
) -> None:
    """Store an API key in the OS keychain (input is hidden)."""
    services = _services(ctx)
    value: str = typer.prompt(f"Paste your {name.value} API key", hide_input=True)
    try:
        services.secrets.set(name, value)
    except (InvalidSecretError, RuntimeError) as exc:
        services.err_console.print(f"[red]Not saved:[/] {exc}")
        raise typer.Exit(code=1) from None
    services.console.print(f"Saved the {name.value} key to the keychain.")


@key_app.command("delete")
def key_delete(
    ctx: typer.Context,
    name: Annotated[SecretName, typer.Argument(help="Which key to delete.")],
) -> None:
    """Remove an API key from the OS keychain."""
    services = _services(ctx)
    if services.secrets.delete(name):
        services.console.print(f"Deleted the {name.value} key.")
    else:
        services.console.print(f"No {name.value} key was stored in the keychain.")


@key_app.command("status")
def key_status(ctx: typer.Context) -> None:
    """Show which keys are configured and where. Never prints the keys themselves."""
    services = _services(ctx)
    for name in SecretName:
        resolved = services.secrets.get(name)
        where = "not set" if resolved is None else f"set ({resolved.source.value})"
        services.console.print(f"{name.value}: {where}", highlight=False)


def main() -> None:
    """Console-script entry point."""
    app()
