"""``yarvis profile``: the user's own description of themselves (ADR 0014)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.markup import escape

from yarvis.cli.services import fail, services_of
from yarvis.profile import (
    MAX_PROFILE_CHARS,
    ProfileError,
    create_profile,
    load_profile,
    profile_path,
)

profile_app = typer.Typer(
    help="Tell Yarvis who you are, what you work on and how to talk to you.",
    no_args_is_help=True,
)


def open_in_editor(path: Path) -> None:  # pragma: no cover - opens a real editor
    """Open ``path`` in the computer's text editor (without waiting on Windows and macOS)."""
    if sys.platform == "win32":
        notepad = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "notepad.exe"
        subprocess.Popen([str(notepad), str(path)])  # noqa: S603 - fixed program, our file
    elif sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/open", "-t", str(path)])  # noqa: S603 - fixed program
    else:
        subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607 - the desktop's opener


@profile_app.command("edit")
def profile_edit(ctx: typer.Context) -> None:
    """Open your profile in a text editor (a template the first time)."""
    services = services_of(ctx)
    services.paths.ensure_all()
    path = profile_path(services.paths)
    created = create_profile(path)
    services.open_editor(path)
    services.console.print(
        f"{'Created' if created else 'Opened'} {escape(str(path))}. Save it, and Yarvis uses "
        "it from your next question (no restart). It is sent to Claude with every request: "
        "no passwords or keys. To check what is sent: yarvis profile show",
        highlight=False,
    )


@profile_app.command("show")
def profile_show(ctx: typer.Context) -> None:
    """Show exactly what Yarvis sends about you (comments and empty lines left out)."""
    services = services_of(ctx)
    try:
        text = load_profile(profile_path(services.paths))
    except ProfileError as exc:
        fail(services, f"Your profile is not used: {exc}")
    if not text:
        services.console.print("Nothing written yet. Start with: yarvis profile edit")
        return
    services.console.print(escape(text), highlight=False)
    services.console.print(
        f"\n[dim]{len(text):,} of {MAX_PROFILE_CHARS:,} characters, sent with every request.[/]"
    )


@profile_app.command("path")
def profile_where(ctx: typer.Context) -> None:
    """Print where your profile lives."""
    services = services_of(ctx)
    services.console.print(str(profile_path(services.paths)), highlight=False)
