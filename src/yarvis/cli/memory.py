"""``yarvis memory``: see, add, forget, export and clear what Yarvis remembers (ADR 0017).

Why it exists: the user stays in charge of what is kept about them, without having to ask
Yarvis out loud. It must never print anything but the facts themselves.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from yarvis.cli.services import fail, services_of
from yarvis.memory import MemoryBook, MemoryBookError

memory_app = typer.Typer(
    help="See and change what Yarvis remembers about you.", no_args_is_help=True
)


@memory_app.command("list")
def memory_list(ctx: typer.Context) -> None:
    """Every fact Yarvis remembers, with its number."""
    services = services_of(ctx)
    book = MemoryBook.of(services.paths)
    try:
        facts = book.load()
    except MemoryBookError as exc:
        fail(services, str(exc))
    if not facts:
        services.console.print(
            'Nothing remembered yet. Tell Yarvis about yourself, or: yarvis memory add "..."'
        )
        return
    table = Table(title="What Yarvis remembers about you")
    table.add_column("#", justify="right")
    table.add_column("Fact")
    table.add_column("Since")
    for fact in facts:
        table.add_row(str(fact.id), escape(fact.text), fact.added.astimezone().strftime("%d %b %Y"))
    services.console.print(table)
    services.console.print(
        f"[dim]Forget one with: yarvis memory forget <number>. Kept in {escape(str(book.path))}[/]",
        highlight=False,
    )


@memory_app.command("add")
def memory_add(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help='The fact, e.g. "Prefers tea over coffee."')],
) -> None:
    """Tell Yarvis a fact to remember."""
    services = services_of(ctx)
    try:
        fact = MemoryBook.of(services.paths).add(text, source="cli")
    except MemoryBookError as exc:
        fail(services, str(exc))
    services.console.print(f"Remembered as fact {fact.id}.")


@memory_app.command("forget")
def memory_forget(
    ctx: typer.Context,
    number: Annotated[int, typer.Argument(min=1, help="The number from: yarvis memory list")],
) -> None:
    """Forget one fact."""
    services = services_of(ctx)
    try:
        fact = MemoryBook.of(services.paths).forget(number)
    except MemoryBookError as exc:
        fail(services, str(exc))
    services.console.print(f"Forgot fact {fact.id}: {escape(fact.text)}", highlight=False)


@memory_app.command("export")
def memory_export(ctx: typer.Context) -> None:
    """Print every fact as JSON (redirect it to a file to keep a copy)."""
    services = services_of(ctx)
    try:
        exported = MemoryBook.of(services.paths).export()
    except MemoryBookError as exc:
        fail(services, str(exc))
    typer.echo(exported)


@memory_app.command("clear")
def memory_clear(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask first.")] = False,
) -> None:
    """Forget everything Yarvis remembers."""
    services = services_of(ctx)
    if not yes and not typer.confirm("Forget everything Yarvis remembers about you?"):
        raise typer.Exit(code=1)
    try:
        count = MemoryBook.of(services.paths).clear()
    except MemoryBookError as exc:
        fail(services, str(exc))
    if count:
        services.console.print(f"Forgot {count} fact{'s' if count != 1 else ''}.")
    else:
        services.console.print("There was nothing to forget.")
