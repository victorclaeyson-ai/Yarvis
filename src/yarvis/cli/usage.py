"""``yarvis usage``: what the cloud services cost, and how fast Yarvis answers (ADR 0015)."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Annotated

import typer
from rich.markup import escape
from rich.table import Table

from yarvis.cli.services import services_of
from yarvis.usage import KEEP_DAYS, PRICES_AS_OF, UsageBook, local_today, percentile, total

if TYPE_CHECKING:
    from yarvis.usage import Day

usage_app = typer.Typer(
    help="Show what Yarvis's cloud services cost, and how fast it answers.",
    invoke_without_command=True,
)


def _usd(value: float) -> str:
    if value == 0:
        return "$0"
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"  # noqa: PLR2004 - a cent


def _seconds(ms: int | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.2f} s"


def _row(label: str, day: Day) -> list[str]:
    return [
        label,
        str(day.turns),
        _seconds(percentile(day.latencies_ms, 0.5)),
        _usd(day.claude_usd),
        _usd(day.search_usd),
        _usd(day.speech_usd),
        _usd(day.total_usd),
    ]


@usage_app.callback()
def usage_show(
    ctx: typer.Context,
    days: Annotated[
        int, typer.Option("--days", min=1, max=KEEP_DAYS, help="How many days to show.")
    ] = 7,
) -> None:
    """Cost and speed per day: Claude, web searches, AssemblyAI, and the time to first word."""
    if ctx.invoked_subcommand is not None:
        return
    services = services_of(ctx)
    record = UsageBook.of(services.paths).days()
    first = local_today() - timedelta(days=days - 1)
    shown = {day: numbers for day, numbers in record.items() if day >= first}
    if not shown:
        services.console.print("Nothing counted yet. Talk to Yarvis, then look again.")
        return
    table = Table(title=f"Yarvis, the last {days} day{'s' if days != 1 else ''}")
    for column in ("Day", "Turns", "First word (p50)", "Claude", "Web search", "Speech", "Total"):
        table.add_column(column, justify="left" if column == "Day" else "right")
    for day, numbers in shown.items():
        table.add_row(*_row(day.strftime("%a %d %b"), numbers))
    together = total(shown.values())
    table.add_section()
    table.add_row(*_row("Together", together), style="bold")
    services.console.print(table)
    p90 = percentile(together.latencies_ms, 0.9)
    if p90 is not None:
        services.console.print(
            f"Time to first word over {len(together.latencies_ms)} spoken turns: "
            f"p50 {_seconds(percentile(together.latencies_ms, 0.5))}, p90 {_seconds(p90)}.",
            highlight=False,
        )
    if together.unpriced:
        services.console.print(
            "[yellow]Some use was of a model without a known price: the totals are too low.[/]"
        )
    services.console.print(
        f"[dim]Estimates at list prices in US dollars, as of {PRICES_AS_OF}; your invoice is "
        "what counts. Only numbers are kept, never what was said: "
        f"{escape(str(UsageBook.of(services.paths).path))}[/]",
        highlight=False,
    )


@usage_app.command("clear")
def usage_clear(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask first.")] = False,
) -> None:
    """Delete every number counted so far."""
    services = services_of(ctx)
    if not yes and not typer.confirm("Delete Yarvis's usage numbers?"):
        raise typer.Exit(code=1)
    if UsageBook.of(services.paths).clear():
        services.console.print("Deleted. Counting starts again from zero.")
    else:
        services.console.print("There was nothing to delete.")
