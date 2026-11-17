"""``yarvis mcp`` and ``yarvis skills``: what the user adds to Yarvis, and approves (ADR 0013)."""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TC003 - typer reads the annotations at runtime
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.markup import escape
from rich.table import Table

from yarvis.cli.services import fail, load_config_or_exit, services_of
from yarvis.security.secrets import InvalidSecretError
from yarvis.tools.mcp import Approvals, Connection, McpServerError, review
from yarvis.tools.mcp_launch import secret_key
from yarvis.tools.skills import SkillError, SkillLibrary, read_skill

if TYPE_CHECKING:
    from yarvis.cli.services import Services
    from yarvis.config.models import McpServerConfig
    from yarvis.tools.mcp import RemoteTool

mcp_app = typer.Typer(help="Connect MCP servers and approve their tools.", no_args_is_help=True)
skills_app = typer.Typer(help="Add Agent Skills (instructions only).", no_args_is_help=True)

_TIERS = {0: "0 read only", 1: "1 reversible", 2: "2 asks first", 3: "3 never runs"}
_SHOWN_CHARS = 400


# --------------------------------------------------------------------------- mcp


@mcp_app.command("list")
def mcp_list(ctx: typer.Context) -> None:
    """Show the configured MCP servers, their tools and what is approved."""
    services = services_of(ctx)
    config = load_config_or_exit(services)
    if not config.mcp.servers:
        services.console.print(
            "No MCP servers yet. Add one under [[mcp.servers]] in the config "
            "(yarvis config path), then: yarvis mcp approve <name>",
            highlight=False,
        )
        return
    approved = Approvals.of(services.paths).load()
    table = Table(title="MCP servers", title_justify="left")
    table.add_column("Server", style="bold")
    table.add_column("Runs")
    table.add_column("Tools (tier)")
    table.add_column("Secrets")
    for server in config.mcp.servers:
        tools = "\n".join(
            f"{escape(name)} ({tier}){'' if name in approved.get(server.name, {}) else ' new'}"
            for name, tier in server.tools.items()
        )
        table.add_row(
            escape(server.name),
            escape(server.url or " ".join(server.command or ())),
            tools or "[dim]none listed[/]",
            _secrets_state(services, server),
        )
    services.console.print(table)


@mcp_app.command("approve")
def mcp_approve(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The server's name in the config.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask.")] = False,
) -> None:
    """Start a server, show its listed tools exactly as it describes them, and approve them."""
    services = services_of(ctx)
    server = _server(services, name)
    services.paths.ensure_all()
    try:
        tools = asyncio.run(
            _listing(Connection(server, services.mcp_opener(services.paths, services.secrets)))
        )
    except McpServerError as exc:
        fail(services, f"Could not start {name}: {exc}")
    approvals = Approvals.of(services.paths)
    before = approvals.load().get(name, {})
    sorted_out = review(server, tools, before)
    reviewed = [*(tool for tool, _ in sorted_out.usable), *sorted_out.unapproved]
    for tool in reviewed:
        state = (
            "approved before"
            if before.get(tool.name) == tool.fingerprint
            else ("changed since you approved it" if tool.name in before else "new")
        )
        _show(services, tool, server.tools[tool.name], state)
    for problem in sorted_out.problems:
        services.console.print(f"[yellow]Not usable:[/] {escape(problem)}")
    if sorted_out.missing:
        services.console.print(
            f"[yellow]Listed but not offered by {escape(name)}:[/] "
            + escape(", ".join(sorted_out.missing))
        )
    if sorted_out.unlisted:
        services.console.print(
            "[dim]Also offered (not used unless you list them under "
            f"{escape('[mcp.servers.tools]')}): {escape(', '.join(sorted_out.unlisted))}[/]"
        )
    if not reviewed:
        services.console.print("Nothing to approve: list the tools you want under the server.")
        return
    question = (
        f"Approve these {len(reviewed)} tool definition(s)? Their descriptions are read by "
        "Claude, so only approve what you trust"
    )
    if not yes and not typer.confirm(question, default=False):
        services.console.print("Nothing approved.")
        raise typer.Exit(code=1)
    approvals.approve(name, reviewed)
    services.console.print(
        f"[green]Approved {len(reviewed)} tool(s) of {escape(name)}.[/] They are used from the "
        "next start of Yarvis."
    )


@mcp_app.command("secret")
def mcp_secret(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The server's name in the config.")],
    variable: Annotated[str, typer.Argument(help="The secret's name, as in the config.")],
) -> None:
    """Store a secret for a server in the keychain (input is hidden)."""
    services = services_of(ctx)
    server = _server(services, name)
    if variable not in (*server.secrets, *((server.token,) if server.token else ())):
        fail(services, f"{variable} is not among the secrets or the token of {name} in the config.")
    value = typer.prompt(f"Paste {variable} for {name}", hide_input=True)
    try:
        services.secrets.set_extra(secret_key(name, variable), value)
    except (InvalidSecretError, RuntimeError) as exc:
        fail(services, f"Not saved: {exc}")
    services.console.print(f"Saved {variable} for {escape(name)} to the keychain.")


@mcp_app.command("forget")
def mcp_forget(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The server's name.")],
) -> None:
    """Withdraw every approval for a server (and delete its secrets from the keychain)."""
    services = services_of(ctx)
    Approvals.of(services.paths).forget(name)
    config = load_config_or_exit(services)
    for server in config.mcp.servers:
        if server.name == name:
            for variable in (*server.secrets, *((server.token,) if server.token else ())):
                services.secrets.delete_extra(secret_key(name, variable))
    services.console.print(f"Forgot {escape(name)}: its tools are no longer approved.")


async def _listing(connection: Connection) -> tuple[RemoteTool, ...]:
    try:
        return await connection.start()
    finally:
        await connection.stop()


def _server(services: Services, name: str) -> McpServerConfig:
    config = load_config_or_exit(services)
    found = next((server for server in config.mcp.servers if server.name == name), None)
    if found is None:
        fail(services, f"There is no MCP server called {name!r} in the config.")
    return found


def _secrets_state(services: Services, server: McpServerConfig) -> str:
    names = (*server.secrets, *((server.token,) if server.token else ()))
    if not names:
        return "-"
    stored = {name: services.secrets.get_extra(secret_key(server.name, name)) for name in names}
    return "\n".join(
        f"{name}: {'set' if value is not None else '[yellow]not set[/]'}"
        for name, value in stored.items()
    )


def _show(services: Services, tool: RemoteTool, tier: int, state: str) -> None:
    """A tool as the server describes it: the text Claude would read."""
    description = tool.description or "(no description)"
    if len(description) > _SHOWN_CHARS:
        description = description[:_SHOWN_CHARS] + f"… ({len(tool.description)} characters)"
    arguments: object = tool.input_schema.get("properties", {})
    keys = cast("dict[object, object]", arguments) if isinstance(arguments, dict) else {}
    names = ", ".join(str(key) for key in keys)
    services.console.print(
        f"\n[bold]{escape(tool.name)}[/] — tier {_TIERS[tier]} — [dim]{state}[/]\n"
        f"  {escape(description)}\n  [dim]arguments: {escape(names or 'none')}[/]",
        highlight=False,
    )


# --------------------------------------------------------------------------- skills


@skills_app.command("add")
def skills_add(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True, help="The skill's folder."),
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask.")] = False,
) -> None:
    """Add (or update) a skill: a folder with a SKILL.md (agentskills.io format)."""
    services = services_of(ctx)
    try:
        skill = read_skill(folder)
    except (SkillError, OSError, UnicodeDecodeError) as exc:
        fail(services, f"Not a usable skill: {exc}")
    services.console.print(
        f"[bold]{escape(skill.name)}[/]: {escape(skill.description)}\n"
        f"[dim]{len(skill.instructions)} characters of instructions, "
        f"{len(skill.files)} reference file(s)[/]",
        highlight=False,
    )
    if skill.has_scripts:
        services.console.print(
            "[yellow]Its scripts are not copied and never run:[/] Yarvis only reads a "
            "skill's instructions."
        )
    if not yes and not typer.confirm(
        "Add it? Claude will read its text when a request calls for it", default=False
    ):
        services.console.print("Not added.")
        raise typer.Exit(code=1)
    services.paths.ensure_all()
    try:
        SkillLibrary.of(services.paths).add(folder)
    except (SkillError, OSError, UnicodeDecodeError) as exc:
        fail(services, f"Not added: {exc}")
    services.console.print(f"[green]Added {escape(skill.name)}.[/] Used from the next start.")


@skills_app.command("list")
def skills_list(ctx: typer.Context) -> None:
    """Show the skills Yarvis uses (and any it leaves out)."""
    services = services_of(ctx)
    skills, problems = SkillLibrary.of(services.paths).skills()
    if not skills and not problems:
        services.console.print("No skills yet. Add one with: yarvis skills add <folder>")
        return
    for skill in skills:
        services.console.print(
            f"[bold]{escape(skill.name)}[/]: {escape(skill.description)}", highlight=False
        )
    for problem in problems:
        services.console.print(f"[yellow]Left out:[/] {escape(problem)}", highlight=False)


@skills_app.command("remove")
def skills_remove(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The skill's name.")],
) -> None:
    """Remove a skill."""
    services = services_of(ctx)
    if SkillLibrary.of(services.paths).remove(name):
        services.console.print(f"Removed {escape(name)}.")
    else:
        services.console.print(f"There is no skill called {escape(name)}.")
