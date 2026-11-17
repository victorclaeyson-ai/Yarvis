"""``yarvis mcp`` and ``yarvis skills``: adding things to Yarvis, and approving them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from test_mcp import in_process, notes_server
from test_skills import make_skill
from yarvis.config import Config
from yarvis.doctor import DoctorContext, Status, check_mcp, check_skills
from yarvis.security.secrets import KEYRING_SERVICE, SecretStore
from yarvis.tools.mcp import Approvals
from yarvis.tools.mcp_launch import secret_key
from yarvis.tools.skills import SkillLibrary

if TYPE_CHECKING:
    from pathlib import Path

    from fakes import FakeAudioProbe, FakeKeyring, Invoke
    from yarvis.cli.services import Services
    from yarvis.paths import AppPaths

NOTES = """
[[mcp.servers]]
name = "notes"
command = ["notes-server"]
secrets = ["NOTES_TOKEN"]
  [mcp.servers.tools]
  add_note = 1
  list_notes = 0
  gone = 0
"""


@pytest.fixture
def configured(services: Services, paths: AppPaths) -> Services:
    paths.ensure_all()
    paths.config_file.write_text(NOTES)
    services.mcp_opener = lambda _paths, _secrets: in_process(notes=notes_server())
    return services


def test_no_servers_yet(invoke: Invoke) -> None:
    result = invoke("mcp", "list")
    assert result.exit_code == 0
    assert "No MCP servers yet" in result.output


def test_listing_shows_tiers_approvals_and_secrets(
    invoke: Invoke, configured: Services, fake_keyring: FakeKeyring
) -> None:
    result = invoke("mcp", "list")
    assert result.exit_code == 0, result.output
    assert "add_note (1) new" in result.output
    assert "NOTES_TOKEN: not set" in result.output
    fake_keyring.store[(KEYRING_SERVICE, "extra:mcp:notes:NOTES_TOKEN")] = "x" * 20
    assert "NOTES_TOKEN: set" in invoke("mcp", "list").output


def test_approving_shows_what_claude_will_read(invoke: Invoke, configured: Services) -> None:
    result = invoke("mcp", "approve", "notes", stdin="y\n")
    assert result.exit_code == 0, result.output
    assert "add_note — tier 1 reversible — new" in result.output
    assert "Store a note." in result.output
    assert "arguments: text, tag" in result.output
    assert "Listed but not offered by notes: gone" in result.output
    assert "Also offered (not used unless you list them under [mcp.servers.tools]): broken" in (
        result.output
    )
    assert "Approved 2 tool(s) of notes" in result.output
    approved = Approvals.of(configured.paths).load()
    assert sorted(approved["notes"]) == ["add_note", "list_notes"]
    again = invoke("mcp", "approve", "notes", "--yes")
    assert "approved before" in again.output
    assert "add_note (1) new" not in invoke("mcp", "list").output


def test_a_changed_definition_is_pointed_out(invoke: Invoke, configured: Services) -> None:
    from yarvis.tools.mcp import RemoteTool  # noqa: PLC0415

    Approvals.of(configured.paths).approve("notes", [RemoteTool("add_note", "Old.", {})])
    result = invoke("mcp", "approve", "notes", stdin="n\n")
    assert "changed since you approved it" in result.output
    assert result.exit_code == 1
    assert "Nothing approved." in result.output


def test_approving_an_unknown_or_broken_server(invoke: Invoke, configured: Services) -> None:
    result = invoke("mcp", "approve", "files")
    assert result.exit_code == 1
    assert "There is no MCP server called 'files'" in result.output
    configured.mcp_opener = lambda _paths, _secrets: in_process()  # nothing runs
    broken = invoke("mcp", "approve", "notes")
    assert broken.exit_code == 1
    assert "Could not start notes" in broken.output


def test_nothing_to_approve(invoke: Invoke, configured: Services, paths: AppPaths) -> None:
    paths.config_file.write_text(NOTES.replace("add_note = 1", "x = 1").replace("list_notes", "y"))
    result = invoke("mcp", "approve", "notes")
    assert result.exit_code == 0
    assert "Nothing to approve" in result.output


def test_secrets_go_to_the_keychain(
    invoke: Invoke, configured: Services, fake_keyring: FakeKeyring
) -> None:
    result = invoke("mcp", "secret", "notes", "NOTES_TOKEN", stdin="abcdefghijklmnopqrst\n")
    assert result.exit_code == 0, result.output
    assert "abcdefghijklmnopqrst" not in result.output
    assert fake_keyring.store[(KEYRING_SERVICE, "extra:mcp:notes:NOTES_TOKEN")] == (
        "abcdefghijklmnopqrst"
    )
    assert invoke("mcp", "secret", "notes", "OTHER").exit_code == 1
    short = invoke("mcp", "secret", "notes", "NOTES_TOKEN", stdin="short\n")
    assert short.exit_code == 1
    assert "Not saved" in short.output


def test_forgetting_a_server(
    invoke: Invoke, configured: Services, fake_keyring: FakeKeyring
) -> None:
    invoke("mcp", "approve", "notes", "--yes")
    fake_keyring.store[(KEYRING_SERVICE, "extra:mcp:notes:NOTES_TOKEN")] = "x" * 20
    result = invoke("mcp", "forget", "notes")
    assert result.exit_code == 0
    assert Approvals.of(configured.paths).load() == {}
    assert (KEYRING_SERVICE, "extra:mcp:notes:NOTES_TOKEN") not in fake_keyring.store


# --------------------------------------------------------------------------- skills


def test_adding_listing_and_removing_skills(
    invoke: Invoke, services: Services, tmp_path: Path
) -> None:
    assert "No skills yet" in invoke("skills", "list").output
    source = make_skill(tmp_path / "source")
    declined = invoke("skills", "add", str(source), stdin="n\n")
    assert declined.exit_code == 1
    assert "Its scripts are not copied and never run" in declined.output
    added = invoke("skills", "add", str(source), stdin="y\n")
    assert added.exit_code == 0, added.output
    assert "Added morning-briefing" in added.output
    assert "morning-briefing: A spoken morning briefing" in invoke("skills", "list").output
    (SkillLibrary.of(services.paths).folder / "broken").mkdir()
    assert "Left out: broken" in invoke("skills", "list").output
    assert "Removed morning-briefing" in invoke("skills", "remove", "morning-briefing").output
    assert "no skill called" in invoke("skills", "remove", "morning-briefing").output


def test_an_invalid_skill_is_not_added(invoke: Invoke, tmp_path: Path) -> None:
    folder = tmp_path / "skill"
    folder.mkdir()
    result = invoke("skills", "add", str(folder), "--yes")
    assert result.exit_code == 1
    assert "Not a usable skill" in result.output


# --------------------------------------------------------------------------- doctor


@pytest.fixture
def ctx(paths: AppPaths, secrets: SecretStore, audio: FakeAudioProbe) -> DoctorContext:
    return DoctorContext(
        paths=paths, secrets=secrets, audio=audio, python_version=(3, 12, 4), system="Linux",
        machine="x86_64",
    )  # fmt: skip


def test_doctor_checks_mcp_servers(ctx: DoctorContext, fake_keyring: FakeKeyring) -> None:
    assert check_mcp(ctx, Config()).detail == "none configured"
    import tomllib  # noqa: PLC0415

    config = Config.model_validate(tomllib.loads(NOTES))
    missing = check_mcp(ctx, config)
    assert missing.status is Status.WARN
    assert missing.hint == "Run: yarvis mcp secret notes NOTES_TOKEN"
    fake_keyring.store[(KEYRING_SERVICE, "extra:" + secret_key("notes", "NOTES_TOKEN"))] = "x" * 20
    pending = check_mcp(ctx, config)
    assert pending.detail == "notes: not approved yet: add_note, list_notes, gone"
    assert pending.hint == "Run: yarvis mcp approve notes"
    from yarvis.tools.mcp import RemoteTool  # noqa: PLC0415

    tools = [RemoteTool(name, "", {}) for name in ("add_note", "list_notes", "gone")]
    Approvals.of(ctx.paths).approve("notes", tools)
    assert check_mcp(ctx, config).status is Status.OK


def test_doctor_checks_skills(ctx: DoctorContext, tmp_path: Path) -> None:
    assert check_skills(ctx, Config()).detail == "none"
    off = Config.model_validate({"tools": {"skills": False}})
    assert check_skills(ctx, off).detail == "off (tools.skills = false)"
    library = SkillLibrary.of(ctx.paths)
    library.add(make_skill(tmp_path / "source"))
    assert check_skills(ctx, Config()).detail == "morning-briefing"
    (library.folder / "morning-briefing" / "notes.txt").write_text("changed")
    assert check_skills(ctx, Config()).status is Status.WARN


# --------------------------------------------------------------------------- keychain


def test_extra_secrets_are_kept_apart(secrets: SecretStore, fake_keyring: FakeKeyring) -> None:
    assert secrets.get_extra("mcp:notes:TOKEN") is None
    secrets.set_extra("mcp:notes:TOKEN", "abcdefghijklmnopqrst")
    value = secrets.get_extra("mcp:notes:TOKEN")
    assert isinstance(value, SecretStr)
    assert value.get_secret_value() == "abcdefghijklmnopqrst"
    assert (KEYRING_SERVICE, "extra:mcp:notes:TOKEN") in fake_keyring.store
    assert secrets.delete_extra("mcp:notes:TOKEN")
    assert not secrets.delete_extra("mcp:notes:TOKEN")
    with pytest.raises(ValueError, match="not a valid secret key"):
        secrets.get_extra("anthropic")
    fake_keyring.available = False
    assert secrets.get_extra("mcp:notes:TOKEN") is None
    with pytest.raises(RuntimeError, match="keychain"):
        secrets.set_extra("mcp:notes:TOKEN", "abcdefghijklmnopqrst")
    fake_keyring.available = True
    fake_keyring.fail_reads = True
    assert secrets.get_extra("mcp:notes:TOKEN") is None


def test_the_cli_starts_real_servers_from_yarvis_folders(
    paths: AppPaths, secrets: SecretStore
) -> None:
    from yarvis.cli.services import mcp_launcher  # noqa: PLC0415
    from yarvis.tools.mcp_launch import Launcher  # noqa: PLC0415

    launcher = mcp_launcher(paths, secrets)
    assert isinstance(launcher, Launcher)
    assert launcher.work_dir == paths.data_dir / "mcp"
    assert launcher.log_dir == paths.log_dir


def test_approving_explains_unusable_and_wordy_tools(
    invoke: Invoke, configured: Services, paths: AppPaths
) -> None:
    from mcp.server.mcpserver import MCPServer  # noqa: PLC0415

    server = MCPServer("notes")

    @server.tool(description="word " * 200)
    def add_note(text: str) -> str:
        return text

    paths.config_file.write_text(
        '[[mcp.servers]]\nname = "notes"\ncommand = ["x"]\n[mcp.servers.tools]\nadd_note = 0\n'
    )
    configured.mcp_opener = lambda _paths, _secrets: in_process(notes=server)
    result = invoke("mcp", "approve", "notes", stdin="n\n")
    assert "(1000 characters)" in result.output
    assert "Secrets" not in result.output
    assert "-" in invoke("mcp", "list").output
