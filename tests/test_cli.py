from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fakes import ASSEMBLYAI_KEY, VALID_KEY, FakeKeyring, Invoke
from yarvis import __version__
from yarvis.cli.app import Services
from yarvis.security.secrets import KEYRING_SERVICE

if TYPE_CHECKING:
    from yarvis.paths import AppPaths


def test_version(invoke: Invoke) -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help(invoke: Invoke) -> None:
    assert "Usage" in invoke().output


def test_doctor_fails_without_api_key(invoke: Invoke) -> None:
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "yarvis key set anthropic" in result.output


def test_doctor_passes_when_ready(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "All good" in result.output


def test_config_path(invoke: Invoke, paths: AppPaths) -> None:
    assert invoke("config", "path").output.strip() == str(paths.config_file)


def test_config_init_and_force(invoke: Invoke, paths: AppPaths) -> None:
    assert invoke("config", "init").exit_code == 0
    assert paths.config_file.exists()
    again = invoke("config", "init")
    assert again.exit_code == 1
    assert "already exists" in again.output
    assert invoke("config", "init", "--force").exit_code == 0


def test_config_show(invoke: Invoke) -> None:
    result = invoke("config", "show")
    assert result.exit_code == 0
    assert json.loads(result.output)["security"]["allow_tier3"] is False


def test_config_show_reports_errors(invoke: Invoke, paths: AppPaths) -> None:
    invoke("config", "init")
    paths.config_file.write_text("[nonsense]\n")
    result = invoke("config", "show")
    assert result.exit_code == 1
    assert "nonsense" in result.output


def test_key_set_stores_hidden_input(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    result = invoke("key", "set", "anthropic", stdin=f"{VALID_KEY}\n")
    assert result.exit_code == 0
    assert VALID_KEY not in result.output
    assert fake_keyring.store[(KEYRING_SERVICE, "anthropic")] == VALID_KEY


def test_key_set_assemblyai_explains_what_is_sent(
    invoke: Invoke, fake_keyring: FakeKeyring
) -> None:
    key = ASSEMBLYAI_KEY
    result = invoke("key", "set", "assemblyai", stdin=f"{key}\n")
    assert result.exit_code == 0, result.output
    assert "https://www.assemblyai.com/dashboard/signup" in result.output
    assert "never the wake word" in result.output
    assert key not in result.output
    assert fake_keyring.store[(KEYRING_SERVICE, "assemblyai")] == key


def test_key_set_rejects_garbage(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    result = invoke("key", "set", "anthropic", stdin="oops\n")
    assert result.exit_code == 1
    assert "Not saved" in result.output
    assert fake_keyring.store == {}


def test_key_status_never_prints_values(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    assert "anthropic: not set" in invoke("key", "status").output
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    output = invoke("key", "status").output
    assert "anthropic: set (keychain)" in output
    assert VALID_KEY not in output


def test_key_delete(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    assert "No anthropic key" in invoke("key", "delete", "anthropic").output
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    assert "Deleted" in invoke("key", "delete", "anthropic").output
    assert fake_keyring.store == {}


def test_default_services_are_real_implementations() -> None:
    services = Services.default()
    assert services.paths.config_file.name == "config.toml"


def test_error_text_is_not_interpreted_as_markup(invoke: Invoke, paths: AppPaths) -> None:
    invoke("config", "init")
    paths.config_file.write_text("[voice]\nvocabulary = ['[bold]', '']\n")
    result = invoke("config", "show")
    assert result.exit_code == 1
    assert "vocabulary" in result.output


SPOTIFY_ID = "0123456789abcdef0123456789abcdef"
SPOTIFY_SECRET = "fedcba9876543210fedcba9876543210"  # noqa: S105  # gitleaks:allow - test value


def test_key_set_spotify_asks_for_id_and_secret(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    result = invoke("key", "set", "spotify", stdin=f"{SPOTIFY_ID}\n{SPOTIFY_SECRET}\n")
    assert result.exit_code == 0, result.output
    assert "Create an app at" in result.output  # the setup instructions are shown
    assert SPOTIFY_SECRET not in result.output
    stored = fake_keyring.store[(KEYRING_SERVICE, "spotify")]
    assert stored == f"{SPOTIFY_ID}:{SPOTIFY_SECRET}"
    assert "spotify: set (keychain)" in invoke("key", "status").output


def test_key_set_spotify_rejects_garbage(invoke: Invoke, fake_keyring: FakeKeyring) -> None:
    result = invoke("key", "set", "spotify", stdin="not-an-id\nnope\n")
    assert result.exit_code == 1
    assert "32 characters" in result.output
    assert fake_keyring.store == {}
