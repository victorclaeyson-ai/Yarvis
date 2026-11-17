from __future__ import annotations

from typing import TYPE_CHECKING

import keyring
import keyring.errors
import pytest

from fakes import ASSEMBLYAI_KEY, VALID_KEY
from yarvis.security.secrets import (
    KEYRING_SERVICE,
    InvalidSecretError,
    SecretName,
    SecretSource,
    SecretStore,
    SystemKeyring,
    validate_secret,
)

if TYPE_CHECKING:
    from fakes import FakeKeyring
    from yarvis.security.redact import Redactor


def test_keychain_value_wins_over_environment(
    secrets: SecretStore, fake_keyring: FakeKeyring, environ: dict[str, str]
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    environ["ANTHROPIC_API_KEY"] = "sk-ant-from-environment-xxxx"  # gitleaks:allow
    resolved = secrets.get(SecretName.ANTHROPIC)
    assert resolved is not None
    assert resolved.source is SecretSource.KEYCHAIN
    assert resolved.value.get_secret_value() == VALID_KEY


def test_environment_is_used_when_keychain_is_empty(
    secrets: SecretStore, environ: dict[str, str]
) -> None:
    environ["ANTHROPIC_API_KEY"] = VALID_KEY
    resolved = secrets.get(SecretName.ANTHROPIC)
    assert resolved is not None
    assert resolved.source is SecretSource.ENVIRONMENT


def test_environment_is_used_when_keychain_unavailable(
    secrets: SecretStore, fake_keyring: FakeKeyring, environ: dict[str, str]
) -> None:
    fake_keyring.available = False
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = (
        "sk-ant-should-not-be-read"  # gitleaks:allow
    )
    environ["ANTHROPIC_API_KEY"] = VALID_KEY
    resolved = secrets.get(SecretName.ANTHROPIC)
    assert resolved is not None
    assert resolved.value.get_secret_value() == VALID_KEY


def test_keychain_errors_fall_back_to_environment(
    secrets: SecretStore,
    fake_keyring: FakeKeyring,
    environ: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_keyring.fail_reads = True
    environ["ANTHROPIC_API_KEY"] = VALID_KEY
    resolved = secrets.get(SecretName.ANTHROPIC)
    assert resolved is not None
    assert resolved.source is SecretSource.ENVIRONMENT
    assert "Keychain lookup failed" in caplog.text


def test_missing_secret_returns_none(secrets: SecretStore) -> None:
    assert secrets.get(SecretName.ANTHROPIC) is None


def test_resolved_secrets_are_registered_for_redaction(
    secrets: SecretStore, environ: dict[str, str], redactor: Redactor
) -> None:
    environ["ANTHROPIC_API_KEY"] = "plain-looking-value-123"  # gitleaks:allow
    secrets.get(SecretName.ANTHROPIC)
    assert redactor.contains_secret("oops plain-looking-value-123")


def test_secret_value_is_masked_in_repr(secrets: SecretStore, environ: dict[str, str]) -> None:
    environ["ANTHROPIC_API_KEY"] = VALID_KEY
    resolved = secrets.get(SecretName.ANTHROPIC)
    assert VALID_KEY not in repr(resolved)
    assert VALID_KEY not in str(resolved)


def test_set_stores_stripped_value(secrets: SecretStore, fake_keyring: FakeKeyring) -> None:
    secrets.set(SecretName.ANTHROPIC, f"  {VALID_KEY}\n")
    assert fake_keyring.store[(KEYRING_SERVICE, "anthropic")] == VALID_KEY


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("", "too short"),
        ("short", "too short"),
        ("sk-ant-has a space-in-it-123", "whitespace"),  # gitleaks:allow
        ("\x16", "invisible characters"),  # a Ctrl+V the terminal did not turn into a paste
        ("sk-ant-0123456789abcdef\x00", "invisible characters"),  # gitleaks:allow
        ("my-anthropic-key-0123456789", "start with 'sk-ant-'"),
        ("sk-ant-0123456789abcdef!", "start with 'sk-ant-'"),  # gitleaks:allow
    ],
)
def test_set_rejects_implausible_values(secrets: SecretStore, value: str, reason: str) -> None:
    with pytest.raises(InvalidSecretError, match=reason):
        secrets.set(SecretName.ANTHROPIC, value)


def test_only_anthropic_keys_need_their_prefix(secrets: SecretStore) -> None:
    secrets.set(SecretName.SPOTIFY, "0123456789abcdef0123456789abcdef:" + "f" * 32)


def test_set_refuses_without_secure_keychain(
    secrets: SecretStore, fake_keyring: FakeKeyring
) -> None:
    fake_keyring.available = False
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        secrets.set(SecretName.ANTHROPIC, VALID_KEY)
    assert fake_keyring.store == {}


def test_delete(secrets: SecretStore, fake_keyring: FakeKeyring) -> None:
    assert not secrets.delete(SecretName.ANTHROPIC)
    secrets.set(SecretName.ANTHROPIC, VALID_KEY)
    assert secrets.delete(SecretName.ANTHROPIC)
    fake_keyring.available = False
    assert not secrets.delete(SecretName.ANTHROPIC)


def test_validate_secret_returns_clean_value() -> None:
    assert validate_secret(f" {VALID_KEY} ") == VALID_KEY


def test_assemblyai_keys_are_letters_and_digits() -> None:
    key = ASSEMBLYAI_KEY
    assert validate_secret(key, SecretName.ASSEMBLYAI) == key
    with pytest.raises(InvalidSecretError, match="AssemblyAI"):
        validate_secret("sk-ant-api03-this-is-the-wrong-key", SecretName.ASSEMBLYAI)
    with pytest.raises(InvalidSecretError, match="AssemblyAI"):
        validate_secret(ASSEMBLYAI_KEY[:-1] + "!", SecretName.ASSEMBLYAI)
    assert SecretName.ASSEMBLYAI.env_var == "ASSEMBLYAI_API_KEY"


def test_keychain_availability_errors_are_treated_as_unavailable() -> None:
    class Exploding:
        def is_available(self) -> bool:
            raise keyring.errors.KeyringError("dbus not running")

    store = SecretStore(backend=Exploding(), environ={})  # pyright: ignore[reportArgumentType]
    assert not store.keychain_available


class _FailBackend:
    pass


_FailBackend.__module__ = "keyring.backends.fail"


class _MacBackend:
    pass


_MacBackend.__module__ = "keyring.backends.macOS"


def test_system_keyring_rejects_fail_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keyring, "get_keyring", _FailBackend)
    assert not SystemKeyring().is_available()


def test_system_keyring_accepts_real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keyring, "get_keyring", _MacBackend)
    system = SystemKeyring()
    assert system.is_available()
    assert system.describe() == "keyring.backends.macOS._MacBackend"


def test_system_keyring_delegates_to_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def get_password(service: str, username: str) -> str:
        calls.append(("get", service, username))
        return "value"

    def set_password(service: str, username: str, password: str) -> None:
        calls.append(("set", service, username, password))

    def delete_password(service: str, username: str) -> None:
        calls.append(("delete", service, username))

    monkeypatch.setattr(keyring, "get_password", get_password)
    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "delete_password", delete_password)
    system = SystemKeyring()
    assert system.get_password("svc", "user") == "value"
    system.set_password("svc", "user", "pw")
    system.delete_password("svc", "user")
    assert calls == [
        ("get", "svc", "user"),
        ("set", "svc", "user", "pw"),
        ("delete", "svc", "user"),
    ]
