"""API keys and other credentials, stored in the OS keychain.

Resolution order for a secret:

1. The OS keychain (macOS Keychain, Secret Service on Linux) via ``keyring``.
2. An environment variable, for headless machines and CI.

Secrets are never read from the config file. Every resolved value is registered with the
redactor so it can never end up in a log line.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol

import keyring
import keyring.errors
from pydantic import SecretStr

from yarvis.security.redact import Redactor, default_redactor

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

KEYRING_SERVICE: Final = "yarvis"
MIN_SECRET_LENGTH: Final = 16


class SecretName(StrEnum):
    """Every credential Yarvis knows about. The value is the user-facing name."""

    ANTHROPIC = "anthropic"
    SPOTIFY = "spotify"
    """A Spotify developer app's ``client_id:client_secret`` (for searching music)."""
    SPOTIFY_LOGIN = "spotify-login"
    """The user's Spotify refresh token (``yarvis spotify login``: their own playlists)."""
    ASSEMBLYAI = "assemblyai"
    """An AssemblyAI API key: live speech recognition in the cloud (optional)."""

    @property
    def env_var(self) -> str:
        """Environment variable used as a fallback source."""
        return _ENV_VARS[self]


_ANTHROPIC_KEY: Final = re.compile(r"^sk-ant-[A-Za-z0-9_-]+$")
_ASSEMBLYAI_KEY: Final = re.compile(r"^[A-Za-z0-9_]{32,128}$")
_ENV_VARS: Final[dict[SecretName, str]] = {
    SecretName.ANTHROPIC: "ANTHROPIC_API_KEY",
    SecretName.SPOTIFY: "YARVIS_SPOTIFY",
    SecretName.SPOTIFY_LOGIN: "YARVIS_SPOTIFY_LOGIN",
    SecretName.ASSEMBLYAI: "ASSEMBLYAI_API_KEY",
}


class SecretSource(StrEnum):
    """Where a secret was found."""

    KEYCHAIN = "keychain"
    ENVIRONMENT = "environment"


@dataclass(frozen=True, slots=True)
class ResolvedSecret:
    """A secret value plus its origin. ``value`` is masked in ``repr`` and ``str``."""

    value: SecretStr
    source: SecretSource


class InvalidSecretError(ValueError):
    """Raised when a value is obviously not a valid credential."""


class KeyringBackend(Protocol):
    """The subset of the ``keyring`` API Yarvis uses (lets tests inject a fake)."""

    def get_password(self, service: str, username: str) -> str | None:
        """Return the stored secret or None."""
        ...

    def set_password(self, service: str, username: str, password: str) -> None:
        """Store a secret."""
        ...

    def delete_password(self, service: str, username: str) -> None:
        """Delete a secret; raise ``keyring.errors.PasswordDeleteError`` if absent."""
        ...

    def is_available(self) -> bool:
        """Return True if a real, secure keychain backend is usable."""
        ...

    def describe(self) -> str:
        """Human-readable backend name, for diagnostics."""
        ...


class SystemKeyring:
    """``KeyringBackend`` backed by the OS keychain through the ``keyring`` package."""

    def get_password(self, service: str, username: str) -> str | None:
        """Return the stored secret or None."""
        return keyring.get_password(service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        """Store a secret."""
        keyring.set_password(service, username, password)

    def delete_password(self, service: str, username: str) -> None:
        """Delete a secret."""
        keyring.delete_password(service, username)

    def is_available(self) -> bool:
        """Return False for the 'fail' and plaintext backends, which are not secure."""
        backend = keyring.get_keyring()
        module = type(backend).__module__
        return not module.startswith(("keyring.backends.fail", "keyrings.alt"))

    def describe(self) -> str:
        """Human-readable backend name."""
        backend = keyring.get_keyring()
        return f"{type(backend).__module__}.{type(backend).__qualname__}"


def validate_secret(value: str, name: SecretName | None = None) -> str:
    """Reject values that cannot be a real credential (empty, whitespace, too short).

    A paste can bring in invisible characters (a terminal's Ctrl+V can arrive as one), and
    a copy can catch the wrong text: both are caught here, before the key is stored and the
    first request fails with a vague "rejected".

    Returns:
        The value, stripped of surrounding whitespace.

    Raises:
        InvalidSecretError: If the value is clearly not a credential.
    """
    value = value.strip()
    if any(not char.isprintable() for char in value):
        msg = "the pasted text contains invisible characters; copy the key again"
        raise InvalidSecretError(msg)
    if len(value) < MIN_SECRET_LENGTH:
        msg = f"value is too short to be an API key (minimum {MIN_SECRET_LENGTH} characters)"
        raise InvalidSecretError(msg)
    if any(char.isspace() for char in value):
        msg = "API keys never contain whitespace"
        raise InvalidSecretError(msg)
    if name is SecretName.ANTHROPIC and not _ANTHROPIC_KEY.match(value):
        msg = "Anthropic API keys start with 'sk-ant-' (create one at console.anthropic.com)"
        raise InvalidSecretError(msg)
    if name is SecretName.ASSEMBLYAI and not _ASSEMBLYAI_KEY.match(value):
        msg = "AssemblyAI API keys are 32 letters and digits (copy yours from assemblyai.com)"
        raise InvalidSecretError(msg)
    return value


class SecretStore:
    """Read, write and delete Yarvis credentials."""

    def __init__(
        self,
        *,
        backend: KeyringBackend | None = None,
        environ: Mapping[str, str] | None = None,
        redactor: Redactor = default_redactor,
    ) -> None:
        self._backend: KeyringBackend = backend or SystemKeyring()
        self._environ = os.environ if environ is None else environ
        self._redactor = redactor

    @property
    def keychain_available(self) -> bool:
        """Whether a secure OS keychain can be used."""
        try:
            return self._backend.is_available()
        except keyring.errors.KeyringError:
            return False

    @property
    def keychain_description(self) -> str:
        """Name of the keychain backend in use."""
        return self._backend.describe()

    def get(self, name: SecretName) -> ResolvedSecret | None:
        """Resolve a secret from the keychain, falling back to the environment."""
        value: str | None = None
        source = SecretSource.KEYCHAIN
        if self.keychain_available:
            try:
                value = self._backend.get_password(KEYRING_SERVICE, name.value)
            except keyring.errors.KeyringError:
                logger.warning("Keychain lookup failed for %s; trying environment", name.value)
        if not value:
            value = self._environ.get(name.env_var)
            source = SecretSource.ENVIRONMENT
        if not value:
            return None
        self._redactor.register(value)
        return ResolvedSecret(value=SecretStr(value), source=source)

    def set(self, name: SecretName, value: str) -> None:
        """Validate and store a secret in the keychain.

        Raises:
            InvalidSecretError: If the value is not a plausible credential.
            RuntimeError: If no secure keychain is available.
        """
        value = validate_secret(value, name)
        if not self.keychain_available:
            msg = (
                "no secure OS keychain is available; "
                f"set the {name.env_var} environment variable instead"
            )
            raise RuntimeError(msg)
        self._redactor.register(value)
        self._backend.set_password(KEYRING_SERVICE, name.value, value)

    def delete(self, name: SecretName) -> bool:
        """Remove a secret from the keychain. Returns False if it was not stored."""
        return self._delete(name.value)

    def get_extra(self, key: str) -> SecretStr | None:
        """A secret stored under a free-form key, such as an MCP server's token.

        Keychain only: these have no environment-variable fallback.
        """
        if not self.keychain_available:
            return None
        try:
            value = self._backend.get_password(KEYRING_SERVICE, _extra(key))
        except keyring.errors.KeyringError:
            logger.warning("Keychain lookup failed for %s", key)
            return None
        if not value:
            return None
        self._redactor.register(value)
        return SecretStr(value)

    def set_extra(self, key: str, value: str) -> None:
        """Validate and store a secret under a free-form key (see ``get_extra``).

        Raises:
            InvalidSecretError: If the value is not a plausible credential.
            RuntimeError: If no secure keychain is available.
        """
        value = validate_secret(value)
        if not self.keychain_available:
            msg = "no secure OS keychain is available"
            raise RuntimeError(msg)
        self._redactor.register(value)
        self._backend.set_password(KEYRING_SERVICE, _extra(key), value)

    def delete_extra(self, key: str) -> bool:
        """Remove a secret stored with ``set_extra``. Returns False if it was not stored."""
        return self._delete(_extra(key))

    def _delete(self, username: str) -> bool:
        if not self.keychain_available:
            return False
        try:
            self._backend.delete_password(KEYRING_SERVICE, username)
        except keyring.errors.PasswordDeleteError:
            return False
        return True


_EXTRA_KEY: Final = re.compile(r"^[a-z][a-z0-9-]*(:[A-Za-z0-9_-]+)+$")


def _extra(key: str) -> str:
    """The keychain entry for a free-form key, kept apart from the ``SecretName`` ones."""
    if not _EXTRA_KEY.match(key):
        msg = f"not a valid secret key: {key!r}"
        raise ValueError(msg)
    return f"extra:{key}"
