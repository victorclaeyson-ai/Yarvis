from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fakes import FakeAudioProbe, FakeKeyring
from yarvis.paths import AppPaths
from yarvis.security.redact import Redactor
from yarvis.security.secrets import SecretStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    return AppPaths.under(tmp_path / "yarvis-home")


@pytest.fixture
def redactor() -> Redactor:
    return Redactor()


@pytest.fixture
def fake_keyring() -> FakeKeyring:
    return FakeKeyring()


@pytest.fixture
def environ() -> dict[str, str]:
    return {}


@pytest.fixture
def secrets(fake_keyring: FakeKeyring, environ: dict[str, str], redactor: Redactor) -> SecretStore:
    return SecretStore(backend=fake_keyring, environ=environ, redactor=redactor)


@pytest.fixture
def audio() -> FakeAudioProbe:
    return FakeAudioProbe()
