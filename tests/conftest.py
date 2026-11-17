from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from fakes import FakeAudioProbe, FakeKeyring, Invoke
from yarvis.cli.app import Services, app
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


@pytest.fixture
def services(paths: AppPaths, secrets: SecretStore, audio: FakeAudioProbe) -> Services:
    return Services(
        paths=paths,
        secrets=secrets,
        audio=audio,
        console=Console(width=200, color_system=None),
        err_console=Console(width=200, color_system=None, stderr=True),
        parakeet_ok=lambda: False,  # the same on every CI machine; tests opt in
        parakeet_build=lambda: "mlx",  # like an Apple Silicon Mac; tests opt in to ONNX
    )


@pytest.fixture
def invoke(services: Services) -> Invoke:
    runner = CliRunner()

    def run(*args: str, stdin: str | None = None) -> Result:
        return runner.invoke(app, list(args), input=stdin, obj=services)

    return run
