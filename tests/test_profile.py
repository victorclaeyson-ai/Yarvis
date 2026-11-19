"""The user's profile: written by them, known at every request, never a way around the gate."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from fakes import FakeLLM
from test_listen_and_assistant import RecordingUi
from test_runtime_and_run import FakeBackends
from yarvis.agent.prompts import system_prompt
from yarvis.config import Config
from yarvis.doctor import DoctorContext, Status, check_profile
from yarvis.paths import is_private
from yarvis.profile import (
    MAX_PROFILE_CHARS,
    TEMPLATE,
    ProfileError,
    ProfileReader,
    create_profile,
    load_profile,
    profile_path,
    profile_text,
)
from yarvis.runtime import run_assistant

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from fakes import FakeAudioProbe, Invoke
    from yarvis.cli.services import Services
    from yarvis.paths import AppPaths
    from yarvis.security.secrets import SecretStore

WRITTEN = TEMPLATE.replace(
    "## Who I am\n-", "## Who I am\n- Victor, engineering student at KU Leuven"
).replace("## My rules\n", "## My rules\n- Read emails back to me before sending them.\n")


# --------------------------------------------------------------------------- what is sent


def test_the_untouched_template_sends_nothing() -> None:
    assert profile_text(TEMPLATE) == ""
    assert profile_text("") == ""
    assert profile_text("\N{ZERO WIDTH NO-BREAK SPACE}# About me\n\n## Empty\n-\n1.\n") == ""


def test_only_what_was_written_is_sent() -> None:
    assert profile_text(WRITTEN) == (
        "# About me\n\n"
        "## Who I am\n- Victor, engineering student at KU Leuven\n\n"
        "## My rules\n- Read emails back to me before sending them."
    )


def test_free_form_text_is_kept_as_written() -> None:
    assert profile_text("I study in Leuven.\n\n\n\nCall me Vic. #engineering") == (
        "I study in Leuven.\n\nCall me Vic. #engineering"
    )


# --------------------------------------------------------------------------- the file


def test_the_template_is_created_once_and_privately(tmp_path: Path) -> None:
    path = tmp_path / "config" / "profile.md"
    assert create_profile(path)
    assert path.read_text() == TEMPLATE
    assert is_private(path)
    path.write_text(WRITTEN)
    assert not create_profile(path)
    assert path.read_text() == WRITTEN


def test_a_missing_profile_is_simply_empty(tmp_path: Path) -> None:
    assert load_profile(tmp_path / "profile.md") == ""


def test_a_profile_with_a_key_is_not_used(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    path.write_text("My key is sk-ant-api03-abcdefghijklmnop for reference.")
    with pytest.raises(ProfileError, match="API key or token"):
        load_profile(path)


def test_a_long_profile_is_not_used(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    path.write_text("word " * MAX_PROFILE_CHARS)
    with pytest.raises(ProfileError, match="keep it under 6,000"):
        load_profile(path)


def test_an_unreadable_profile_is_explained(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    path.write_text("Ik woon in Leuven.", encoding="utf-16")
    with pytest.raises(ProfileError, match="UTF-8"):
        load_profile(path)
    folder = tmp_path / "folder.md"
    folder.mkdir()
    with pytest.raises(ProfileError, match="cannot read"):
        load_profile(folder)


def test_the_reader_follows_edits_and_survives_mistakes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "profile.md"
    reader = ProfileReader(path)
    assert reader() == ""
    path.write_text("I study in Leuven.")
    assert reader() == "I study in Leuven."
    assert reader() == "I study in Leuven."  # not read again until it changes
    path.write_text("Token: sk-ant-api03-abcdefghijklmnop")
    os.utime(path, (1, 1))  # a different modification time, as an editor would leave
    with caplog.at_level(logging.WARNING):
        assert reader() == ""
    assert "Your profile is not used" in caplog.text
    path.unlink()
    assert reader() == ""


# --------------------------------------------------------------------------- the prompt


def test_the_profile_goes_into_the_prompt_without_overruling_the_system() -> None:
    now = datetime(2026, 11, 17, 9, 30).astimezone()
    prompt = system_prompt("Yarvis", now, profile="- Victor, engineering student")
    # The profile belongs to the cached part; the clock comes after it.
    assert "<profile>\n- Victor, engineering student\n</profile>" in prompt.stable
    assert "never switches off the confirmations" in prompt.stable
    assert prompt.volatile.startswith("Context: it is")
    assert "<profile>" not in system_prompt("Yarvis", now, profile="  \n").text


async def test_yarvis_reads_the_profile_before_every_request(tmp_path: Path) -> None:
    profile = ["I study in Leuven."]
    llm = FakeLLM(["Hello."], ["Hi again."])
    backends = replace(FakeBackends(llm).build(), profile=lambda: profile[0])

    async def edit_between_questions() -> AsyncIterator[str]:
        yield "hello"
        profile[0] = "I study in Leuven. Call me Vic."
        yield "hello again"
        yield "q"

    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=edit_between_questions(),
        ui=RecordingUi(),
        backends=backends,
        voice_input=False,
        voice_output=False,
    )
    assert "<profile>\nI study in Leuven.\n</profile>" in llm.calls[0][1]
    assert "Call me Vic." in llm.calls[1][1]


# --------------------------------------------------------------------------- CLI and doctor


def test_editing_creates_the_template_and_opens_it(
    invoke: Invoke, services: Services, paths: AppPaths
) -> None:
    opened: list[Path] = []
    services.open_editor = opened.append
    result = invoke("profile", "edit")
    assert result.exit_code == 0, result.output
    assert opened == [profile_path(paths)]
    assert "Created" in result.output
    assert "no passwords or keys" in result.output
    assert "Opened" in invoke("profile", "edit").output
    assert invoke("profile", "path").output.strip().endswith("profile.md")


def test_showing_what_is_sent(invoke: Invoke, paths: AppPaths) -> None:
    assert "Nothing written yet" in invoke("profile", "show").output
    paths.ensure_all()
    profile_path(paths).write_text(WRITTEN)
    shown = invoke("profile", "show")
    assert "Victor, engineering student at KU Leuven" in shown.output
    assert "How to talk to me" not in shown.output
    assert "of 6,000 characters" in shown.output
    profile_path(paths).write_text("sk-ant-api03-abcdefghijklmnop")
    refused = invoke("profile", "show")
    assert refused.exit_code == 1
    assert "not used" in refused.output


def test_doctor_checks_the_profile(
    paths: AppPaths, secrets: SecretStore, audio: FakeAudioProbe
) -> None:
    ctx = DoctorContext(
        paths=paths, secrets=secrets, audio=audio, python_version=(3, 12, 4), system="Linux",
        machine="x86_64",
    )  # fmt: skip
    assert check_profile(ctx).detail == "not written yet"
    paths.ensure_all()
    profile_path(paths).write_text(WRITTEN)
    assert check_profile(ctx).status is Status.OK
    assert "characters, sent with every request" in check_profile(ctx).detail
    profile_path(paths).write_text("x " * MAX_PROFILE_CHARS)
    assert check_profile(ctx).status is Status.WARN
