from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from fakes import POSIX_ONLY, posix_mode, symlink_or_skip
from yarvis import paths as yarvis_paths
from yarvis.paths import (
    HOME_ENV_VAR,
    REPLACE_TRIES,
    AppPaths,
    ensure_private_dir,
    is_private,
    private_lock,
    replace_private_file,
    write_private_file,
)


def test_home_env_var_places_everything_under_one_root(tmp_path: Path) -> None:
    paths = AppPaths.default({HOME_ENV_VAR: str(tmp_path)})
    assert all(directory.parent == tmp_path for directory in paths.all_dirs())
    assert paths.config_file == tmp_path / "config" / "config.toml"


def test_platform_default_paths_mention_app_name() -> None:
    paths = AppPaths.default({})
    assert all("yarvis" in str(directory) for directory in paths.all_dirs())


def test_ensure_private_dir_tightens_existing_permissions(tmp_path: Path) -> None:
    directory = tmp_path / "open"
    directory.mkdir(mode=0o755)
    ensure_private_dir(directory)
    assert posix_mode(directory) in {0o700, None}
    assert is_private(directory)


@POSIX_ONLY
def test_is_private_detects_group_access(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_text("x")
    file.chmod(0o640)
    assert not is_private(file)


def test_write_private_file_creates_0600(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "secret.toml"
    write_private_file(target, "hello")
    assert target.read_text() == "hello"
    assert posix_mode(target) in {0o600, None}
    assert posix_mode(target.parent) in {0o700, None}


def test_write_private_file_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    target = tmp_path / "file"
    write_private_file(target, "one")
    with pytest.raises(FileExistsError):
        write_private_file(target, "two")
    write_private_file(target, "two", overwrite=True)
    assert target.read_text() == "two"


def test_write_private_file_does_not_follow_symlinks(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("original")
    link = tmp_path / "link"
    symlink_or_skip(link, victim)
    with pytest.raises(OSError, match=r"symbolic links|Too many levels"):
        write_private_file(link, "attacker", overwrite=True)
    assert victim.read_text() == "original"


def test_without_o_nofollow_symlinks_are_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(yarvis_paths, "_NO_FOLLOW", 0)  # as on Windows
    victim = tmp_path / "victim"
    victim.write_text("original")
    link = tmp_path / "link"
    symlink_or_skip(link, victim)
    with pytest.raises(OSError, match="symbolic links"):
        write_private_file(link, "attacker", overwrite=True)
    assert victim.read_text() == "original"


def test_on_windows_private_means_inside_the_user_profile(tmp_path: Path) -> None:
    inside = ensure_private_dir(tmp_path / "AppData" / "Local" / "yarvis")
    inside.chmod(0o777)  # Windows reports every folder like this
    assert is_private(inside, windows=True, home=tmp_path)
    assert not is_private(inside, windows=True, home=tmp_path / "other-user")
    assert not is_private(inside, windows=False)  # on a Mac or Linux, 0o777 is open to all


def test_ensure_all_creates_private_directories(tmp_path: Path) -> None:
    paths = AppPaths.under(tmp_path / "deep" / "home")
    paths.ensure_all()
    assert all(is_private(directory) for directory in paths.all_dirs())


# --------------------------------------------------------------------------- safe replace


def test_replace_private_file_writes_0600_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "data" / "memory.json"
    replace_private_file(target, "one")
    replace_private_file(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert posix_mode(target) in (None, 0o600)
    assert [p.name for p in target.parent.iterdir()] == ["memory.json"]


def test_replace_private_file_refuses_symbolic_links(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("original")
    link = tmp_path / "link"
    symlink_or_skip(link, victim)
    with pytest.raises(OSError, match="symbolic links"):
        replace_private_file(link, "attacker")
    assert victim.read_text() == "original"
    assert link.is_symlink()


def test_replace_private_file_retries_while_another_program_holds_the_file(
    tmp_path: Path,
) -> None:
    # On Windows, os.replace fails while another process has the target open.
    target = tmp_path / "memory.json"
    target.write_text("old")
    failures = [PermissionError("in use"), PermissionError("in use")]
    waits: list[float] = []

    def busy_replace(source: Path, destination: Path) -> None:
        if failures:
            raise failures.pop()
        source.replace(destination)

    replace_private_file(target, "new", replace=busy_replace, sleep=waits.append)
    assert target.read_text() == "new"
    assert len(waits) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["memory.json"]


def test_replace_private_file_gives_up_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "memory.json"
    target.write_text("old")

    def always_busy(source: Path, destination: Path) -> None:
        raise PermissionError("in use")

    waits: list[float] = []
    with pytest.raises(PermissionError):
        replace_private_file(target, "new", replace=always_busy, sleep=waits.append)
    assert target.read_text() == "old"
    assert len(waits) == REPLACE_TRIES - 1
    assert [p.name for p in tmp_path.iterdir()] == ["memory.json"]  # temporary file removed


# --------------------------------------------------------------------------- lock file


class FakeClock:
    """``time.monotonic`` and ``time.sleep`` that only move when slept."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def lock_is_free(lock: Path) -> bool:
    clock = FakeClock()
    try:
        with private_lock(lock, timeout_s=0.1, sleep=clock.sleep, monotonic=clock.monotonic):
            return True
    except TimeoutError:
        return False


def test_private_lock_is_held_inside_and_released_after(tmp_path: Path) -> None:
    lock = tmp_path / "memory.json.lock"
    with private_lock(lock):
        assert not lock_is_free(lock)  # a second holder (another process) has to wait
        assert posix_mode(lock) in (None, 0o600)
    assert lock_is_free(lock)


def test_private_lock_is_released_after_an_error(tmp_path: Path) -> None:
    lock = tmp_path / "memory.json.lock"
    with pytest.raises(ValueError, match="boom"), private_lock(lock):
        msg = "boom"
        raise ValueError(msg)
    assert lock_is_free(lock)


def test_a_lock_held_by_someone_else_times_out(tmp_path: Path) -> None:
    lock = tmp_path / "memory.json.lock"
    clock = FakeClock()
    with (
        private_lock(lock),  # another process is saving
        pytest.raises(TimeoutError),
        private_lock(lock, timeout_s=2, sleep=clock.sleep, monotonic=clock.monotonic),
    ):
        pytest.fail("must not get the lock")  # pragma: no cover
    assert clock.now >= 2  # it waited the whole time


def test_a_lock_file_left_by_a_crash_does_not_block(tmp_path: Path) -> None:
    lock = tmp_path / "memory.json.lock"
    lock.write_text("")  # the file stays; the operating system's lock died with its owner
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert lock_is_free(lock)


def test_private_lock_refuses_a_symbolic_link(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("original")
    lock = tmp_path / "memory.json.lock"
    symlink_or_skip(lock, victim)
    with pytest.raises(OSError, match="symbolic links"), private_lock(lock):
        pytest.fail("must not get the lock")  # pragma: no cover
    assert victim.read_text() == "original"


def test_replace_private_file_reports_the_real_error_if_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory.json"

    def busy(source: Path, destination: Path) -> None:
        raise PermissionError("in use")

    def unlink_fails(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("antivirus has it open")

    monkeypatch.setattr(Path, "unlink", unlink_fails)
    with pytest.raises(PermissionError, match="in use"):
        replace_private_file(target, "new", replace=busy, sleep=lambda _s: None)
