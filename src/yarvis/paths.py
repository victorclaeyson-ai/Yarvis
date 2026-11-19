"""Filesystem locations used by Yarvis, created with private permissions.

Everything Yarvis stores (config, memory, logs) lives in per-user directories that are
readable only by the current user. ``YARVIS_HOME`` overrides all locations at once, which
is useful for portable installs and for tests.
"""

from __future__ import annotations

import logging
import os
import secrets
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from platformdirs import PlatformDirs

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

logger = logging.getLogger(__name__)

APP_NAME = "yarvis"
HOME_ENV_VAR = "YARVIS_HOME"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
WINDOWS = sys.platform == "win32"
_NO_FOLLOW: int = getattr(os, "O_NOFOLLOW", 0)
"""``O_NOFOLLOW`` where it exists (not on Windows, where creating a link needs admin rights)."""

REPLACE_TRIES: Final = 5
REPLACE_DELAY_S: Final = 0.05
"""On Windows a file cannot be replaced while another program has it open: try again."""
LOCK_TIMEOUT_S: Final = 2.0
"""How long to wait for another process to finish a save (it takes milliseconds)."""
_LOCK_POLL_S: Final = 0.05


@dataclass(frozen=True, slots=True)
class AppPaths:
    """The set of directories Yarvis reads from and writes to."""

    config_dir: Path
    data_dir: Path
    log_dir: Path
    cache_dir: Path

    @property
    def config_file(self) -> Path:
        """Path of the main TOML configuration file."""
        return self.config_dir / "config.toml"

    @classmethod
    def default(cls, environ: Mapping[str, str] | None = None) -> AppPaths:
        """Resolve platform-appropriate paths, honouring ``YARVIS_HOME`` if set."""
        env = os.environ if environ is None else environ
        if home := env.get(HOME_ENV_VAR):
            return cls.under(Path(home).expanduser())
        dirs = PlatformDirs(APP_NAME, appauthor=False)
        return cls(
            config_dir=dirs.user_config_path,
            data_dir=dirs.user_data_path,
            log_dir=dirs.user_log_path,
            cache_dir=dirs.user_cache_path,
        )

    @classmethod
    def under(cls, root: Path) -> AppPaths:
        """Place every directory under a single root (portable mode)."""
        return cls(
            config_dir=root / "config",
            data_dir=root / "data",
            log_dir=root / "logs",
            cache_dir=root / "cache",
        )

    def all_dirs(self) -> tuple[Path, ...]:
        """Every directory, in a stable order."""
        return (self.config_dir, self.data_dir, self.log_dir, self.cache_dir)

    def ensure_all(self) -> None:
        """Create every directory with private permissions.

        Call this before writing anything below these directories: ``Path.mkdir`` creates
        missing *parents* with default (often world-readable) permissions.
        """
        for directory in self.all_dirs():
            ensure_private_dir(directory)


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and restrict it to the current user."""
    path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    path.chmod(PRIVATE_DIR_MODE)
    return path


def is_private(path: Path, *, windows: bool = WINDOWS, home: Path | None = None) -> bool:
    """Return True if only the current user can access ``path``.

    On macOS and Linux: neither group nor others have any permission bits. Windows has no
    such bits (Python reports every folder as open to all); there, Windows' own default
    permissions keep everything inside the user's profile folder (``home``) to that user,
    the system and administrators, so a path inside it counts as private.
    """
    if windows:
        return path.resolve().is_relative_to((home or Path.home()).resolve())
    mode = path.stat().st_mode
    return stat.S_IMODE(mode) & 0o077 == 0


def write_private_file(path: Path, content: str, *, overwrite: bool = False) -> None:
    """Write ``content`` to ``path`` with mode 0600, never following an existing symlink.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    ensure_private_dir(path.parent)
    if not _NO_FOLLOW and path.is_symlink():
        msg = f"refusing to write through symbolic links: {path}"
        raise OSError(msg)
    flags = os.O_WRONLY | os.O_CREAT | _NO_FOLLOW
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    fd = os.open(path, flags, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(PRIVATE_FILE_MODE)


def replace_private_file(
    path: Path,
    content: str,
    *,
    replace: Callable[[Path, Path], None] = os.replace,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Replace ``path`` with ``content`` all at once (mode 0600, never through a symlink).

    The content goes to a temporary file next to it first, so a crash halfway leaves the
    old file whole. On Windows, replacing fails while another program has the file open
    (``PermissionError``): it is tried ``REPLACE_TRIES`` times, ``REPLACE_DELAY_S`` apart.

    Raises:
        OSError: If ``path`` is a symbolic link, or it still cannot be replaced.
    """
    ensure_private_dir(path.parent)
    if path.is_symlink():
        msg = f"refusing to write through symbolic links: {path}"
        raise OSError(msg)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NO_FOLLOW, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(PRIVATE_FILE_MODE)
        for attempt in range(1, REPLACE_TRIES + 1):
            try:
                replace(temporary, path)
            except PermissionError:
                if attempt == REPLACE_TRIES:
                    raise
                sleep(REPLACE_DELAY_S)
            else:
                return
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:  # never hide the real error behind this one
            logger.warning("Could not remove %s: %s", temporary.name, type(exc).__name__)


@contextmanager
def private_lock(
    path: Path,
    *,
    timeout_s: float = LOCK_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Generator[None]:
    """Hold an exclusive lock on ``path`` for a read-modify-write two processes may race.

    The lock is the operating system's, held on the open file (``flock`` on macOS and
    Linux, ``msvcrt.locking`` on Windows): it is released when the process ends, even in a
    crash, so a lock is never left behind and nothing has to guess whether one is stale.
    The file itself stays, empty and private (mode 0600, never opened through a symlink).

    Raises:
        TimeoutError: If another process still holds it after ``timeout_s``.
        OSError: If ``path`` is a symbolic link or cannot be opened.
    """
    ensure_private_dir(path.parent)
    if path.is_symlink():
        msg = f"refusing to lock through symbolic links: {path}"
        raise OSError(msg)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | _NO_FOLLOW, PRIVATE_FILE_MODE)
    try:
        deadline = monotonic() + timeout_s
        while not _try_lock(fd):
            if monotonic() >= deadline:
                msg = f"{path.name} is held by another process"
                raise TimeoutError(msg)
            sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _try_lock(fd: int) -> bool:
    """Take the lock on ``fd`` if it is free; False if another process holds it."""
    if sys.platform == "win32":
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
