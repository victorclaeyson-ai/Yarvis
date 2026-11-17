"""Filesystem locations used by Yarvis, created with private permissions.

Everything Yarvis stores (config, memory, logs) lives in per-user directories that are
readable only by the current user. ``YARVIS_HOME`` overrides all locations at once, which
is useful for portable installs and for tests.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import PlatformDirs

if TYPE_CHECKING:
    from collections.abc import Mapping

APP_NAME = "yarvis"
HOME_ENV_VAR = "YARVIS_HOME"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
WINDOWS = sys.platform == "win32"
_NO_FOLLOW: int = getattr(os, "O_NOFOLLOW", 0)
"""``O_NOFOLLOW`` where it exists (not on Windows, where creating a link needs admin rights)."""


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
