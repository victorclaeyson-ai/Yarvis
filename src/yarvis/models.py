"""Speech models: where they come from and how they are verified.

Every model is pinned:

* Files Yarvis downloads itself (Kokoro) are pinned by URL, size and SHA-256. A file is
  only moved into place after its hash matches, and redirects to non-HTTPS URLs are
  refused.
* Whisper models come from Hugging Face through ``faster-whisper``, pinned to an exact
  git commit, so a changed upstream repository cannot change what we load.
* The Silero VAD model ships inside the ``faster-whisper`` wheel, which is itself pinned by
  hash in ``uv.lock``.

Only ONNX and CTranslate2 formats are used: never pickle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import httpx2

from yarvis.paths import PRIVATE_FILE_MODE, ensure_private_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    from yarvis.config.models import SttModel

CHUNK_SIZE: Final = 1 << 20  # 1 MiB


@dataclass(frozen=True, slots=True)
class ModelFile:
    """A file downloaded from a fixed URL and verified by size and SHA-256."""

    name: str
    url: str
    sha256: str
    size: int


_KOKORO_RELEASE: Final = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)

KOKORO_MODEL: Final = ModelFile(
    name="kokoro-v1.0.onnx",
    url=f"{_KOKORO_RELEASE}/kokoro-v1.0.onnx",
    sha256="beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
    size=325_505_369,
)
KOKORO_VOICES: Final = ModelFile(
    name="voices-v1.0.bin",
    url=f"{_KOKORO_RELEASE}/voices-v1.0.bin",
    sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    size=28_214_398,
)
KOKORO_FILES: Final = (KOKORO_MODEL, KOKORO_VOICES)

WHISPER_REVISIONS: Final[dict[SttModel, str]] = {
    "tiny.en": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
    "base.en": "3d3d5dee26484f91867d81cb899cfcf72b96be6c",
    "small.en": "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
}
"""Pinned commits of the ``Systran/faster-whisper-*`` repositories on Hugging Face."""

WHISPER_APPROX_MB: Final[dict[SttModel, int]] = {"tiny.en": 75, "base.en": 145, "small.en": 485}


class ModelIntegrityError(Exception):
    """A downloaded file did not match its pinned size or hash."""


class Download(Protocol):
    """A streamed HTTP download."""

    def iter_bytes(self) -> Iterator[bytes]:
        """Yield the body in chunks."""
        ...


def kokoro_dir(cache_dir: Path) -> Path:
    """Directory holding the Kokoro files."""
    return cache_dir / "models" / "kokoro"


def whisper_dir(cache_dir: Path) -> Path:
    """Directory holding Whisper models (Hugging Face cache layout)."""
    return cache_dir / "models" / "whisper"


def whisper_present(cache_dir: Path, model: SttModel) -> bool:
    """Whether the pinned revision of a Whisper model is already in the local cache."""
    snapshot = (
        whisper_dir(cache_dir)
        / f"models--Systran--faster-whisper-{model}"
        / "snapshots"
        / WHISPER_REVISIONS[model]
    )
    return (snapshot / "model.bin").exists()


def is_present(model: ModelFile, directory: Path) -> bool:
    """Cheap check: the file exists with the expected size (full hash on download)."""
    path = directory / model.name
    return path.is_file() and path.stat().st_size == model.size


def missing(models: tuple[ModelFile, ...], directory: Path) -> list[ModelFile]:
    """The models that still need downloading."""
    return [model for model in models if not is_present(model, directory)]


def download(
    model: ModelFile,
    directory: Path,
    *,
    opener: Callable[[str], AbstractContextManager[Download]] | None = None,
    progress: Callable[[int], None] | None = None,
) -> Path:
    """Download ``model`` into ``directory`` and verify it.

    The data is written to a temporary file and only renamed into place after the size
    and SHA-256 match. Partial or tampered downloads never become visible.

    Raises:
        ModelIntegrityError: If the size or hash does not match.
        httpx2.HTTPError: On network failures.
    """
    ensure_private_dir(directory)
    target = directory / model.name
    partial = directory / f".{model.name}.part"
    digest = hashlib.sha256()
    received = 0
    open_url = opener or _https_stream
    try:
        with open_url(model.url) as response, partial.open("wb") as handle:
            partial.chmod(PRIVATE_FILE_MODE)
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > model.size:
                    msg = f"{model.name} is larger than expected; refusing to continue"
                    raise ModelIntegrityError(msg)
                digest.update(chunk)
                handle.write(chunk)
                if progress is not None:
                    progress(len(chunk))
        if received != model.size:
            msg = f"{model.name}: expected {model.size} bytes, got {received}"
            raise ModelIntegrityError(msg)
        if digest.hexdigest() != model.sha256:
            msg = f"{model.name}: SHA-256 mismatch, the download may have been tampered with"
            raise ModelIntegrityError(msg)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


class _HttpsStream:
    """Context manager streaming an HTTPS URL, refusing any non-HTTPS redirect."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client = httpx2.Client(
            follow_redirects=True,
            timeout=httpx2.Timeout(30.0, read=60.0),
            event_hooks={"request": [_require_https]},
        )
        self._response: httpx2.Response | None = None

    def __enter__(self) -> _HttpsStream:
        request = self._client.build_request("GET", self._url)
        self._response = self._client.send(request, stream=True)
        self._response.raise_for_status()
        return self

    def iter_bytes(self) -> Iterator[bytes]:
        if self._response is None:
            msg = "download not started"
            raise RuntimeError(msg)
        return self._response.iter_bytes(CHUNK_SIZE)

    def __exit__(self, *_: object) -> None:
        if self._response is not None:
            self._response.close()
        self._client.close()


def _require_https(request: httpx2.Request) -> None:
    if request.url.scheme != "https":
        msg = f"refusing non-HTTPS model URL: {request.url}"
        raise ModelIntegrityError(msg)


def _https_stream(url: str) -> _HttpsStream:
    return _HttpsStream(url)
