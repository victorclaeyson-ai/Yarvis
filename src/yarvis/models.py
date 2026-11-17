"""Speech models: where they come from and how they are verified.

Every model is pinned:

* Files Yarvis downloads itself (Kokoro, the wake word) are pinned by URL, size and
  SHA-256. A file is only moved into place after its hash matches, and redirects to
  non-HTTPS URLs are refused.
* Parakeet (speech recognition) is downloaded the same way, from URLs pinned to an exact
  Hugging Face commit: the MLX build for the Apple GPU, the ONNX build for every other
  computer's processor. Small files kept in git rather than LFS only have a published
  git object id: they are checked against that.
* Whisper models come from Hugging Face through ``faster-whisper``, pinned to an exact
  git commit, so a changed upstream repository cannot change what we load.
* The Silero VAD model ships inside the ``faster-whisper`` wheel, which is itself pinned by
  hash in ``uv.lock``.

Only ONNX, CTranslate2 and safetensors formats are used: never pickle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol

import httpx2

from yarvis.paths import PRIVATE_FILE_MODE, ensure_private_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    from yarvis.config.models import SttModel

CHUNK_SIZE: Final = 1 << 20  # 1 MiB


type HashAlgorithm = Literal["sha256", "git-blob"]
type ParakeetBuild = Literal["mlx", "onnx"]
"""Parakeet for the Apple GPU (MLX), or for any processor (ONNX Runtime)."""


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ModelFile:
    """A file downloaded from a fixed URL and verified by size and hash.

    The hash is normally SHA-256. ``git-blob`` is the git object id of a file stored in git
    itself (SHA-1 of a ``blob <size>`` header, a NUL byte and the content), which is all
    Hugging Face publishes for small files kept outside LFS.
    """

    name: str
    url: str
    digest: str
    size: int
    algorithm: HashAlgorithm = "sha256"

    def hasher(self) -> _Hasher:
        """A fresh hash object for this file's algorithm."""
        if self.algorithm == "sha256":
            return hashlib.sha256()
        # Matches what git and Hugging Face publish; the SHA-256-checked weights are what
        # matter for safety, and the commit-pinned URL already fixes this file.
        git = hashlib.sha1(usedforsecurity=False)
        git.update(f"blob {self.size}\0".encode())
        return git


_KOKORO_RELEASE: Final = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)

KOKORO_MODEL: Final = ModelFile(
    name="kokoro-v1.0.onnx",
    url=f"{_KOKORO_RELEASE}/kokoro-v1.0.onnx",
    digest="beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
    size=325_505_369,
)
KOKORO_VOICES: Final = ModelFile(
    name="voices-v1.0.bin",
    url=f"{_KOKORO_RELEASE}/voices-v1.0.bin",
    digest="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    size=28_214_398,
)
KOKORO_FILES: Final = (KOKORO_MODEL, KOKORO_VOICES)

_OWW_RELEASE: Final = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

WAKE_WORD_FILES: Final = (
    ModelFile(
        name="melspectrogram.onnx",
        url=f"{_OWW_RELEASE}/melspectrogram.onnx",
        digest="ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
        size=1_087_958,
    ),
    ModelFile(
        name="embedding_model.onnx",
        url=f"{_OWW_RELEASE}/embedding_model.onnx",
        digest="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        size=1_326_578,
    ),
    ModelFile(
        name="hey_jarvis_v0.1.onnx",
        url=f"{_OWW_RELEASE}/hey_jarvis_v0.1.onnx",
        digest="94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
        size=1_271_370,
    ),
)
"""openWakeWord models (code Apache-2.0; pre-trained models CC BY-NC-SA 4.0)."""

_PARAKEET_REPO: Final = "https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/resolve"
_PARAKEET_COMMIT: Final = "8ae155301e23d820d82aa60d24817c900e69e487"

PARAKEET_FILES: Final = (
    ModelFile(
        name="config.json",
        url=f"{_PARAKEET_REPO}/{_PARAKEET_COMMIT}/config.json",
        digest="8955c588b5549ef70811f2121c6c8bda33508992",
        size=36_176,
        algorithm="git-blob",
    ),
    ModelFile(
        name="model.safetensors",
        url=f"{_PARAKEET_REPO}/{_PARAKEET_COMMIT}/model.safetensors",
        digest="b958c37a6baa6874a279108755c8f2818e27bf647d72d54800a234a421341dfe",
        size=2_471_559_904,
    ),
)
"""NVIDIA Parakeet TDT 0.6B v2 converted to MLX (CC BY 4.0): English speech recognition."""

_PARAKEET_ONNX_REPO: Final = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve"
_PARAKEET_ONNX_COMMIT: Final = "0bbb45a3365852604aef28b538a8f066f4ccaa85"


def _onnx(name: str, digest: str, size: int, algorithm: HashAlgorithm = "sha256") -> ModelFile:
    url = f"{_PARAKEET_ONNX_REPO}/{_PARAKEET_ONNX_COMMIT}/{name}"
    return ModelFile(name=name, url=url, digest=digest, size=size, algorithm=algorithm)


PARAKEET_ONNX_FILES: Final = (
    _onnx("config.json", "02a773005666393de591dfd55230b778da1653d8", 97, "git-blob"),
    _onnx("vocab.txt", "f0742785f6073e80b911964c455b05f3609bf23b", 9_384, "git-blob"),
    _onnx(
        "decoder_joint-model.onnx",
        "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191",
        35_792_059,
    ),
    _onnx(
        "encoder-model.onnx",
        "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3",
        41_770_866,
    ),
    _onnx(
        "encoder-model.onnx.data",
        "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41",
        2_435_420_160,
    ),
)
"""The same model exported to ONNX (full precision) by Ilya Stupakov, for onnx-asr: any CPU.

onnx-asr ships the audio preprocessing itself, so only the model and its vocabulary are
downloaded.
"""


def parakeet_files(build: ParakeetBuild = "mlx") -> tuple[ModelFile, ...]:
    """The Parakeet files for one build."""
    return PARAKEET_FILES if build == "mlx" else PARAKEET_ONNX_FILES


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


def wake_word_dir(cache_dir: Path) -> Path:
    """Directory holding the wake-word models."""
    return cache_dir / "models" / "wakeword"


def parakeet_dir(cache_dir: Path, build: ParakeetBuild = "mlx") -> Path:
    """Directory holding the Parakeet files of one build."""
    name = "parakeet-tdt-0.6b-v2" if build == "mlx" else "parakeet-tdt-0.6b-v2-onnx"
    return cache_dir / "models" / name


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
    digest = model.hasher()
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
        if digest.hexdigest() != model.digest:
            msg = f"{model.name}: hash mismatch, the download may have been tampered with"
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
