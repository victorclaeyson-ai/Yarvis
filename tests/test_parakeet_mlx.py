"""The real Parakeet code path on MLX, with a tiny randomly initialised model.

Runs wherever ``parakeet-mlx`` is installed (Apple Silicon Macs, including CI's macOS
runners) and is skipped elsewhere. The model is too small to understand anything, but it
goes through exactly the same loading and spectrogram code as the real one: that is where
a dtype mistake once stopped Yarvis at start-up.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from yarvis.speech import parakeet

mx: Any = pytest.importorskip("mlx.core")
utils: Any = pytest.importorskip("parakeet_mlx.utils")

if TYPE_CHECKING:
    from pathlib import Path

_DURATIONS = [0, 1, 2, 3, 4]
_VOCABULARY = ["▁play", "▁on", "▁spot", "ify", "▁we", "▁go", "s", "a", "e", "o"]
_TINY_TDT: dict[str, object] = {
    "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    "model_defaults": {"tdt_durations": _DURATIONS},
    "preprocessor": {
        "sample_rate": 16_000, "normalize": "per_feature", "window_size": 0.025,
        "window_stride": 0.01, "window": "hann", "features": 128, "n_fft": 512,
        "dither": 1e-05, "pad_to": 0, "pad_value": 0.0,
    },
    "encoder": {
        "feat_in": 128, "n_layers": 1, "d_model": 32, "n_heads": 2,
        "ff_expansion_factor": 2, "subsampling_factor": 8, "self_attention_model": "rel_pos",
        "subsampling": "dw_striding", "conv_kernel_size": 9, "subsampling_conv_channels": 16,
        "pos_emb_max_len": 5000,
    },
    "decoder": {
        "blank_as_pad": True, "vocab_size": len(_VOCABULARY),
        "prednet": {"pred_hidden": 16, "pred_rnn_layers": 1},
    },
    "joint": {
        "num_classes": len(_VOCABULARY), "vocabulary": _VOCABULARY,
        "num_extra_outputs": len(_DURATIONS),
        "jointnet": {
            "joint_hidden": 16, "activation": "relu", "encoder_hidden": 32, "pred_hidden": 16,
        },
    },
    "decoding": {"model_type": "tdt", "durations": _DURATIONS, "greedy": {"max_symbols": 10}},
}  # fmt: skip


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    mx.set_default_device(mx.cpu)  # CI machines may have no usable GPU
    (tmp_path / "config.json").write_text(json.dumps(_TINY_TDT))
    utils.from_config(_TINY_TDT).save_weights(str(tmp_path / "model.safetensors"))
    return tmp_path


@pytest.mark.parametrize("seconds", [0.005, 0.5, 3.0])
def test_the_real_loader_transcribes_microphone_audio(tiny_model: Path, seconds: float) -> None:
    engine = parakeet._load_mlx(tiny_model)  # pyright: ignore[reportPrivateUsage]
    audio = np.random.default_rng(0).normal(0, 0.1, round(16_000 * seconds))
    text = engine.transcribe(audio.astype(np.float32))
    assert isinstance(text, str)
    if seconds < 0.01:  # shorter than one analysis step
        assert text == ""
