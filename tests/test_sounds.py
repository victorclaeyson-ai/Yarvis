from __future__ import annotations

import numpy as np
import pytest

from yarvis.audio.sounds import chime


@pytest.mark.parametrize("rate", [16_000, 24_000, 48_000])
def test_chime_is_short_soft_and_clean(rate: int) -> None:
    sound = chime(rate)
    assert sound.dtype == np.float32
    assert 0.15 < len(sound) / rate < 0.25
    assert 0.05 < float(np.abs(sound).max()) <= 0.2  # audible, never loud
    assert abs(float(sound[0])) < 0.01  # no click at the start


def test_chime_needs_a_sample_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        chime(0)
