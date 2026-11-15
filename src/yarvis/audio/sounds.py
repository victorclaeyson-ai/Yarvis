"""Short interface sounds, generated in code (no audio files to ship or license)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from yarvis.speech.base import AudioArray

# Two rising notes (E6, A6): short, soft and clearly "I'm listening".
_NOTES: Final = ((1318.5, 0.07), (1760.0, 0.12))
_VOLUME: Final = 0.16
_ATTACK_SECONDS: Final = 0.004
_DECAY_PER_SECOND: Final = 22.0


def chime(sample_rate: int) -> AudioArray:
    """The wake-up sound at ``sample_rate``."""
    if sample_rate <= 0:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    parts: list[AudioArray] = []
    for frequency, seconds in _NOTES:
        t = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, t / _ATTACK_SECONDS) * np.exp(-t * _DECAY_PER_SECOND)
        parts.append((_VOLUME * np.sin(2 * np.pi * frequency * t) * envelope).astype(np.float32))
    return np.concatenate(parts)
