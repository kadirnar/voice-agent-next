"""Energy-based VAD with an adaptive noise floor. Zero dependencies beyond numpy.

Good enough for clean audio, tests and synthetic benchmarks; use Silero/TEN VAD for
real microphones.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ..registry import register_provider
from ..vad import VAD, VADOptions

__all__ = ["EnergyVAD"]


class _EnergyInference:
    def __init__(
        self, threshold_db: float, margin_db: float, adaptive: bool, slope_db: float
    ) -> None:
        self.threshold_db = threshold_db
        self.margin_db = margin_db
        self.adaptive = adaptive
        self.slope_db = slope_db
        self.reset()

    def reset(self) -> None:
        self.noise_floor = -90.0
        self._initialized = False

    def __call__(self, window: npt.NDArray[np.float32]) -> float:
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) + 1e-9
        db = 20.0 * math.log10(rms)
        threshold = self.threshold_db
        if self.adaptive:
            if not self._initialized:
                self.noise_floor = min(db, self.threshold_db - self.margin_db)
                self._initialized = True
            elif db < self.noise_floor:
                self.noise_floor = 0.9 * self.noise_floor + 0.1 * db  # fall quickly
            else:
                self.noise_floor += (db - self.noise_floor) * 0.002  # rise slowly
            threshold = max(threshold, self.noise_floor + self.margin_db)
        x = (db - threshold) / self.slope_db
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


@register_provider(
    "vad",
    "energy",
    description="Energy (RMS) VAD with adaptive noise floor — zero dependencies",
    local=True,
)
class EnergyVAD(VAD):
    """RMS-energy voice activity detector.

    Args:
        threshold_db: minimum level (dBFS) considered speech.
        margin_db: with ``adaptive``, speech must exceed the noise floor by this much.
        adaptive: track the noise floor.
        window_ms: analysis window.
        sample_rate: analysis sample rate.
    """

    provider = "energy"

    def __init__(
        self,
        *,
        model: str | None = None,
        threshold_db: float = -40.0,
        margin_db: float = 12.0,
        adaptive: bool = True,
        window_ms: float = 20.0,
        sample_rate: int = 16_000,
        options: VADOptions | None = None,
        **option_overrides: float,
    ) -> None:
        opts = options or VADOptions(**option_overrides)
        super().__init__(
            sample_rate=sample_rate,
            window_samples=round(sample_rate * window_ms / 1000.0),
            options=opts,
            model=model or "energy",
        )
        self.threshold_db = threshold_db
        self.margin_db = margin_db
        self.adaptive = adaptive

    def _new_inference(self) -> _EnergyInference:
        return _EnergyInference(self.threshold_db, self.margin_db, self.adaptive, slope_db=2.0)
