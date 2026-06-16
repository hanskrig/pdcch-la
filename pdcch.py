from __future__ import annotations

import numpy as np
from config import AL_SET, PdcchCurveConfig


def pdcch_error_prob(cqi: float, al: int, curve: PdcchCurveConfig) -> float:
    if al not in curve.cqi_1pct:
        raise ValueError(f"Unsupported aggregation level {al}")
    p = curve.target * np.exp(-(float(cqi) - curve.cqi_1pct[al]) / curve.slope)
    return float(np.clip(p, curve.p_min, curve.p_max))


def select_aggregation_level(cqi_eff: float, curve: PdcchCurveConfig, target_pdcch_error: float = 0.01) -> int:
    for al in AL_SET:
        if pdcch_error_prob(cqi_eff, al, curve) <= target_pdcch_error:
            return al
    return AL_SET[-1]
