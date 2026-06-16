from __future__ import annotations

import numpy as np


def pdsch_decode(rng: np.random.Generator, fail_probability: float) -> bool:
    return bool(rng.random() > fail_probability)
