"""M5-shaped synthetic daily demand for smoke tests without the real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["fresh", "chilled", "ambient"]


def make_demand(n_stores: int = 8, n_skus: int = 60, days: int = 300,
                seed: int = 0) -> pd.DataFrame:
    """Negative-binomial daily demand with a weekly cycle and SKU-level skew."""
    rng = np.random.default_rng(seed)
    base = 150.0 / np.arange(1, n_skus + 1) ** 0.8         # skewed SKU volumes, units/day
    dispersion = rng.uniform(1.5, 8.0, size=n_skus)       # low = lumpy (Z), high = smooth (X)
    temp = rng.choice(CLASSES, size=n_skus, p=[0.35, 0.25, 0.40])
    store_scale = rng.uniform(0.7, 1.3, size=n_stores)
    weekly = np.array([0.9, 0.85, 0.9, 0.95, 1.05, 1.25, 1.2])

    rows = []
    for s in range(n_stores):
        for k in range(n_skus):
            mu = base[k] * store_scale[s] * weekly[np.arange(days) % 7]
            r = dispersion[k]
            demand = rng.negative_binomial(r, r / (r + mu))
            rows.append(pd.DataFrame({"store": f"S{s + 1}", "sku": f"SKU{k + 1:03d}",
                                      "temp_class": temp[k], "day": np.arange(days),
                                      "demand": demand}))
    return pd.concat(rows, ignore_index=True)
