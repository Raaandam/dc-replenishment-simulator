"""Load M5 daily sales into a long store x SKU x day demand table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# M5 FOODS has three departments; map them to grocery temperature classes so
# perishability can be modelled. The mapping is an assumption, not M5 data.
DEPT_TO_CLASS = {"FOODS_1": "chilled", "FOODS_2": "ambient", "FOODS_3": "fresh"}


def load_m5(path: str | Path, category: str, stores: list[str], n_skus: int,
            days: int) -> pd.DataFrame:
    """Return columns: store, sku, temp_class, day, demand (the last `days` days)."""
    wide = pd.read_csv(path)
    wide = wide[(wide["cat_id"] == category) & (wide["store_id"].isin(stores))]
    day_cols = [c for c in wide.columns if c.startswith("d_")][-days:]

    totals = wide.groupby("item_id")[day_cols].sum().sum(axis=1)
    top = totals.nlargest(n_skus).index
    wide = wide[wide["item_id"].isin(top)]

    long = wide.melt(id_vars=["store_id", "item_id", "dept_id"], value_vars=day_cols,
                     var_name="d", value_name="demand")
    long["day"] = long["d"].str[2:].astype(int)
    long["day"] -= long["day"].min()
    long["temp_class"] = long["dept_id"].map(DEPT_TO_CLASS)
    return (long.rename(columns={"store_id": "store", "item_id": "sku"})
                [["store", "sku", "temp_class", "day", "demand"]])


def split_history(demand: pd.DataFrame, history_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on the first history_days, simulate on the rest (re-indexed from day 0)."""
    hist = demand[demand["day"] < history_days]
    sim = demand[demand["day"] >= history_days].copy()
    sim["day"] -= history_days
    return hist, sim
