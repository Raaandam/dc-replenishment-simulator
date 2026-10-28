"""ABC-XYZ classification and reorder parameters per store x SKU."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def abc_xyz(history: pd.DataFrame, a_share: float = 0.8, b_share: float = 0.95,
            x_cv: float = 0.5, y_cv: float = 1.0) -> pd.DataFrame:
    """ABC on SKU volume share; XYZ on the coefficient of variation of daily demand."""
    sku = history.groupby(["sku", "temp_class"])["demand"].sum().reset_index(name="volume")
    sku = sku.sort_values("volume", ascending=False)
    share = sku["volume"] / sku["volume"].sum()
    start = share.cumsum() - share
    sku["abc"] = np.where(start < a_share, "A", np.where(start < b_share, "B", "C"))

    stats = history.groupby(["store", "sku"])["demand"].agg(mean="mean", std="std").reset_index()
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    stats["xyz"] = np.where(stats["cv"] <= x_cv, "X", np.where(stats["cv"] <= y_cv, "Y", "Z"))
    out = stats.merge(sku[["sku", "temp_class", "abc"]], on="sku")
    out["cls"] = out["abc"] + out["xyz"]
    return out


def reorder_parameters(classes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Safety stock, reorder point (s) and order-up-to level (S) per store x SKU.

    Lead-time demand covers the review period plus transport lead time:
        SS = z * sigma_daily * sqrt(L + R)
        s  = mean_daily * (L + R) + SS
        S  = s + order_up_to_days[temp_class] * mean_daily
    Perishables use a short S so stock cannot outlive its shelf life.
    """
    lead_days = cfg["network"]["dc_to_store_lead_time_h"] / 24 + 1  # +1: daily dispatch cycle
    p = classes.copy()
    p["service_level"] = p["cls"].map(cfg["service_level"]).fillna(0.9)
    p["z"] = p["service_level"].map(NormalDist().inv_cdf)
    p["safety_stock"] = p["z"] * p["std"].fillna(0) * np.sqrt(lead_days)
    p["s"] = np.ceil(p["mean"] * lead_days + p["safety_stock"])
    cover = p["temp_class"].map(cfg["order_up_to_days"])
    p["S"] = np.ceil(p["s"] + cover * p["mean"]).clip(lower=p["s"] + 1)
    # Never hold more than can be sold in the shelf life left on arrival.
    # Stock spends about a day at the DC first, so that day is already gone.
    remaining = (p["temp_class"].map(cfg["shelf_life_days"]) - 1).clip(lower=1)
    p["S"] = np.minimum(p["S"], np.ceil(p["mean"] * remaining + p["safety_stock"]))
    p["S"] = np.maximum(p["S"], p["s"] + 1)
    return p
