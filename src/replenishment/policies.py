"""Store replenishment policies. Each returns an order quantity for one store x SKU."""

from __future__ import annotations

import numpy as np

from .inventory import Stock


def s_S(stock: Stock, p: dict, day: int) -> float:
    """Review inventory position daily; below s, order up to S."""
    return max(0.0, p["S"] - stock.position) if stock.position <= p["s"] else 0.0


def min_max(stock: Stock, p: dict, day: int, case_pack: int = 6) -> float:
    """The common store-ops rule: compare on-hand (not position) with min, then
    order up to max rounded up to whole supplier cases.

    Ignoring stock already on a truck double-orders whenever transit is longer
    than the review cycle, and case rounding overshoots max on slow SKUs. Both
    are expensive for perishables; the simulation shows how much.
    """
    if stock.on_hand > p["s"]:
        return 0.0
    need = max(0.0, p["S"] - stock.on_hand)
    return float(np.ceil(need / case_pack) * case_pack)


def periodic(stock: Stock, p: dict, day: int, review_days: int = 2) -> float:
    """Review every R days and order up to S + R days of extra cover."""
    if day % review_days:
        return 0.0
    target = p["S"] + (review_days - 1) * p["mean"]
    return max(0.0, target - stock.position)


def fixed_qty(stock: Stock, p: dict, day: int, days_of_cover: int = 3) -> float:
    """Baseline: when position drops below s, order a fixed days-of-cover quantity."""
    return days_of_cover * p["mean"] if stock.position <= p["s"] else 0.0


POLICIES = {"s_S": s_S, "min_max": min_max, "periodic": periodic, "fixed_qty": fixed_qty}
