"""Store picking throughput: how many pickers each hour needs to hold the SLA.

Orders arrive as a Poisson stream, wait for a picker, then for a pack station.
For each hour's arrival rate the smallest picker count that keeps the target
share of orders within the pick-and-pack SLA becomes that hour's staffing norm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import simpy


def _run_hour(rate_per_h: float, pickers: int, cfg: dict, hours: float = 6.0,
              seed: int = 0) -> dict:
    """Simulate a steady arrival rate for a few hours and measure the SLA."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    pick = simpy.Resource(env, capacity=pickers)
    pack = simpy.Resource(env, capacity=cfg["pack_stations"])
    times: list[float] = []

    # Task times are gamma-distributed (CV = 1/sqrt(shape)); exponential times
    # are far more variable than real pick times and make any SLA unreachable
    k = cfg["task_time_shape"]

    def order():
        t0 = env.now
        with pick.request() as r:
            yield r
            yield env.timeout(rng.gamma(k, cfg["pick_time_min"] / k))
        with pack.request() as r:
            yield r
            yield env.timeout(rng.gamma(k, cfg["pack_time_min"] / k))
        times.append(env.now - t0)

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(60.0 / rate_per_h))
            env.process(order())

    env.process(arrivals())
    env.run(until=hours * 60)
    t = np.asarray(times)
    # Drop the first hour as warm-up
    t = t[int(len(t) / hours):] if len(t) > hours else t
    return {"sla_share": float((t <= cfg["sla_min"]).mean()) if len(t) else 1.0,
            "p90_min": float(np.percentile(t, 90)) if len(t) else 0.0,
            "orders_per_h": len(times) / hours}


def pickers_needed(rate_per_h: float, cfg: dict) -> tuple[int, dict]:
    lo, hi = cfg["pickers_range"]
    for n in range(lo, hi + 1):
        res = _run_hour(rate_per_h, n, cfg)
        if res["sla_share"] >= cfg["sla_target"]:
            return n, res
    return hi, res


def staffing_plan(cfg: dict, peak: bool = False) -> pd.DataFrame:
    """One row per operating hour (06:00 onwards) with the required picker count."""
    mult = cfg["peak_multiplier"] if peak else 1.0
    rows = []
    for i, base in enumerate(cfg["hourly_orders"]):
        rate = base * mult
        n, res = pickers_needed(rate, cfg)
        rows.append({"hour": 6 + i, "orders_per_h": rate, "pickers": n,
                     "sla_share": res["sla_share"], "p90_min": res["p90_min"]})
    return pd.DataFrame(rows)
