"""SimPy discrete-event model of one DC replenishing a set of dark stores.

Time is in hours. Each simulated day runs:

    00:00  DC reviews its own stock and orders from suppliers
    02:00  each store's policy places an order; the DC picks it from FIFO stock,
           trucks queue for a limited number of loading docks, then drive out
    12:00  the day's customer demand is served FIFO; unmet demand is a lost sale
    23:30  expired batches are written off at stores and the DC
"""

from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist

import numpy as np
import pandas as pd
import simpy

from .inventory import Stock
from .policies import POLICIES


class Network:
    def __init__(self, params: pd.DataFrame, demand: pd.DataFrame, cfg: dict,
                 policy: str, policy_kwargs: dict | None = None):
        self.cfg, self.net = cfg, cfg["network"]
        self.env = simpy.Environment()
        self.docks = simpy.Resource(self.env, capacity=self.net["dc_dock_doors"])
        self.policy = POLICIES[policy]
        self.policy_kwargs = policy_kwargs or {}

        self.params = {(r.store, r.sku): r._asdict() for r in params.itertuples(index=False)}
        self.shelf_h = {r.sku: cfg["shelf_life_days"][r.temp_class] * 24
                        for r in params.itertuples(index=False)}
        self.temp = {r.sku: r.temp_class for r in params.itertuples(index=False)}
        self.demand = demand.set_index(["day", "store", "sku"])["demand"].to_dict()
        self.stores = sorted(params["store"].unique())
        self.skus = sorted(params["sku"].unique())
        self.days = int(demand["day"].max()) + 1

        self.store_stock = {k: Stock() for k in self.params}
        self.dc_stock = {sku: Stock() for sku in self.skus}
        self.m = defaultdict(float)
        self.daily_on_hand: list[float] = []
        self._init_stock()

    def _dc_targets(self, sku: str) -> tuple[float, float]:
        """DC (s, S) on aggregate store demand over the supplier lead time."""
        rows = [self.params[(s, sku)] for s in self.stores]
        mean = sum(r["mean"] for r in rows)
        std = np.sqrt(sum((r["std"] or 0) ** 2 for r in rows))
        L = self.net["supplier_lead_time_days"] + 1
        z = NormalDist().inv_cdf(0.98)
        s = mean * L + z * std * np.sqrt(L)
        cover = self.cfg["order_up_to_days"][self.temp[sku]]
        return s, s + cover * mean

    def _init_stock(self) -> None:
        for (store, sku), p in self.params.items():
            self.store_stock[store, sku].receive(p["S"], self.shelf_h[sku])
        for sku in self.skus:
            _, S = self._dc_targets(sku)
            self.dc_stock[sku].receive(S, self.shelf_h[sku])

    # --- processes -------------------------------------------------------
    def dc_replenish(self):
        L = self.net["supplier_lead_time_days"] * 24
        while True:
            for sku in self.skus:
                st = self.dc_stock[sku]
                s, S = self._dc_targets(sku)
                if st.position <= s:
                    qty = S - st.position
                    st.on_order += qty
                    self.env.process(self._supplier_delivery(sku, qty, L))
            yield self.env.timeout(24)

    def _supplier_delivery(self, sku, qty, lead_h):
        yield self.env.timeout(lead_h)
        st = self.dc_stock[sku]
        st.on_order -= qty
        st.receive(qty, self.env.now + self.shelf_h[sku])

    def dispatch(self):
        yield self.env.timeout(self.net["dispatch_hour"])
        cap = self.net["truck_capacity_units"] * self.net["trucks_per_store_per_day"]
        while True:
            day = int(self.env.now // 24)
            for store in self.stores:
                want = {sku: self.policy(self.store_stock[store, sku], self.params[store, sku],
                                         day, **self.policy_kwargs)
                        for sku in self.skus}
                total = sum(want.values())
                if total <= 0:
                    continue
                scale = min(1.0, cap / total)   # truck capacity cuts every line pro rata
                self.m["truck_cut_units"] += total * (1 - scale)
                load = {}
                for sku, q in want.items():
                    batches = self.dc_stock[sku].take_batches(np.floor(q * scale))
                    issued = sum(b.qty for b in batches)
                    self.m["dc_short_units"] += np.floor(q * scale) - issued
                    if issued:
                        load[sku] = batches
                        self.store_stock[store, sku].on_order += issued
                if load:
                    self.m["trucks"] += 1
                    self.env.process(self._truck(store, load))
            yield self.env.timeout(24)

    def _truck(self, store, load):
        arrived = self.env.now
        with self.docks.request() as req:
            yield req
            self.m["dock_wait_h"] += self.env.now - arrived
            yield self.env.timeout(self.net["load_time_h"])
        yield self.env.timeout(self.net["dc_to_store_lead_time_h"])
        for sku, batches in load.items():
            st = self.store_stock[store, sku]
            for b in batches:
                st.on_order -= b.qty
                st.receive(b.qty, b.expires_at)

    def serve_demand(self):
        yield self.env.timeout(12)
        while True:
            day = int(self.env.now // 24)
            for (store, sku), st in self.store_stock.items():
                d = self.demand.get((day, store, sku), 0)
                served = st.take(d)
                self.m["demand"] += d
                self.m["served"] += served
                self.m[f"served_{self.temp[sku]}"] += served
            self.daily_on_hand.append(sum(st.on_hand for st in self.store_stock.values()))
            yield self.env.timeout(24)

    def expire(self):
        yield self.env.timeout(23.5)
        while True:
            for (store, sku), st in self.store_stock.items():
                w = st.expire(self.env.now)
                self.m["wasted_store"] += w
                self.m[f"wasted_{self.temp[sku]}"] += w
            for sku, st in self.dc_stock.items():
                w = st.expire(self.env.now)
                self.m["wasted_dc"] += w
                self.m[f"wasted_{self.temp[sku]}"] += w
            yield self.env.timeout(24)

    # --- run -------------------------------------------------------------
    def run(self) -> dict:
        for proc in (self.dc_replenish, self.dispatch, self.serve_demand, self.expire):
            self.env.process(proc())
        self.env.run(until=self.days * 24)
        m = self.m
        perishable_served = m["served_fresh"] + m["served_chilled"]
        perishable_wasted = m["wasted_fresh"] + m["wasted_chilled"]
        return {
            "fill_rate": m["served"] / m["demand"] if m["demand"] else 1.0,
            "lost_sales_units": m["demand"] - m["served"],
            "wasted_units": m["wasted_store"] + m["wasted_dc"],
            "perishable_wastage_pct": 100 * perishable_wasted / max(1.0, perishable_served + perishable_wasted),
            "avg_store_on_hand": float(np.mean(self.daily_on_hand)) if self.daily_on_hand else 0.0,
            "trucks": m["trucks"],
            "truck_cut_units": m["truck_cut_units"],
            "dc_short_units": m["dc_short_units"],
            "avg_dock_wait_h": m["dock_wait_h"] / max(1.0, m["trucks"]),
        }


def simulate(params: pd.DataFrame, demand: pd.DataFrame, cfg: dict, policy: dict) -> dict:
    kwargs = {k: v for k, v in policy.items() if k != "name"}
    return Network(params, demand, cfg, policy["name"], kwargs).run()
