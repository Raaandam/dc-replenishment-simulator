import pytest
import yaml

from replenishment.classify import abc_xyz, reorder_parameters
from replenishment.data import split_history
from replenishment.inventory import Stock
from replenishment.pipeline import run
from replenishment.policies import min_max, s_S
from replenishment.synthetic import make_demand
from replenishment.throughput import pickers_needed


@pytest.fixture
def cfg():
    with open("configs/config.yaml") as f:
        c = yaml.safe_load(f)
    c["data"].update(history_days=60, sim_days=30)
    c["throughput"]["hourly_orders"] = [30, 90]
    return c


@pytest.fixture
def demand():
    return make_demand(n_stores=3, n_skus=12, days=90, seed=3)


def test_fifo_and_expiry():
    st = Stock()
    st.receive(5, expires_at=10)
    st.receive(5, expires_at=20)
    assert st.take(3) == 3
    assert st.expire(now=15) == 2      # only the older batch's remainder
    assert st.on_hand == 5


def test_min_max_ignores_pipeline():
    st = Stock()
    st.receive(2, 100)
    st.on_order = 50
    p = {"s": 10, "S": 30, "mean": 5}
    assert s_S(st, p, 0) == 0          # position 52 is above s
    assert min_max(st, p, 0, case_pack=6) == 30   # on hand 2 < min: 28 rounded up to cases


def test_reorder_parameters_sane(cfg, demand):
    hist, _ = split_history(demand, 60)
    p = reorder_parameters(abc_xyz(hist), cfg)
    assert (p["S"] > p["s"]).all()
    assert (p["safety_stock"] >= 0).all()
    assert set(p["abc"]) <= {"A", "B", "C"}


def test_more_load_needs_more_pickers(cfg):
    low, _ = pickers_needed(20, cfg["throughput"])
    high, _ = pickers_needed(120, cfg["throughput"])
    assert high > low


def test_pipeline_smoke(cfg, demand, tmp_path):
    m = run(cfg, demand, tmp_path)
    assert (tmp_path / "planner" / "policy_comparison.csv").exists()
    for r in m["policies"].values():
        assert 0 <= r["fill_rate"] <= 1
