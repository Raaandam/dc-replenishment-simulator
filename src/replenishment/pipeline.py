"""CLI entry point: classify SKUs, set reorder parameters, simulate each policy, size staffing.

    python -m replenishment.pipeline --config configs/config.yaml
    python -m replenishment.pipeline --synthetic --output-dir reports/synthetic_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from . import data, synthetic
from .classify import abc_xyz, reorder_parameters
from .sim import simulate
from .throughput import staffing_plan


def run(cfg: dict, demand: pd.DataFrame, output_dir: Path) -> dict:
    planner = output_dir / "planner"
    planner.mkdir(parents=True, exist_ok=True)

    hist, sim_demand = data.split_history(demand, cfg["data"]["history_days"])
    sim_demand = sim_demand[sim_demand["day"] < cfg["data"]["sim_days"]]
    params = reorder_parameters(abc_xyz(hist), cfg)
    params.to_csv(planner / "reorder_parameters.csv", index=False)

    results = {}
    for policy in cfg["policies"]:
        results[policy["name"]] = simulate(params, sim_demand, cfg, policy)
    comparison = pd.DataFrame(results).T
    comparison.index.name = "policy"
    comparison.to_csv(planner / "policy_comparison.csv")

    normal = staffing_plan(cfg["throughput"])
    peak = staffing_plan(cfg["throughput"], peak=True)
    staffing = normal.merge(peak, on="hour", suffixes=("_normal", "_peak"))
    staffing.to_csv(planner / "staffing_plan.csv", index=False)

    metrics = {
        "policies": results,
        "class_mix": params["cls"].value_counts().to_dict(),
        "picker_hours_normal_day": int(normal["pickers"].sum()),
        "picker_hours_peak_day": int(peak["pickers"].sum()),
        "max_pickers_peak": int(peak["pickers"].max()),
    }
    if "fixed_qty" in results:
        base = results["fixed_qty"]
        metrics["vs_fixed_qty"] = {
            name: {"fill_rate_pts": 100 * (r["fill_rate"] - base["fill_rate"]),
                   "wastage_reduction_pct": 100 * (1 - r["wasted_units"] / base["wasted_units"])
                   if base["wasted_units"] else 0.0}
            for name, r in results.items() if name != "fixed_qty"
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic demand instead of the M5 file")
    ap.add_argument("--output-dir", default="reports")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]
    days = d["history_days"] + d["sim_days"]
    if args.synthetic:
        demand = synthetic.make_demand(len(d["stores"]), d["n_skus"], days, d["seed"])
    else:
        demand = data.load_m5(d["sales"], d["category"], d["stores"], d["n_skus"], days)

    metrics = run(cfg, demand, Path(args.output_dir))
    table = pd.DataFrame(metrics["policies"]).T
    print(table[["fill_rate", "perishable_wastage_pct", "avg_store_on_hand", "trucks"]]
          .round(3).to_string())
    print(f"picker-hours: normal {metrics['picker_hours_normal_day']}, "
          f"peak {metrics['picker_hours_peak_day']} (max {metrics['max_pickers_peak']} at once)")


if __name__ == "__main__":
    main()
