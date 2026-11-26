"""Scores any full-instance solution written in the official format, the same way for every method.

feasibility : the official Grubhub evaluator (mdrplib/MDRP_code/compute_performance_summary.py)
objective   : the team's weighted cost (baselines.py weights) recomputed from the solution files
lower bound : per-order bound on the delivery-time + lateness part, assuming a courier is waiting at the
              restaurant when the food is ready and drives straight to the customer

Usage: python full_eval.py <instance_dir> <solution_dir>
"""
import contextlib
import importlib.util
import io
import math
import os
import sys

import pandas as pd

from baselines import W_DELIVERY, W_LATE, W_WAIT, W_TRAVEL
from mdrp_data import Instance

_EVAL = os.path.join("mdrplib", "MDRP_code", "compute_performance_summary.py")


def official_check(inst_dir, sol_dir, out_dir=None):
    """Runs the official evaluator. Returns (feasible, its performance summary text)."""
    spec = importlib.util.spec_from_file_location("mdrp_evaluator", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out_dir = out_dir or sol_dir
    os.makedirs(out_dir, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        feasible = mod.compute_performance_summary(inst_dir, sol_dir, out_dir)[0]
    perf = os.path.join(out_dir, "solution_performance.txt")
    text = open(perf).read() if feasible and os.path.exists(perf) else ""
    return bool(feasible), text


def metrics(inst_dir, sol_dir, inst=None):
    inst = inst or Instance(inst_dir)
    ps, ds, target = inst.pickup_service, inst.dropoff_service, inst.target_c2d
    o = pd.read_table(os.path.join(sol_dir, "solution_info_orders.txt"), sep=" ").set_index("order")
    for c in ("pickup_time", "dropoff_time"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    dl = o.dropna(subset=["dropoff_time"])
    c2d = dl.dropoff_time - dl.placement_time
    late = (c2d - target).clip(lower=0)
    wait = (dl.pickup_time - ps - dl.ready_time).clip(lower=0)      # food ready, courier not there yet
    travel = 0
    with open(os.path.join(sol_dir, "solution_info_couriers.txt")) as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) == 4:
                travel += inst.travel(p[2] if p[2] != "0" else p[0], p[3])
    service = float((W_DELIVERY * c2d + W_LATE * late + W_WAIT * wait).sum())
    obj = service + W_TRAVEL * travel
    # lower bound on W_DELIVERY * c2d + W_LATE * lateness for each delivered order
    orders = inst.orders
    lb = 0.0
    for oid, row in dl.iterrows():
        rest = orders.restaurant[oid]
        c = orders.ready_time[oid] + ps + inst.travel(rest, oid) + ds - orders.placement_time[oid]
        lb += W_DELIVERY * c + W_LATE * max(0.0, c - target)
    n = len(dl)
    return {
        "orders": int(len(o)), "delivered": int(n), "unserved": int(len(o) - n),
        "objective": obj, "obj_per_order": obj / max(1, n),
        "mean_c2d": float(c2d.mean()), "p90_c2d": float(c2d.quantile(0.9)),
        "late_pct": float(100 * (c2d > target).mean()), "mean_lateness": float(late.mean()),
        "mean_food_wait": float(wait.mean()), "travel_min": int(travel),
        "service_gap_to_lb_pct": float(100 * ((W_DELIVERY * c2d + W_LATE * late).sum() / lb - 1)) if lb else math.nan,
    }


if __name__ == "__main__":
    inst_dir, sol_dir = sys.argv[1], sys.argv[2]
    ok, _ = official_check(inst_dir, sol_dir)
    print("official evaluator:", "FEASIBLE" if ok else "INFEASIBLE")
    for k, v in metrics(inst_dir, sol_dir).items():
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")
