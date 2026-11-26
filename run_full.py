"""Runs methods on full instances in parallel, checks every solution with the official evaluator and scores
them with the same metric code (full_eval.py).

Method specs: "nn", "greedy" (baselines.py), or "alns:key=value,..." for online_alns.OnlineALNS, e.g.
  alns:budget=5,interval=5,mode=event,lazy=1        (budget=0 gives the rolling greedy ablation)

Examples
  python run_full.py --set test --methods nn greedy alns:budget=5 --seeds 0 1 2 --tag test
  python run_full.py --set tune --methods alns:budget=2,xi=0.1 alns:budget=2,xi=0.4 --tag tune
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np

from mdrp_chunks import ROOT, select_instances, CAL_SEEDS, TEST_SEEDS


def test_instances(k=10):
    """k test-seed instances evenly spaced by order count (covers every variant type and all 7 test seeds)."""
    test = [n for s, n in select_instances() if s in TEST_SEEDS]
    size = {n: sum(1 for _ in open(os.path.join(ROOT, n, "orders.txt"))) - 1 for n in test}
    test.sort(key=lambda n: (size[n], n))
    return [test[i] for i in np.linspace(0, len(test) - 1, k).round().astype(int)]


def tune_instances():
    """Calibration seeds only: never used for testing."""
    return [n for s, n in select_instances() if s in CAL_SEEDS]


def parse(spec):
    name, _, rest = spec.partition(":")
    cfg = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        cfg[k] = v
    return name, cfg


def job(args):
    inst, spec, seed, tag = args
    from full_eval import metrics, official_check
    name, cfg = parse(spec)
    label = spec.replace(":", "_").replace(",", "_").replace("=", "")
    sol_dir = os.path.join("solutions", tag, inst, f"{label}_s{seed}")
    inst_dir = os.path.join(ROOT, inst)
    try:
        t0 = time.perf_counter()
        info = {}
        if name in ("nn", "greedy"):
            import baselines
            ctx, routes, _ = baselines.run(inst_dir, name)
            rt = time.perf_counter() - t0
            baselines.write_solution(ctx, routes, sol_dir)
        else:
            import alns as A
            from online_alns import OnlineALNS
            params = {k: float(cfg[k]) for k in ("xi", "cool", "w_start") if k in cfg}
            if "patience" in cfg:
                params["patience"] = int(cfg["patience"]) or None
            kn = int(cfg.get("k_near", 20))
            sim = OnlineALNS(inst_dir, budget=float(cfg.get("budget", 5)), interval=int(cfg.get("interval", 5)),
                             mode=cfg.get("mode", "event"), lazy=bool(int(cfg.get("lazy", 1))),
                             k_near=kn if kn > 0 else None, seed=seed, params=params)
            sim.run()
            rt = sim.runtime
            info = sim.write_solution(sol_dir)
            info.pop("operator_weights", None)
            info.pop("runtime_s", None)
        ok, _ = official_check(inst_dir, sol_dir, sol_dir)
        m = metrics(inst_dir, sol_dir)
        return dict(instance=inst, method=spec, seed=seed, feasible=ok, runtime_s=rt, **m, **info)
    except Exception:
        return dict(instance=inst, method=spec, seed=seed, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["test", "tune"], default="test")
    ap.add_argument("--instances", nargs="*")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    insts = args.instances or (test_instances() if args.set == "test" else tune_instances())
    jobs = []
    for spec in args.methods:
        deterministic = spec in ("nn", "greedy") or parse(spec)[1].get("budget") == "0"
        seeds = [0] if deterministic else args.seeds                      # no search, no randomness
        for inst in insts:
            for s in seeds:
                jobs.append((inst, spec, s, args.tag))
    # longest jobs first so the pool finishes together
    size = {n: sum(1 for _ in open(os.path.join(ROOT, n, "orders.txt"))) for n in insts}
    jobs.sort(key=lambda j: (-("alns" in j[1]), -size[j[0]]))
    out = f"results_full_{args.tag}.jsonl"
    print(f"{len(jobs)} jobs on {len(insts)} instances with {args.workers} workers -> {out}", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if "error" in r:
                print("ERROR", r["instance"], r["method"], r["seed"], r["error"][-400:], flush=True)
            else:
                print(f"[{time.time() - t0:6.0f}s] {r['instance']:18s} {r['method']:40s} s{r['seed']} "
                      f"feasible={r['feasible']} obj/order {r['obj_per_order']:.3f} delivered {r['delivered']}/{r['orders']} "
                      f"c2d {r['mean_c2d']:.2f} late {r['late_pct']:.1f}% travel {r['travel_min']} "
                      f"({r['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
