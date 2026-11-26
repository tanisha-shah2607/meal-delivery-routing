"""Aggregates results_full_<tag>.jsonl: per-instance table and gaps to the greedy-insertion baseline.

Gaps are computed per instance (method vs greedy on the same instance), averaged over ALNS run seeds, then
summarized across instances and across instance *seeds* (the 10 test instances come from 7 seeds), as the
proposal asks. Usage: python summarize_full.py <tag> [--ref greedy] [--md]"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np

KEYS = ("obj_per_order", "objective", "delivered", "mean_c2d", "late_pct", "mean_food_wait", "travel_min",
        "runtime_s", "service_gap_to_lb_pct")


def load(tag):
    rows = [json.loads(l) for l in open(f"results_full_{tag}.jsonl")]
    bad = [r for r in rows if "error" in r]
    for r in bad:
        print("ERROR row:", r["instance"], r["method"], r["seed"])
    rows = [r for r in rows if "error" not in r]
    infeasible = [(r["instance"], r["method"], r["seed"]) for r in rows if not r["feasible"]]
    if infeasible:
        print("INFEASIBLE (official evaluator):", infeasible)
    return rows


def aggregate(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["instance"], r["method"])].append(r)
    agg = {}
    for k, rs in g.items():
        a = {key: float(np.mean([r[key] for r in rs])) for key in KEYS}
        a["seed_std_obj"] = float(np.std([r["obj_per_order"] for r in rs]))
        a["runs"] = len(rs)
        a["feasible"] = all(r["feasible"] for r in rs)
        a["alns_iters"] = float(np.mean([r.get("alns_iters", 0) for r in rs]))
        a["mean_requests"] = float(np.mean([r.get("mean_requests", 0) for r in rs]))
        agg[k] = a
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ref", default="greedy")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    rows = load(args.tag)
    agg = aggregate(rows)
    insts = sorted({i for i, _ in agg}, key=lambda n: agg[(n, args.ref)]["delivered"] if (n, args.ref) in agg else 0)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    rng = np.random.default_rng(0)

    print(f"\nper instance: objective per delivered order (lower is better), ALNS averaged over run seeds")
    print(f"{'instance':18s} {'orders':>6s} " + " ".join(f"{m[:22]:>22s}" for m in methods))
    for n in insts:
        cells = []
        for m in methods:
            a = agg.get((n, m))
            cells.append(f"{a['obj_per_order']:10.3f} ({100 * (a['obj_per_order'] / agg[(n, args.ref)]['obj_per_order'] - 1):+5.2f}%)"
                         if a else "-")
        ref = agg[(n, args.ref)]
        print(f"{n:18s} {int(ref['delivered']):6d} " + " ".join(f"{c:>22s}" for c in cells))

    print(f"\nsummary over {len(insts)} instances (gap = per-instance % change vs {args.ref}; negative = better)")
    hdr = (f"{'method':40s} {'mean gap':>9s} {'95% CI (instances)':>19s} {'seed-level':>16s} {'wins':>5s} "
           f"{'deliv':>6s} {'c2d':>6s} {'late%':>6s} {'wait':>5s} {'travel':>7s} {'LBgap%':>7s} {'sec/run':>8s} {'run std':>8s}")
    print(hdr)
    summary = {}
    for m in methods:
        gaps, seeds, keep = [], [], []
        for n in insts:
            if (n, m) in agg and (n, args.ref) in agg:
                gaps.append(100 * (agg[(n, m)]["obj_per_order"] / agg[(n, args.ref)]["obj_per_order"] - 1))
                seeds.append(int(re.match(r"\d+", n).group()))
                keep.append(n)
        gaps, seeds = np.array(gaps), np.array(seeds)
        bs = gaps[rng.integers(0, len(gaps), size=(10000, len(gaps)))].mean(1)
        per_seed = np.array([gaps[seeds == s].mean() for s in sorted(set(seeds))])
        mean = lambda key: np.mean([agg[(n, m)][key] for n in keep])
        s = {"mean_gap_pct": float(gaps.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             "seed_mean_gap": float(per_seed.mean()), "seed_std_gap": float(per_seed.std()), "n_seeds": int(len(per_seed)),
             "wins": int((gaps < -1e-9).sum()), "n": int(len(gaps)),
             **{k: float(mean(k)) for k in ("delivered", "mean_c2d", "late_pct", "mean_food_wait", "travel_min",
                                             "runtime_s", "service_gap_to_lb_pct", "seed_std_obj")}}
        summary[m] = s
        print(f"{m:40s} {s['mean_gap_pct']:+8.2f}% [{s['ci95'][0]:+6.2f},{s['ci95'][1]:+6.2f}] "
              f"{s['seed_mean_gap']:+6.2f}+/-{s['seed_std_gap']:4.2f} {s['wins']:2d}/{s['n']:<2d} "
              f"{s['delivered']:6.0f} {s['mean_c2d']:6.2f} {s['late_pct']:6.2f} {s['mean_food_wait']:5.2f} "
              f"{s['travel_min']:7.0f} {s['service_gap_to_lb_pct']:7.2f} {s['runtime_s']:8.0f} {s['seed_std_obj']:8.4f}")
    json.dump({"summary": summary, "per_instance": {f"{n}|{m}": a for (n, m), a in agg.items()}},
              open(f"summary_full_{args.tag}.json", "w"), indent=1)
    if args.md:
        print("\n| Method | Gap vs greedy (95% CI) | Wins | Mean click-to-door | Late % | Food wait | Travel min | Runtime |")
        print("|---|---|---|---|---|---|---|---|")
        for m, s in summary.items():
            print(f"| {m} | {s['mean_gap_pct']:+.2f}% [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] | {s['wins']}/{s['n']} | "
                  f"{s['mean_c2d']:.2f} | {s['late_pct']:.1f} | {s['mean_food_wait']:.2f} | {s['travel_min']:.0f} | {s['runtime_s']:.0f} s |")


if __name__ == "__main__":
    main()
