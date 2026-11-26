"""Tunes ALNS removal fraction and cooling on chunks cut from the 3 *calibration* seeds only (never the test
chunks). Usage: python tune_alns_chunks.py [--budget 250] [--workers 8]"""
import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import chunk_env as E
from mdrp_chunks import ROOT, CAL_SEEDS, real_chunk, select_instances
from mdrp_data import Instance

PATH = os.path.join("chunks", "tune_chunks.json")


def make_tune_chunks(per_size=50):
    if os.path.exists(PATH):
        return json.load(open(PATH))
    cal = [n for s, n in select_instances() if s in CAL_SEEDS]
    insts = {n: Instance(os.path.join(ROOT, n)) for n in cal}
    rng = np.random.default_rng(2024)
    out = {}
    for n in (5, 10, 20):
        out[str(n)] = [real_chunk(insts[cal[i % len(cal)]], cal[i % len(cal)], n, rng) for i in range(per_size)]
    json.dump(out, open(PATH, "w"))
    return out


def job(a):
    n, i, xi, cool, seed, budget = a
    import alns as A
    ch = json.load(open(PATH))[str(n)][i]
    seq, _ = A.solve_chunk(ch, time_limit=budget / 1000, seed=seed, params={"xi": xi, "cool": cool})
    return n, xi, cool, seed, i, E.route_cost(ch, seq) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    chs = make_tune_chunks()
    grid = list(itertools.product((0.1, 0.25, 0.4), (0.1, 0.01, 0.001)))
    jobs = [(n, i, xi, c, s, args.budget) for n in (5, 10, 20) for i in range(len(chs[str(n)]))
            for xi, c in grid for s in (0, 1)]
    t0 = time.time()
    with Pool(args.workers) as p:
        res = p.map(job, jobs, chunksize=8)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s (budget {args.budget} ms, calibration-seed chunks)")
    table = {}
    for n in (5, 10, 20):
        ins = np.mean([E.route_cost(c, E.insertion_route(c)) / n for c in chs[str(n)]])
        print(f"\nn={n}: insertion {ins:.2f}/order")
        print(f"{'xi':>5s} {'cool':>6s} {'cost/order':>11s} {'vs insertion':>13s}")
        for xi, c in grid:
            v = np.mean([r[5] for r in res if r[0] == n and r[1] == xi and r[2] == c])
            table[f"{n}|{xi}|{c}"] = float(v)
            print(f"{xi:5.2f} {c:6.3f} {v:11.3f} {100 * (v / ins - 1):+12.2f}%")
    # pick the setting with the best average relative gap over the three sizes
    ins = {n: np.mean([E.route_cost(c, E.insertion_route(c)) / n for c in chs[str(n)]]) for n in (5, 10, 20)}
    score = {(xi, c): np.mean([table[f"{n}|{xi}|{c}"] / ins[n] for n in (5, 10, 20)]) for xi, c in grid}
    best = min(score, key=score.get)
    print(f"\nbest over sizes: xi={best[0]} cool={best[1]} (mean gap {100 * (score[best] - 1):+.2f}% vs insertion)")
    json.dump({"budget_ms": args.budget, "table": table, "best": {"xi": best[0], "cool": best[1]}},
              open("tune_alns_chunks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
