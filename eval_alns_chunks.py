"""Per-courier chunk benchmark: ALNS against the attention model and the baselines on the *same* problems.

This is the like-for-like comparison: every method gets the same 100 real held-out chunks per size (7 unseen
seeds), is scored by the same cost function (chunk_env.route_cost), and is timed on the same machine in the
same run. ALNS is run at several wall-clock budgets per chunk so the comparison can be read at matched time,
with 3 independent run seeds (mean and spread). Statistics follow stats_eval.py: paired bootstrap 95% CI of
the gap to cheapest insertion over chunks, and the spread of that gap across instance seeds.

Usage: python eval_alns_chunks.py [--budgets 10 50 250 1000] [--seeds 0 1 2] [--workers 8]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import chunk_env as E

SIZES = (5, 10, 20)


def tuned_params():
    """Removal fraction and cooling chosen by tune_alns_chunks.py on calibration-seed chunks."""
    if os.path.exists("tune_alns_chunks.json"):
        return json.load(open("tune_alns_chunks.json"))["best"]
    return {}


def alns_job(a):
    n, idx, budget_ms, seed = a
    import alns as A
    ch = E.load_chunks(n)[idx]
    t0 = time.perf_counter()
    seq, iters = A.solve_chunk(ch, time_limit=budget_ms / 1000.0, seed=seed, params=tuned_params())
    ms = 1000 * (time.perf_counter() - t0)
    assert E.valid(ch, seq)
    return n, idx, budget_ms, seed, E.route_cost(ch, seq) / n, ms, iters


def seed_of(ch):
    import re
    return int(re.match(r"\d+", ch["instance"]).group())


def gap_ci(m, ref, rng, B=10000):
    idx = rng.integers(0, len(m), size=(B, len(m)))
    g = 100 * (m[idx].mean(1) / ref[idx].mean(1) - 1)
    return 100 * (m.mean() / ref.mean() - 1), np.percentile(g, 2.5), np.percentile(g, 97.5)


def seed_spread(seeds, m, ref):
    gaps = [100 * (m[seeds == s].mean() / ref[seeds == s].mean() - 1) for s in sorted(set(seeds))]
    return float(np.mean(gaps)), float(np.std(gaps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 50, 250, 1000], help="ms per chunk")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default="attn.pt")
    ap.add_argument("--out", default="results_alns_chunks.json")
    args = ap.parse_args()
    print("ALNS parameters (tuned on calibration seeds):", tuned_params() or "defaults")

    # ALNS first, in parallel (each worker is one single-threaded process)
    jobs = [(n, i, b, s) for n in SIZES for i in range(100) for b in args.budgets for s in args.seeds]
    jobs.sort(key=lambda j: -j[2])
    t0 = time.time()
    with Pool(args.workers) as pool:
        alns_res = pool.map(alns_job, jobs, chunksize=4)
    print(f"ALNS: {len(jobs)} runs in {time.time() - t0:.0f}s")

    import torch
    from eval_attention import attn_routes, load_model
    from local_search import polish
    torch.manual_seed(0)
    model = load_model(args.ckpt)
    rng = np.random.default_rng(0)
    results = {}
    for n in SIZES:
        chs = E.load_chunks(n)
        seeds = np.array([seed_of(c) for c in chs])
        costs, ms, extra = {}, {}, {}

        def run(name, fn):
            t = time.perf_counter()
            seqs = fn()
            ms[name] = 1000 * (time.perf_counter() - t) / len(chs)
            assert all(E.valid(c, s) for c, s in zip(chs, seqs))
            costs[name] = np.array([E.route_cost(c, s) / n for c, s in zip(chs, seqs)])
            return seqs

        run("nearest-nbr", lambda: [E.nn_route(c) for c in chs])
        ins = run("insertion", lambda: [E.insertion_route(c) for c in chs])
        att = run("attention (greedy)", lambda: attn_routes(model, chs)[0])
        run("attention (best of 32)", lambda: attn_routes(model, chs, 32)[0])
        run("attention + polish", lambda: [polish(c, s) for c, s in zip(chs, att)])
        run("insertion + polish", lambda: [polish(c, s) for c, s in zip(chs, ins)])
        ms["attention + polish"] += ms["attention (greedy)"]
        ms["insertion + polish"] += ms["insertion"]
        for b in args.budgets:
            name = f"ALNS ({b} ms)"
            per_seed = np.zeros((len(args.seeds), len(chs)))
            t_ms, its = [], []
            for (nn, i, bb, s, c, m, it) in alns_res:
                if nn == n and bb == b:
                    per_seed[args.seeds.index(s), i] = c
                    t_ms.append(m)
                    its.append(it)
            costs[name] = per_seed.mean(0)
            ms[name] = float(np.mean(t_ms))
            extra[name] = {"run_seed_means": per_seed.mean(1).tolist(),
                           "run_seed_std": float(per_seed.mean(1).std()), "mean_iters": float(np.mean(its))}
        opt = None
        if n == 5:
            t = time.perf_counter()
            opt = np.array([E.route_cost(c, E.optimal_route(c)) / n for c in chs])
            ms["exact optimum"] = 1000 * (time.perf_counter() - t) / len(chs)

        ref = costs["insertion"]
        print(f"\nn={n} orders per courier: {len(chs)} real chunks from {len(set(seeds))} held-out seeds")
        hdr = f"{'method':24s} {'cost/order':>10s} {'vs insertion [95% CI]':>24s} {'across seeds':>15s}"
        hdr += f" {'vs optimum':>10s}" if opt is not None else ""
        print(hdr + f" {'ms/chunk':>9s}")
        results[n] = {}
        for name, c in costs.items():
            g, lo, hi = gap_ci(c, ref, rng)
            sm, ss = seed_spread(seeds, c, ref)
            row = {"cost_per_order": float(c.mean()), "gap_vs_insertion_pct": float(g), "ci95": [float(lo), float(hi)],
                   "seed_mean_gap": sm, "seed_std_gap": ss, "ms_per_chunk": float(ms[name]), **extra.get(name, {})}
            line = f"{name:24s} {c.mean():10.2f} {g:+7.2f}% [{lo:+5.1f},{hi:+5.1f}] {sm:+6.2f} +/- {ss:4.2f}"
            if opt is not None:
                go = 100 * (c.mean() / opt.mean() - 1)
                row["gap_vs_optimum_pct"] = float(go)
                row["at_optimum"] = int((np.abs(c - opt) < 1e-6).sum())
                line += f" {go:+9.2f}%"
            if name in extra:
                line += f" {ms[name]:9.1f}  (run-seed std {extra[name]['run_seed_std']:.3f}, {extra[name]['mean_iters']:.0f} iters)"
            else:
                line += f" {ms[name]:9.1f}"
            print(line)
            results[n][name] = row
        if opt is not None:
            results[n]["exact optimum"] = {"cost_per_order": float(opt.mean()), "ms_per_chunk": ms["exact optimum"]}
            print(f"{'exact optimum':24s} {opt.mean():10.2f} {'':>24s} {'':>15s} {'':>10s} {ms['exact optimum']:9.1f}")
        if n == 20:
            t = time.perf_counter()
            for c in chs[:20]:
                attn_routes(model, [c])
            results[n]["attention single-chunk latency ms"] = 1000 * (time.perf_counter() - t) / 20
    results["alns_params"] = tuned_params()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
