import argparse
import json
import re
import time
import numpy as np
import torch

import chunk_env as E
from eval_attention import attn_routes, load_model
from local_search import polish


def seed_of(ch):
    return int(re.match(r"\d+", ch["instance"]).group())


def per_chunk(chs, seqs):
    return np.array([E.route_cost(c, s) / len(c["orders"]) for c, s in zip(chs, seqs)])


def gap_ci(m, ref, rng, B=10000):
    """Mean cost vs reference, in %, with a paired bootstrap 95% interval over chunks."""
    idx = rng.integers(0, len(m), size=(B, len(m)))
    g = 100 * (m[idx].mean(1) / ref[idx].mean(1) - 1)
    return 100 * (m.mean() / ref.mean() - 1), np.percentile(g, 2.5), np.percentile(g, 97.5)


def seed_spread(chs, m, ref):
    """Gap vs reference computed separately on each seed's chunks."""
    seeds = np.array([seed_of(c) for c in chs])
    gaps = [100 * (m[seeds == s].mean() / ref[seeds == s].mean() - 1) for s in sorted(set(seeds))]
    return np.mean(gaps), np.std(gaps), min(gaps), max(gaps), len(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="attn.pt")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = load_model(args.ckpt)
    results = {}

    for n in (5, 10, 20):
        chs = E.load_chunks(n)
        print(f"\nn={n} orders per courier ({len(chs)} real chunks, "
              f"{len(set(seed_of(c) for c in chs))} held-out seeds)")
        costs, ms = {}, {}

        def run(name, fn):
            t0 = time.time()
            seqs = fn()
            ms[name] = 1000 * (time.time() - t0) / len(chs)
            costs[name] = per_chunk(chs, seqs)
            return seqs

        run("nearest-nbr", lambda: [E.nn_route(c) for c in chs])
        ins = run("insertion", lambda: [E.insertion_route(c) for c in chs])
        att = run("attention (greedy)", lambda: attn_routes(model, chs)[0])
        run(f"attention (best of {args.samples})", lambda: attn_routes(model, chs, args.samples)[0])
        run("attention + polish", lambda: [polish(c, s) for c, s in zip(chs, att)])
        run("insertion + polish", lambda: [polish(c, s) for c, s in zip(chs, ins)])
        ms["attention + polish"] += ms["attention (greedy)"]      # include the time to build the starting route
        ms["insertion + polish"] += ms["insertion"]

        ref = costs["insertion"]
        print(f"{'method':26s} {'cost/order':>10s} {'vs insertion (95% CI)':>28s} {'across seeds':>26s} {'ms/chunk':>9s}")
        results[n] = {}
        for name, c in costs.items():
            g, lo, hi = gap_ci(c, ref, rng)
            sm, ss, smin, smax, k = seed_spread(chs, c, ref)
            tag = "" if name == "insertion" else f"{g:+5.1f}% [{lo:+5.1f}, {hi:+5.1f}]"
            sp = "" if name == "insertion" else f"{sm:+5.1f} +/- {ss:3.1f} ({smin:+.1f}..{smax:+.1f})"
            print(f"{name:26s} {c.mean():10.2f} {tag:>28s} {sp:>26s} {ms[name]:9.1f}")
            results[n][name] = {"cost_per_order": float(c.mean()), "gap_vs_insertion_pct": float(g),
                                "ci95": [float(lo), float(hi)], "seed_mean_gap": float(sm),
                                "seed_std_gap": float(ss), "ms_per_chunk": float(ms[name])}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()