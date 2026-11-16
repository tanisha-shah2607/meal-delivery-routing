import argparse
import time
import numpy as np
import torch

import chunk_env
from attention_model import AttentionRouter, make_batch


def load_model(path):
    m = AttentionRouter()
    m.load_state_dict(torch.load(path)["state"])
    m.eval()
    return m


@torch.no_grad()
def attn_routes(model, chs, samples=0):
    """samples=0: greedy decoding. samples=k: sample k routes per chunk and keep the cheapest."""
    b = make_batch(chs)
    if samples == 0:
        seq, _, cost = model(b, greedy=True)
        return seq.tolist(), cost
    best_seq, best_cost = None, None
    for _ in range(samples):
        seq, _, cost = model(b)
        if best_cost is None:
            best_seq, best_cost = seq.clone(), cost.clone()
        else:
            better = cost < best_cost
            best_seq[better] = seq[better]
            best_cost = torch.where(better, cost, best_cost)
    return best_seq.tolist(), best_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="attn.pt")
    ap.add_argument("--samples", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(0)
    model = load_model(args.ckpt)

    for n in (5, 10, 20):
        chs = chunk_env.load_chunks(n)
        print(f"\nn={n} orders per courier ({len(chs)} real test chunks)")
        print(f"{'method':18s} {'cost/order':>10s} {'click-to-door':>14s} {'late %':>7s} {'travel min':>11s} {'ms/chunk':>9s}")
        rows = {}

        def report(name, seqs, secs):
            r = chunk_env.summarize(chs, seqs)
            rows[name] = r
            print(f"{name:18s} {r[0]:10.2f} {r[1]:14.1f} {r[2]:7.1f} {r[3]:11.1f} {1000 * secs / len(chs):9.1f}")

        t0 = time.time(); seqs = [chunk_env.nn_route(c) for c in chs]; report("nearest-nbr", seqs, time.time() - t0)
        t0 = time.time(); seqs = [chunk_env.insertion_route(c) for c in chs]; report("insertion", seqs, time.time() - t0)
        t0 = time.time(); seqs, cost = attn_routes(model, chs); report("attention (greedy)", seqs, time.time() - t0)
        py = np.array([chunk_env.route_cost(c, s) / n for c, s in zip(chs, seqs)])
        assert np.abs(py - cost.numpy()).max() < 1e-6, "model cost disagrees with evaluator"
        t0 = time.time(); seqs, _ = attn_routes(model, chs, args.samples); report(f"attention (best of {args.samples})", seqs, time.time() - t0)

        t0 = time.time()
        for c in chs[:20]:
            attn_routes(model, [c])
        print(f"   single-chunk attention latency (greedy, batch of 1): {1000 * (time.time() - t0) / 20:.1f} ms")
        ref = rows["insertion"][0]
        for k in rows:
            if k.startswith("attention"):
                print(f"   {k}: {100 * (rows[k][0] / ref - 1):+.1f}% vs insertion, "
                      f"{100 * (rows[k][0] / rows['nearest-nbr'][0] - 1):+.1f}% vs nearest-nbr")
        if n == 5:
            sub = chs[:30]
            opt = chunk_env.summarize(sub, [chunk_env.optimal_route(c) for c in sub])[0]
            att = chunk_env.summarize(sub, attn_routes(model, sub)[0])[0]
            print(f"   first 30 chunks: attention greedy is {100 * (att / opt - 1):.1f}% above the exact optimum")


if __name__ == "__main__":
    main()