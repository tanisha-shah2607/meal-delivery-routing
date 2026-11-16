import json
import math
import random
import numpy as np
from baselines import W_DELIVERY, W_LATE, W_WAIT, W_TRAVEL

# A stop index s < n is the pickup of order s; s >= n is the delivery of order s - n.


def stop_xy(ch):
    if "_xy" not in ch:
        ch["_xy"] = [(o["rx"], o["ry"]) for o in ch["orders"]] + [(o["cx"], o["cy"]) for o in ch["orders"]]
    return ch["_xy"]


def step(ch, xy, x, y, t, s):
    n = len(ch["orders"])
    o = ch["orders"][s % n]
    nx, ny = xy[s]
    tt = math.ceil(math.hypot(nx - x, ny - y) / ch["speed"])
    arr = t + tt
    if s < n:
        done = max(arr, o["ready"]) + ch["ps"]
        c = W_WAIT * max(0, done - ch["ps"] - o["ready"])
    else:
        done = arr + ch["ds"]
        c2d = done - o["placement"]
        c = W_DELIVERY * c2d + W_LATE * max(0, c2d - ch["target"])
    return nx, ny, done, c + W_TRAVEL * tt, tt


def run_seq(ch, seq):
    xy = stop_xy(ch)
    x, y = ch["courier"]
    t, cost, travel, c2d = 0, 0.0, 0, []
    n = len(ch["orders"])
    for s in seq:
        x, y, t, c, tt = step(ch, xy, x, y, t, s)
        cost += c
        travel += tt
        if s >= n:
            c2d.append(t - ch["orders"][s - n]["placement"])
    return cost, travel, c2d


def route_cost(ch, seq):
    return run_seq(ch, seq)[0]


def valid(ch, seq):
    n = len(ch["orders"])
    pos = {s: i for i, s in enumerate(seq)}
    return len(seq) == 2 * n and len(pos) == 2 * n and all(pos[i] < pos[i + n] for i in range(n))


def random_route(ch, rng):
    n = len(ch["orders"])
    seq, picked = [], set()
    while len(seq) < 2 * n:
        cand = [s for s in range(n) if s not in picked] + [s + n for s in sorted(picked) if s + n not in seq]
        s = rng.choice(cand)
        if s < n:
            picked.add(s)
        seq.append(s)
    return seq


def nn_route(ch):
    n = len(ch["orders"])
    xy = stop_xy(ch)
    x, y = ch["courier"]
    t, seq, picked = 0, [], set()
    while len(seq) < 2 * n:
        cand = [s for s in range(n) if s not in picked] + [s + n for s in sorted(picked) if s + n not in seq]
        s = min(cand, key=lambda k: (math.hypot(xy[k][0] - x, xy[k][1] - y), k))
        if s < n:
            picked.add(s)
        x, y, t, _, _ = step(ch, xy, x, y, t, s)
        seq.append(s)
    return seq


def insertion_route(ch):
    n = len(ch["orders"])
    seq = []
    for i in sorted(range(n), key=lambda k: ch["orders"][k]["placement"]):
        best = None
        for a in range(len(seq) + 1):
            for b in range(a, len(seq) + 1):
                new = seq[:a] + [i] + seq[a:b] + [i + n] + seq[b:]
                c = route_cost(ch, new)
                if best is None or c < best[0]:
                    best = (c, new)
        seq = best[1]
    return seq


def optimal_route(ch):
    """Exact via branch and bound; only for tiny chunks (n <= 6)."""
    n = len(ch["orders"])
    assert n <= 6
    xy = stop_xy(ch)
    best = [float("inf"), None]

    def rec(seq, x, y, t, cost, picked, delivered):
        if cost >= best[0]:
            return
        if len(seq) == 2 * n:
            best[0], best[1] = cost, list(seq)
            return
        opts = []
        for s in range(2 * n):
            if s < n:
                if s in picked:
                    continue
            elif (s - n) not in picked or s in delivered:
                continue
            nx, ny, nt, c, _ = step(ch, xy, x, y, t, s)
            opts.append((cost + c, s, nx, ny, nt))
        opts.sort()
        for nc, s, nx, ny, nt in opts:
            seq.append(s)
            (picked if s < n else delivered).add(s)
            rec(seq, nx, ny, nt, nc, picked, delivered)
            (picked if s < n else delivered).discard(s)
            seq.pop()

    rec([], ch["courier"][0], ch["courier"][1], 0, 0.0, set(), set())
    return best[1]


def summarize(chs, seqs):
    cost, c2d, travel, late, tot = [], [], [], 0, 0
    for ch, seq in zip(chs, seqs):
        assert valid(ch, seq), "invalid route"
        c, tr, cc = run_seq(ch, seq)
        cost.append(c / len(ch["orders"]))
        travel.append(tr)
        c2d += cc
        late += sum(x > ch["target"] for x in cc)
        tot += len(cc)
    return np.mean(cost), np.mean(c2d), 100 * late / tot, np.mean(travel)


def load_chunks(n):
    with open(f"chunks/test_n{n}.json") as f:
        return json.load(f)


if __name__ == "__main__":
    rng = random.Random(0)
    for n in (5, 10, 20):
        chs = load_chunks(n)
        print(f"\nn={n} orders per courier ({len(chs)} real test chunks)")
        print(f"{'method':14s} {'cost/order':>10s} {'click-to-door':>14s} {'late %':>7s} {'travel min':>11s}")
        for name, fn in [("random", lambda c: random_route(c, rng)), ("nearest-nbr", nn_route),
                         ("insertion", insertion_route)]:
            r = summarize(chs, [fn(c) for c in chs])
            print(f"{name:14s} {r[0]:10.2f} {r[1]:14.1f} {r[2]:7.1f} {r[3]:11.1f}")
        if n == 5:
            sub = chs[:30]
            opt = summarize(sub, [optimal_route(c) for c in sub])
            gre = summarize(sub, [insertion_route(c) for c in sub])
            nnr = summarize(sub, [nn_route(c) for c in sub])
            print(f"exact optimum on first 30 chunks: {opt[0]:.2f}/order | insertion {gre[0]:.2f} "
                  f"({100*(gre[0]/opt[0]-1):.1f}% above) | nearest-nbr {nnr[0]:.2f} "
                  f"({100*(nnr[0]/opt[0]-1):.1f}% above)")