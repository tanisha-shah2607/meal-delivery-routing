import os
import sys
import json
import math
import numpy as np
from mdrp_data import Instance

ROOT = os.path.join("mdrplib", "public_instances")
OUT = "chunks"
CAL_SEEDS = [0, 1, 2]
TEST_SEEDS = [3, 4, 5, 6, 7, 8, 9]
SIZES = ["o100", "o50", "r50"]


def select_instances():
    """30 instances: 3 per seed (one per size type); s/t/p variants cycle so all appear."""
    out = []
    for seed in range(10):
        for k, size in enumerate(SIZES):
            idx = seed * 3 + k
            s = "s1" if idx % 2 == 0 else "s2"
            t = "t100" if (idx // 2) % 2 == 0 else "t75"
            p = "p100" if (idx // 4) % 2 == 0 else "p125"
            out.append((seed, f"{seed}{size}{t}{s}{p}"))
    return out


def calibrate(names):
    gaps, delays, offs, rests, cours, speeds, rates, cratios = [], [], [], [], [], [], [], [] 
    ps = ds = target = None
    for name in names:
        inst = Instance(os.path.join(ROOT, name))
        o = inst.orders
        gaps += list(np.diff(np.sort(o.placement_time.values)))
        delays += list((o.ready_time - o.placement_time).values)
        rxy = inst.restaurants.loc[o.restaurant.values, ["x", "y"]].values.astype(float)
        cxy = o[["x", "y"]].values.astype(float)
        offs += list(cxy - rxy)
        rests += list(inst.restaurants[["x", "y"]].values.astype(float))
        cours += list(inst.couriers[["x", "y"]].values.astype(float))
        speeds.append(inst.speed)
        rates.append(len(o) / max(1, o.placement_time.max() - o.placement_time.min()))
        cratios.append(float((inst.couriers.off_time - inst.couriers.on_time).sum()) / max(1, o.placement_time.max() - o.placement_time.min()) / rates[-1])
        ps, ds, target = inst.pickup_service, inst.dropoff_service, inst.target_c2d
    return dict(gaps=np.array(gaps, dtype=float), delays=np.array(delays, dtype=float),
                offs=np.array(offs), rests=np.array(rests), cours=np.array(cours),
                speeds=np.array(speeds),rates=np.array(rates), cratio=np.array(np.mean(cratios)), cratios=np.array(cratios),
                ps=np.array(ps), ds=np.array(ds), target=np.array(target))


def real_chunk(inst, name, n, rng):
    o = inst.orders
    pl = o.placement_time.values.astype(int)
    rd = o.ready_time.values.astype(int)
    rxy = inst.restaurants.loc[o.restaurant.values, ["x", "y"]].values.astype(float)
    cxy = o[["x", "y"]].values.astype(float)
    a = int(rng.integers(len(pl)))
    w = 10
    while True:                                   # widen the window until it holds >= n orders
        cand = np.where(np.abs(pl - pl[a]) <= w)[0]
        if len(cand) >= n or w > 5000:
            break
        w += 5
    d = np.hypot(rxy[cand, 0] - rxy[a, 0], rxy[cand, 1] - rxy[a, 1])
    pick = cand[np.argsort(d, kind="stable")[:n]]
    t0 = int(pl[pick].max())
    c = inst.couriers
    on = c[(c.on_time <= t0) & (c.off_time > t0)]
    if len(on) == 0:
        on = c
    dc = np.hypot(on.x.values - rxy[a, 0], on.y.values - rxy[a, 1])
    j = int(np.argmin(dc))
    return {"instance": name, "n": n, "speed": float(inst.speed), "ps": int(inst.pickup_service),
            "ds": int(inst.dropoff_service), "target": float(inst.target_c2d),
            "courier": [float(on.x.values[j]), float(on.y.values[j])],
            "orders": [{"id": str(o.index[k]),
                        "rx": float(rxy[k, 0]), "ry": float(rxy[k, 1]),
                        "cx": float(cxy[k, 0]), "cy": float(cxy[k, 1]),
                        "placement": int(pl[k] - t0), "ready": int(rd[k] - t0)} for k in pick]}

class SyntheticGen:
    """Random batches whose statistics are bootstrapped from the calibration seeds only."""

    def __init__(self, path=os.path.join(OUT, "pools.npz"), seed=0):
        self.P = dict(np.load(path))
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        P, rng = self.P, self.rng
        lam = float(np.exp(rng.uniform(np.log(0.25), np.log(3.5))))   # orders/min, published range 200-3000 orders
        w = 10
        while 2 * w * lam < n:                       # same widening rule as real_chunk
            w += 5
        M = max(n, int(round(2 * w * lam)))          # orders competing to join the batch
        r = P["rests"][rng.integers(len(P["rests"]), size=M)] + rng.normal(0, 150, (M, 2))
        cust = r + P["offs"][rng.integers(len(P["offs"]), size=M)]
        pl_all = np.round(rng.uniform(-w, w, size=M)).astype(int)
        pl_all[0] = 0                                # order 0 is the anchor
        keep = np.argsort(np.hypot(r[:, 0] - r[0, 0], r[:, 1] - r[0, 1]), kind="stable")[:n]
        r, cust, pl = r[keep], cust[keep], pl_all[keep]
        pl = pl - pl.max()                           # decision time = 0
        rd = pl + np.round(P["delays"][rng.integers(len(P["delays"]), size=n)]).astype(int)
        K = max(3, int(round(float(P["cratio"]) * lam)))   # on-shift couriers scale with order rate
        cs = P["cours"][rng.integers(len(P["cours"]), size=K)]
        cour = cs[np.argmin(np.hypot(cs[:, 0] - r[0, 0], cs[:, 1] - r[0, 1]))]
        return {"instance": "synthetic", "n": n, "speed": float(rng.choice(P["speeds"])),
                "ps": int(P["ps"]), "ds": int(P["ds"]), "target": float(P["target"]),
                "courier": [float(cour[0]), float(cour[1])],
                "orders": [{"id": f"s{k}", "rx": float(r[k, 0]), "ry": float(r[k, 1]),
                            "cx": float(cust[k, 0]), "cy": float(cust[k, 1]),
                            "placement": int(pl[k]), "ready": int(rd[k])} for k in range(n)]}
def describe(chs):
    span, delay, trip, cdist = [], [], [], []
    for ch in chs:
        o = ch["orders"]
        span.append(-min(a["placement"] for a in o))
        delay += [a["ready"] - a["placement"] for a in o]
        trip += [math.hypot(a["cx"] - a["rx"], a["cy"] - a["ry"]) for a in o]
        mx = np.mean([a["rx"] for a in o]); my = np.mean([a["ry"] for a in o])
        cdist.append(math.hypot(ch["courier"][0] - mx, ch["courier"][1] - my))
    return {"placement span (min)": np.mean(span), "ready delay (min)": np.mean(delay),
            "restaurant->customer (m)": np.mean(trip), "courier->batch (m)": np.mean(cdist)}


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = select_instances()
    missing = [n for _, n in sel if not os.path.isdir(os.path.join(ROOT, n))]
    if missing:
        print("MISSING instance folders:", missing)
        sys.exit(1)
    cal = [n for s, n in sel if s in CAL_SEEDS]
    test = [n for s, n in sel if s in TEST_SEEDS]
    print(f"{len(sel)} instances: {len(cal)} calibration (seeds {CAL_SEEDS}), "
          f"{len(test)} test (seeds {TEST_SEEDS})")

    pools = calibrate(cal)
    np.savez(os.path.join(OUT, "pools.npz"), **pools)
    print(f"calibration: mean order gap {pools['gaps'].mean():.2f} min | "
          f"median ready delay {np.median(pools['delays']):.0f} min | "
          f"median restaurant->customer {np.median(np.hypot(*pools['offs'].T)):.0f} m | "
          f"speeds {sorted(set(pools['speeds'].tolist()))}")

    insts = {n: Instance(os.path.join(ROOT, n)) for n in test}
    sizes = {n: len(i.orders) for n, i in insts.items()}
    print("test instance sizes (orders): min", min(sizes.values()), "max", max(sizes.values()))
    rng = np.random.default_rng(0)
    gen = SyntheticGen()
    for n in (5, 10, 20):
        chunks = [real_chunk(insts[test[i % len(test)]], test[i % len(test)], n, rng) for i in range(100)]
        with open(os.path.join(OUT, f"test_n{n}.json"), "w") as f:
            json.dump(chunks, f)
        syn = [gen.sample(n) for _ in range(100)]
        print(f"\nn={n}: wrote 100 real test chunks -> chunks/test_n{n}.json")
        dr, ds = describe(chunks), describe(syn)
        for k in dr:
            print(f"   {k:28s} real {dr[k]:8.1f} | synthetic {ds[k]:8.1f}")


if __name__ == "__main__":
    main()