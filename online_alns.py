"""Online (rolling-horizon) ALNS on full Grubhub MDRP instances.

Orders are revealed at their placement time. The simulation keeps one planned route per courier and moves
through decision times:

  * advance: every stop whose departure falls before the current time is committed (the courier is already
    on the way or done); committed pickups turn their orders into delivery-only requests that must stay on
    that courier,
  * insert: newly placed orders are inserted by cheapest insertion, exactly like the greedy-insertion baseline,
  * re-optimize: every `interval` minutes, ALNS improves the plan of all not-yet-committed stops for
    `budget` seconds, warm-started from the current plan (operator weights carry over between epochs).

mode="event" inserts each order the minute it is placed and runs ALNS every `interval` minutes.
mode="batch" is the literal proposal: new orders wait for the next epoch, are inserted, then ALNS runs.
With lazy=True a courier leaves for a pickup only as late as still arrives by the ready time. The planned
cost is unchanged, but the stop stays open to re-optimization for longer.

Output is the official solution format, so the repository's evaluator checks feasibility for every method.

Usage: python online_alns.py <instance_dir> <out_dir> [--budget 5] [--interval 5] [--mode event] [--seed 0]
"""
import argparse
import json
import math
import os
import time

import numpy as np

import alns as A
from baselines import W_DELIVERY
from mdrp_data import Instance


class LazyCand(dict):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __missing__(self, r):
        v = self[r] = self.fn(r)
        return v


class OnlineALNS:
    def __init__(self, inst_dir, budget=5.0, interval=5, mode="event", lazy=True, k_near=20,
                 seed=0, params=None, max_iters=None):
        self.inst = inst = Instance(inst_dir)
        self.budget, self.interval, self.mode, self.lazy = budget, interval, mode, lazy
        self.k_near, self.max_iters = k_near, max_iters
        o = inst.orders
        self.oid = sorted(o.index, key=lambda x: int(x[1:]))
        self.rest = [o.restaurant[x] for x in self.oid]
        self.place = [int(o.placement_time[x]) for x in self.oid]
        self.ready = [int(o.ready_time[x]) for x in self.oid]
        sx, sy, kind, tw = [], [], [], []
        for r, x in enumerate(self.oid):           # stop 2r = pickup of order r, 2r + 1 = its delivery
            (rx, ry), (cx, cy) = inst.loc[self.rest[r]], inst.loc[x]
            sx += [rx, cx]
            sy += [ry, cy]
            kind += [0, 1]
            tw += [self.ready[r], self.place[r]]
        self.M = A.Model(sx, sy, kind, tw, inst.speed, inst.pickup_service, inst.dropoff_service, inst.target_c2d)
        c = inst.couriers
        self.cid = sorted(c.index, key=lambda x: int(x[1:]))
        self.on = [float(c.on_time[x]) for x in self.cid]
        self.off = [float(c.off_time[x]) for x in self.cid]
        K = len(self.cid)
        self.cx = [float(c.x[x]) for x in self.cid]
        self.cy = [float(c.y[x]) for x in self.cid]
        self.node = list(self.cid)                  # node id where each courier currently is
        self.free = list(self.on)                   # time each courier finishes its last committed stop
        self.plan = [()] * K
        self.pstart = list(self.on)
        self.log = [[] for _ in range(K)]           # committed (stop, dep, arr, done, decided_at)
        self.status = ["unknown"] * len(self.oid)   # unknown / open / onboard / done
        self.carrier = [-1] * len(self.oid)
        self.unassigned = set()
        self.alns = A.ALNS(dict({"patience": 2000}, **(params or {})), seed)
        self.cache = ({}, {})
        self.epochs = []
        self.last_T = 0.0

    # --------------------------------------------------------------------------------------------------
    def node_of(self, s):
        r = s // 2
        return self.rest[r] if s % 2 == 0 else self.oid[r]

    def advance(self, T):
        """Commit every planned stop that departs before T."""
        M = self.M
        sp, ps, ds = M.speed, M.ps, M.ds
        for k, plan in enumerate(self.plan):
            if not plan:
                continue
            t, x, y = self.pstart[k], self.cx[k], self.cy[k]
            n = 0
            for s in plan:
                nx, ny = M.sx[s], M.sy[s]
                tt = math.ceil(math.hypot(nx - x, ny - y) / sp)
                pick = M.kind[s] == 0
                dep = max(t, M.tw[s] - tt) if (pick and self.lazy) else t
                if dep >= T:
                    break
                arr = dep + tt
                done = (max(arr, M.tw[s]) + ps) if pick else arr + ds
                self.log[k].append((s, dep, arr, done, self.last_T))
                r = s // 2
                if pick:
                    self.status[r] = "onboard"
                else:
                    self.status[r] = "done"
                t, x, y = done, nx, ny
                n += 1
            if n:
                self.plan[k] = plan[n:]
                self.free[k], self.cx[k], self.cy[k] = t, x, y
                self.node[k] = self.node_of(plan[n - 1])

    def snapshot(self, T):
        """Static problem at time T: every courier route starts where and when its committed work ends."""
        K = len(self.cid)
        starts = []
        for k in range(K):
            st = max(self.free[k], T, self.on[k])
            self.pstart[k] = st
            starts.append((self.cx[k], self.cy[k], float(st), self.off[k]))
        req_p, req_d = {}, {}
        fixed = {}
        for k, plan in enumerate(self.plan):
            for s in plan:
                if s % 2 == 1:
                    r = s // 2
                    req_d[r] = s
                    if self.status[r] == "onboard":
                        req_p[r] = -1
                        fixed[r] = [k]
                    else:
                        req_p[r] = s - 1
        for r in self.unassigned:
            req_p[r], req_d[r] = 2 * r, 2 * r + 1
        sx_np = np.array([s[0] for s in starts])
        sy_np = np.array([s[1] for s in starts])
        st_np = np.array([s[2] for s in starts])
        off_np = np.array(self.off)
        M = self.M
        where = {}
        for k, plan in enumerate(self.plan):
            for s in plan:
                if s % 2 == 1:
                    where[s // 2] = k

        def cand(r):
            if r in fixed:
                return fixed[r]
            p = 2 * r
            est = st_np + np.ceil(np.hypot(sx_np - M.sx[p], sy_np - M.sy[p]) / M.speed)
            ok = np.maximum(est, M.tw[p]) + M.ps <= off_np
            idx = np.nonzero(ok)[0]
            if self.k_near is not None and len(idx) > self.k_near:
                idx = idx[np.argsort(est[idx], kind="stable")[:self.k_near]]
            out = sorted(int(i) for i in idx)
            k = where.get(r)
            if k is not None and k not in out:
                out.append(k)
            return out

        prob = A.Problem(M, starts, req_p, req_d, LazyCand(cand), cache=self.cache)
        sol = A.build_solution(prob, list(self.plan), self.unassigned)
        return prob, sol

    def insert_new(self, prob, sol, reqs):
        for r in sorted(reqs, key=lambda r: (self.place[r], r)):
            prob.req_p[r], prob.req_d[r] = 2 * r, 2 * r + 1
            A.repair(prob, sol, [r], 1)

    def commit_plan(self, sol):
        self.plan = list(sol.routes)
        self.unassigned = set(sol.unassigned)

    # --------------------------------------------------------------------------------------------------
    def run(self, verbose=False):
        t_wall = time.perf_counter()
        order_by_time = {}
        for r, p in enumerate(self.place):
            order_by_time.setdefault(p, []).append(r)
        first, last = min(self.place), max(self.place)
        epochs = set()
        if self.interval:
            epoch0 = self.interval * math.ceil(first / self.interval)
            epochs = set(range(epoch0, last + self.interval + 1, self.interval))
        times = sorted(epochs | (set(order_by_time) if self.mode == "event" else set()))
        pending = []
        T = None
        for T in times:
            self.advance(T)
            self.last_T = float(T)
            if self.mode == "event":
                new = order_by_time.get(T, [])
            else:
                new = pending + [r for t in range(int(T) - self.interval + 1, int(T) + 1) for r in order_by_time.get(t, [])]
                pending = []
            for r in new:
                self.status[r] = "open"
            if T not in epochs:
                prob, sol = self.snapshot(T)
                self.insert_new(prob, sol, new)
                self.commit_plan(sol)
                continue
            prob, sol = self.snapshot(T)
            self.insert_new(prob, sol, new)
            retry = sorted(sol.unassigned)
            if retry:
                A.repair(prob, sol, retry, 1)
            f0 = sol.objective()
            sunk = sum(W_DELIVERY * (T - self.place[r]) for r in prob.req_d)
            t0 = time.perf_counter()
            if self.budget or self.max_iters:
                sol = self.alns.run(prob, sol, time_limit=self.budget or None, max_iters=self.max_iters, sunk=sunk)
            self.epochs.append({"T": T, "requests": len(prob.req_d), "open": sum(1 for r in prob.req_d if prob.req_p[r] >= 0),
                                "unassigned": len(sol.unassigned), "f_start": f0, "f_end": sol.objective(),
                                "iters": self.alns.stats.get("last_iters", 0) if (self.budget or self.max_iters) else 0,
                                "secs": time.perf_counter() - t0})
            self.commit_plan(sol)
            if verbose and len(self.epochs) % 20 == 0:
                e = self.epochs[-1]
                print(f"  T={T:5.0f} requests {e['requests']:4d} (open {e['open']:3d}) unassigned {e['unassigned']:3d} "
                      f"iters {e['iters']:5d} f {e['f_start']:10.1f} -> {e['f_end']:10.1f}", flush=True)
        # orders still unassigned get one more chance at every later epoch while some courier can take them
        T = (T or 0) + self.interval
        while self.unassigned and self.interval and T <= max(self.off):
            self.advance(T)
            self.last_T = float(T)
            prob, sol = self.snapshot(T)
            A.repair(prob, sol, sorted(sol.unassigned), 1)
            self.commit_plan(sol)
            T += self.interval
        self.advance(float("inf"))                   # no new information arrives: execute every plan
        self.runtime = time.perf_counter() - t_wall
        return self

    # --------------------------------------------------------------------------------------------------
    def write_solution(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        assign, moves, pick, drop, who = [], [], {}, {}, {}
        for k, log in enumerate(self.log):
            c, loc = self.cid[k], self.cid[k]
            for s, dep, arr, done, dec in log:
                node = self.node_of(s)
                if node != loc:
                    moves.append(f"{c} {int(dep)} {loc} {node}")
                o = self.oid[s // 2]
                if s % 2 == 0:
                    pick[o], who[o] = int(done), c
                    assign.append(f"{int(dec)} {int(done)} {c} {o}")
                else:
                    drop[o] = int(done)
                loc = node
        with open(os.path.join(out_dir, "solution_info_assignments.txt"), "w") as f:
            f.write("assignment_time pickup_time courier bundle\n" + "\n".join(assign) + "\n")
        with open(os.path.join(out_dir, "solution_info_couriers.txt"), "w") as f:
            f.write("courier departure_time origin destination\n" + "\n".join(moves) + "\n")
        with open(os.path.join(out_dir, "solution_info_orders.txt"), "w") as f:
            f.write("order placement_time ready_time pickup_time dropoff_time courier\n")
            for r, o in enumerate(self.oid):
                if o in drop:
                    f.write(f"{o} {self.place[r]} {self.ready[r]} {pick[o]} {drop[o]} {who[o]}\n")
                else:
                    f.write(f"{o} {self.place[r]} {self.ready[r]} NA NA NA\n")
        info = {"runtime_s": self.runtime, "epochs": len(self.epochs),
                "alns_iters": int(sum(e["iters"] for e in self.epochs)),
                "alns_secs": float(sum(e["secs"] for e in self.epochs)),
                "mean_requests": float(np.mean([e["requests"] for e in self.epochs])) if self.epochs else 0.0,
                "mean_open": float(np.mean([e["open"] for e in self.epochs])) if self.epochs else 0.0,
                "planned_gain": float(sum(e["f_start"] - e["f_end"] for e in self.epochs)),
                "operator_weights": {"destroy": dict(zip(A.ALNS.DESTROY, self.alns.wd)),
                                     "repair": dict(zip(A.ALNS.REPAIR, self.alns.wr))}}
        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=1)
        return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--budget", type=float, default=5.0, help="ALNS seconds per epoch (0 = no ALNS)")
    ap.add_argument("--interval", type=int, default=5, help="minutes between ALNS epochs")
    ap.add_argument("--mode", choices=["event", "batch"], default="event")
    ap.add_argument("--no_lazy", action="store_true")
    ap.add_argument("--k_near", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xi", type=float, default=A.DEFAULTS["xi"])
    ap.add_argument("--cool", type=float, default=A.DEFAULTS["cool"])
    args = ap.parse_args()
    sim = OnlineALNS(args.instance_dir, args.budget, args.interval, args.mode, not args.no_lazy,
                     args.k_near if args.k_near > 0 else None, args.seed, {"xi": args.xi, "cool": args.cool})
    sim.run(verbose=True)
    info = sim.write_solution(args.out_dir)
    print(json.dumps({k: v for k, v in info.items() if k != "operator_weights"}, indent=1))


if __name__ == "__main__":
    main()
