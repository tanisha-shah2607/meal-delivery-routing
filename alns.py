"""Adaptive Large Neighborhood Search for meal-delivery pickup-and-delivery (Ropke and Pisinger, 2006).

One engine serves two settings:
  * a single courier's batch of orders (the per-courier chunks used for the attention model), and
  * one re-optimization epoch of the online full-instance simulation (many couriers, see online_alns.py).

A *request* is one order: a pickup stop followed later by a delivery stop on the same route. Orders already
picked up become delivery-only requests that must stay on their courier. Destroy operators remove requests
(random, worst, related/Shaw); repair operators reinsert them (greedy, regret-2, regret-3). Operator weights
adapt to past success and a simulated-annealing rule accepts worse solutions.

Costs use the team's objective (baselines.py): delivery time + lateness + food waiting + courier travel.
Insertion is evaluated for every (pickup slot, delivery slot) pair at once with numpy, using the fact that
inserting stops can only delay later stops and that waiting at a restaurant absorbs delay. Every accepted
route is re-simulated exactly, so the search objective always equals the shared cost function.
"""
import math
import random
import time

import numpy as np

from baselines import W_DELIVERY, W_LATE, W_WAIT, W_TRAVEL

INF = float("inf")
PENALTY = 10000.0      # objective cost of leaving one known order unassigned
PY_MAX_STOPS = 9       # routes up to this length use pure-Python insertion, longer ones numpy
LN2 = math.log(2.0)

DEFAULTS = dict(
    xi=0.25,           # removal fraction: q is drawn from [1, max(4, xi * #requests)] (R&P: [4, 0.4 n])
    q_max=30,          # hard cap on q
    w_start=0.05,      # start temperature: a solution 5% worse is accepted with prob. 0.5 (R&P)
    cool=0.1,          # cooling: temperature falls geometrically to cool * T0 at the end of the budget
    segment=50,        # iterations per weight-update segment
    reaction=0.1,      # weight reaction factor r
    sigma=(33, 9, 13), # scores: new global best / better than current / accepted worse (R&P)
    p_worst=3.0,       # randomization of worst removal
    p_shaw=6.0,        # randomization of related removal
    patience=None,     # stop early after this many iterations without a new best (None = use the budget)
)


class Model:
    """Stop data. Stop s: location (sx[s], sy[s]), kind[s] (0 pickup, 1 delivery), and tw[s] = ready time
    for a pickup or placement time for a delivery."""

    def __init__(self, sx, sy, kind, tw, speed, ps, ds, target):
        self.sx = [float(v) for v in sx]
        self.sy = [float(v) for v in sy]
        self.kind = [int(v) for v in kind]
        self.tw = [float(v) for v in tw]
        self.speed, self.ps, self.ds, self.target = float(speed), float(ps), float(ds), float(target)

    def tt(self, x1, y1, x2, y2):
        return math.ceil(math.hypot(x2 - x1, y2 - y1) / self.speed)


def simulate(M, start, stops):
    """Exact cost and feasibility of serving `stops` from start = (x, y, t, off). Pickups must finish by off."""
    x, y, t, off = start
    sx, sy, kind, tw = M.sx, M.sy, M.kind, M.tw
    sp, ps, ds, tg = M.speed, M.ps, M.ds, M.target
    cost, ok = 0.0, True
    for s in stops:
        nx, ny = sx[s], sy[s]
        tt = math.ceil(math.hypot(nx - x, ny - y) / sp)
        arr = t + tt
        if kind[s] == 0:
            r = tw[s]
            c = W_WAIT * (arr - r) if arr > r else 0.0
            t = (arr if arr > r else r) + ps
            if t > off:
                ok = False
        else:
            t = arr + ds
            c2d = t - tw[s]
            c = W_DELIVERY * c2d + (W_LATE * (c2d - tg) if c2d > tg else 0.0)
        cost += c + W_TRAVEL * tt
        x, y = nx, ny
    return cost, ok


class RouteEval:
    """Schedule of one route plus fast evaluation of every way to insert a request into it."""

    def __init__(self, M, start, stops):
        self.stops = stops
        L = self.L = len(stops)
        x, y, t, off = start
        self.off = off
        sx, sy, kind, tw = M.sx, M.sy, M.kind, M.tw
        sp, ps, ds, tg = M.speed, M.ps, M.ds, M.target
        arr, done, wl, ttl = [0.0] * L, [0.0] * L, [0.0] * L, [0.0] * L
        pc, pt, px, py = [0.0] * (L + 1), [float(t)] * (L + 1), [x] * (L + 1), [y] * (L + 1)
        cost, ok = 0.0, True
        for k, s in enumerate(stops):
            nx, ny = sx[s], sy[s]
            tt = math.ceil(math.hypot(nx - x, ny - y) / sp)
            a = t + tt
            if kind[s] == 0:
                r = tw[s]
                c = W_WAIT * (a - r) if a > r else 0.0
                wl[k] = r - a if r > a else 0.0
                t = (a if a > r else r) + ps
                if t > off:
                    ok = False
            else:
                t = a + ds
                c2d = t - tw[s]
                c = W_DELIVERY * c2d + (W_LATE * (c2d - tg) if c2d > tg else 0.0)
            cost += c + W_TRAVEL * tt
            arr[k], done[k], ttl[k] = a, t, tt
            x, y = nx, ny
            pc[k + 1], pt[k + 1], px[k + 1], py[k + 1] = cost, t, x, y
        self.cost, self.ok = cost, ok
        self._raw = (arr, done, wl, ttl, pc, pt, px, py)
        self._np = False
        self.G = None

    def _arrays(self, M):
        if self._np:
            return
        arr, done, wl, ttl, pc, pt, px, py = self._raw
        st = self.stops
        self.arr, self.done, self.w, self.tt = (np.array(v) for v in (arr, done, wl, ttl))
        self.pc, self.pt, self.px, self.py = (np.array(v) for v in (pc, pt, px, py))
        self.sx = np.array([M.sx[s] for s in st])
        self.sy = np.array([M.sy[s] for s in st])
        self.tw = np.array([M.tw[s] for s in st])
        self.ispick = np.array([M.kind[s] == 0 for s in st], dtype=bool)
        self.Wc = np.concatenate([[0.0], np.cumsum(self.w)])
        self._np = True

    def _dsvc(self, M, S):
        """Change in service cost (wait, delivery, lateness) of each stop when its arrival is delayed by S."""
        a, tw = self.arr[None, :], self.tw[None, :]
        pick = W_WAIT * (np.maximum(0.0, a + S - tw) - np.maximum(0.0, a - tw))
        c2d = self.done[None, :] - tw
        tg = M.target
        dlv = W_DELIVERY * S + W_LATE * (np.maximum(0.0, c2d + S - tg) - np.maximum(0.0, c2d - tg))
        return np.where(self.ispick[None, :], pick, dlv)

    def _build_G(self, M, D):
        """G[k, d] = cost of stops k.. (excluding stop k's incoming travel) when stop k's arrival is delayed by d.

        A delay passes unchanged through every stop except a pickup where the courier would wait for the food,
        which absorbs up to its waiting slack w. So the table is a reverse cumulative sum between such pickups."""
        L = self.L
        sh = np.arange(D + 1, dtype=float)[None, :]
        tw, w = self.tw[:, None], self.w[:, None]
        pick = W_WAIT * np.maximum(0.0, self.arr[:, None] + sh - tw)
        c2d = (self.done[:, None] - tw) + sh
        svc = np.where(self.ispick[:, None], pick,
                       W_DELIVERY * c2d + W_LATE * np.maximum(0.0, c2d - M.target))
        if self.off < INF:
            late_pick = self.ispick[:, None] & (self.done[:, None] + np.maximum(0.0, sh - w) > self.off)
            svc = np.where(late_pick, INF, svc)
        if L > 1:
            svc[:-1] += W_TRAVEL * self.tt[1:, None]
        G = np.zeros((L + 1, D + 1))
        hi = L                                            # G[hi] is already final
        for b in [k for k in range(L) if self.w[k] > 0][::-1] + [-1]:
            if hi - b > 1:                                # stops b+1 .. hi-1 pass the delay unchanged
                seg = svc[b + 1:hi]
                G[b + 1:hi] = np.cumsum(seg[::-1], axis=0)[::-1] + G[hi][None, :]
            if b >= 0:
                out = np.maximum(0.0, sh[0] - self.w[b]).astype(np.int64)
                G[b] = svc[b] + G[b + 1, out]
            hi = b
        self.G, self.D = G, D

    def _G_at(self, M, k, d):
        idx = np.maximum(d, 0.0).astype(np.int64)
        need = int(idx.max()) if idx.size else 0
        if self.G is None or need > self.D:
            self._build_G(M, max(need, 2 * self.D if self.G is not None else 256))
        return self.G[k, idx]

    def insert_full(self, M, p, d):
        """Route cost after inserting pickup p before stop i and delivery d before stop j (j >= i, d after p
        when j == i). Returns an (L+1, L+1) array, inf where j < i or the result is infeasible."""
        self._arrays(M)
        L, sp = self.L, M.speed
        rx, ry, rdy = M.sx[p], M.sy[p], M.tw[p]
        cx, cy, plc = M.sx[d], M.sy[d], M.tw[d]
        tg = M.target
        tP = np.ceil(np.hypot(self.px - rx, self.py - ry) / sp)
        arrP = self.pt + tP
        doneP = np.maximum(arrP, rdy) + M.ps
        costP = W_WAIT * np.maximum(0.0, arrP - rdy) + W_TRAVEL * tP
        costP = np.where(doneP > self.off, INF, costP)
        tPD = math.ceil(math.hypot(cx - rx, cy - ry) / sp)
        doneD = doneP + tPD + M.ds
        c2d = doneD - plc
        diag = self.pc + costP + W_DELIVERY * c2d + W_LATE * np.maximum(0.0, c2d - tg) + W_TRAVEL * tPD
        out = np.full((L + 1, L + 1), INF)
        if L == 0:
            out[0, 0] = diag[0]
            return out
        ar = np.arange(L)
        tDn = np.ceil(np.hypot(self.sx - cx, self.sy - cy) / sp)          # stop k <-> delivery
        diag[:L] += W_TRAVEL * tDn + self._G_at(M, ar, doneD[:L] + tDn - self.arr)
        out[np.arange(L + 1), np.arange(L + 1)] = diag
        # delivery after at least one original stop: rows i (pickup slot), columns m = j - 1
        tPn = np.ceil(np.hypot(self.sx - rx, self.sy - ry) / sp)          # pickup -> stop i
        di = doneP[:L] + tPn - self.arr                                    # delay entering stop i
        tri = ar[None, :] >= ar[:, None]
        S = np.maximum(0.0, di[:, None] - (self.Wc[None, :L] - self.Wc[:L, None]))
        so = np.maximum(0.0, S - self.w[None, :])                          # delay leaving stop m
        cum = np.cumsum(np.where(tri, self._dsvc(M, S), 0.0), axis=1)
        bad = np.logical_or.accumulate(tri & self.ispick[None, :] & (self.done[None, :] + so > self.off), axis=1)
        doneD2 = self.done[None, :] + so + tDn[None, :] + M.ds
        c2d2 = doneD2 - plc
        tot = (self.pc[None, 1:] + (costP[:L] + W_TRAVEL * (tPn - self.tt))[:, None] + cum
               + W_DELIVERY * c2d2 + W_LATE * np.maximum(0.0, c2d2 - tg) + W_TRAVEL * tDn[None, :])
        if L > 1:
            d2 = doneD2[:, :L - 1] + tDn[None, 1:] - self.arr[None, 1:]
            kk = np.broadcast_to(ar[None, 1:], d2.shape)
            tot[:, :L - 1] += W_TRAVEL * tDn[None, 1:] + self._G_at(M, kk, d2)
        block = out[:L, 1:]
        ok = tri & ~bad
        block[ok] = tot[ok]
        return out

    # Short routes (the usual case in full instances): plain enumeration from the stored prefix states is
    # faster than numpy's per-call overhead. Every per-stop cost is >= 0, so partial costs prune the search.
    def _tail(self, M, k0, x, y, t, c, best):
        sx, sy, kind, tw, stops = M.sx, M.sy, M.kind, M.tw, self.stops
        sp, ps, ds, tg, off = M.speed, M.ps, M.ds, M.target, self.off
        for k in range(k0, self.L):
            if c >= best:
                return INF
            s = stops[k]
            nx, ny = sx[s], sy[s]
            tt = math.ceil(math.hypot(nx - x, ny - y) / sp)
            a = t + tt
            if kind[s] == 0:
                r = tw[s]
                if a > r:
                    c += W_WAIT * (a - r)
                    t = a + ps
                else:
                    t = r + ps
                if t > off:
                    return INF
            else:
                t = a + ds
                c2d = t - tw[s]
                c += W_DELIVERY * c2d + (W_LATE * (c2d - tg) if c2d > tg else 0.0)
            c += W_TRAVEL * tt
            x, y = nx, ny
        return c

    def best_full_py(self, M, p, d):
        """(best route cost, i, j) over all insertions of pickup p / delivery d, or (inf, -1, -1)."""
        L, stops = self.L, self.stops
        _, _, _, _, pc, pt, px, py = self._raw
        sx, sy, kind, tw = M.sx, M.sy, M.kind, M.tw
        sp, ps, ds, tg, off = M.speed, M.ps, M.ds, M.target, self.off
        rx, ry, rdy = sx[p], sy[p], tw[p]
        cx, cy, plc = sx[d], sy[d], tw[d]
        best, bi, bj = INF, -1, -1
        for i in range(L + 1):
            x, y, t, c = px[i], py[i], pt[i], pc[i]
            tt = math.ceil(math.hypot(rx - x, ry - y) / sp)
            a = t + tt
            if a > rdy:
                c += W_WAIT * (a - rdy)
                t = a + ps
            else:
                t = rdy + ps
            if t > off:
                continue
            c += W_TRAVEL * tt
            x, y = rx, ry
            for j in range(i, L + 1):
                if c >= best:
                    break
                tt = math.ceil(math.hypot(cx - x, cy - y) / sp)
                t2 = t + tt + ds
                c2d = t2 - plc
                c2 = c + W_DELIVERY * c2d + (W_LATE * (c2d - tg) if c2d > tg else 0.0) + W_TRAVEL * tt
                c2 = self._tail(M, j, cx, cy, t2, c2, best)
                if c2 < best:
                    best, bi, bj = c2, i, j
                if j == L:
                    break
                s = stops[j]                                  # advance past stop j: delivery goes after it
                nx, ny = sx[s], sy[s]
                tt = math.ceil(math.hypot(nx - x, ny - y) / sp)
                a = t + tt
                if kind[s] == 0:
                    r = tw[s]
                    if a > r:
                        c += W_WAIT * (a - r)
                        t = a + ps
                    else:
                        t = r + ps
                    if t > off:
                        break
                else:
                    t = a + ds
                    c2d = t - tw[s]
                    c += W_DELIVERY * c2d + (W_LATE * (c2d - tg) if c2d > tg else 0.0)
                c += W_TRAVEL * tt
                x, y = nx, ny
        return best, bi, bj

    def best_delivery_py(self, M, d):
        L = self.L
        _, _, _, _, pc, pt, px, py = self._raw
        cx, cy, plc = M.sx[d], M.sy[d], M.tw[d]
        sp, ds, tg = M.speed, M.ds, M.target
        best, bj = INF, -1
        for j in range(L + 1):
            tt = math.ceil(math.hypot(cx - px[j], cy - py[j]) / sp)
            t = pt[j] + tt + ds
            c2d = t - plc
            c = pc[j] + W_DELIVERY * c2d + (W_LATE * (c2d - tg) if c2d > tg else 0.0) + W_TRAVEL * tt
            c = self._tail(M, j, cx, cy, t, c, best)
            if c < best:
                best, bj = c, j
        return best, bj

    def insert_delivery(self, M, d):
        """Route cost after inserting a delivery-only stop before stop j, for j = 0..L."""
        self._arrays(M)
        L, sp = self.L, M.speed
        cx, cy, plc = M.sx[d], M.sy[d], M.tw[d]
        tD = np.ceil(np.hypot(self.px - cx, self.py - cy) / sp)
        doneD = self.pt + tD + M.ds
        c2d = doneD - plc
        cost = self.pc + W_DELIVERY * c2d + W_LATE * np.maximum(0.0, c2d - M.target) + W_TRAVEL * tD
        if L:
            tDn = np.ceil(np.hypot(self.sx - cx, self.sy - cy) / sp)
            cost[:L] += W_TRAVEL * tDn + self._G_at(M, np.arange(L), doneD[:L] + tDn - self.arr)
        return cost


class Problem:
    """A static snapshot: route start states, requests and the routes each request may use."""

    def __init__(self, M, starts, req_p, req_d, cand, cache=None):
        self.M = M
        self.starts = starts            # route k -> (x, y, t, off)
        self.req_p = req_p              # request -> pickup stop, or -1 if already picked up
        self.req_d = req_d              # request -> delivery stop
        self.cand = cand                # request -> list of route ids it may be inserted into
        self.single = len(starts) == 1
        # caches keyed by (start state, stops) so they stay valid across epochs of the online simulation
        self._ev, self._ins = cache if cache is not None else ({}, {})

    def ev(self, k, stops):
        key = (self.starts[k], stops)
        e = self._ev.get(key)
        if e is None:
            if len(self._ev) > 50000:
                self._ev.clear()
            e = self._ev[key] = RouteEval(self.M, self.starts[k], stops)
        return e

    def options(self, r, k, stops):
        """Insertion options of request r into route k as a sorted list of (delta, k, i, j). Multi-route
        problems keep only the best slot; single-route problems keep the best slot for the 3 best distinct
        pickup positions, which is what regret-k compares there."""
        key = (r, self.req_p[r], k, self.starts[k], stops)
        o = self._ins.get(key)
        if o is not None:
            return o
        if len(self._ins) > 200000:
            self._ins.clear()
        e = self.ev(k, stops)
        p, d = self.req_p[r], self.req_d[r]
        if not self.single and e.L <= PY_MAX_STOPS:
            if p < 0:
                c, j = e.best_delivery_py(self.M, d)
                i = -1
            else:
                c, i, j = e.best_full_py(self.M, p, d)
            o = [] if c == INF else [(c - e.cost, k, i, j)]
        elif p < 0:
            c = e.insert_delivery(self.M, d) - e.cost
            j = int(np.argmin(c))
            o = [] if not np.isfinite(c[j]) else [(float(c[j]), k, -1, j)]
        else:
            c = e.insert_full(self.M, p, d) - e.cost
            if self.single:
                jbest = np.argmin(c, axis=1)
                row = c[np.arange(c.shape[0]), jbest]
                order = np.argsort(row, kind="stable")[:3]
                o = [(float(row[i]), k, int(i), int(jbest[i])) for i in order if np.isfinite(row[i])]
            else:
                f = int(np.argmin(c))
                i, j = divmod(f, c.shape[1])
                o = [] if not np.isfinite(c[i, j]) else [(float(c[i, j]), k, i, j)]
        self._ins[key] = o
        return o


class Solution:
    __slots__ = ("routes", "where", "unassigned", "costs", "total")

    def __init__(self, routes, where, unassigned, costs):
        self.routes, self.where, self.unassigned, self.costs = routes, where, unassigned, costs
        self.total = sum(costs)

    def copy(self):
        s = Solution.__new__(Solution)
        s.routes, s.where, s.unassigned, s.costs = list(self.routes), dict(self.where), set(self.unassigned), list(self.costs)
        s.total = self.total
        return s

    def objective(self):
        return self.total + PENALTY * len(self.unassigned)

    def key(self):
        return hash(tuple(self.routes))


def build_solution(prob, routes, unassigned=()):
    """Wraps route stop lists into a Solution (computing exact costs)."""
    routes = [tuple(r) for r in routes]
    where = {}
    stop_req = {}
    for r in prob.req_d:
        stop_req[prob.req_d[r]] = r
    for k, st in enumerate(routes):
        for s in st:
            if s in stop_req:
                where[stop_req[s]] = k
    costs = [prob.ev(k, st).cost if st else 0.0 for k, st in enumerate(routes)]
    return Solution(routes, where, set(unassigned), costs)


def _set_route(prob, sol, k, stops):
    sol.routes[k] = stops
    c = prob.ev(k, stops).cost if stops else 0.0
    sol.total += c - sol.costs[k]
    sol.costs[k] = c


def remove_request(prob, sol, r):
    k = sol.where.pop(r)
    p, d = prob.req_p[r], prob.req_d[r]
    _set_route(prob, sol, k, tuple(s for s in sol.routes[k] if s != p and s != d))


def insert_request(prob, sol, r, k, i, j):
    st = sol.routes[k]
    p, d = prob.req_p[r], prob.req_d[r]
    new = st[:j] + (d,) + st[j:] if p < 0 else st[:i] + (p,) + st[i:j] + (d,) + st[j:]
    _set_route(prob, sol, k, new)
    sol.where[r] = k


# ----------------------------------------------------------------------------------------------- operators
def random_removal(prob, sol, q, rng):
    reqs = list(sol.where)
    out = rng.sample(reqs, min(q, len(reqs)))
    for r in out:
        remove_request(prob, sol, r)
    return out


def worst_removal(prob, sol, q, rng, p_worst):
    """Removes requests whose removal saves the most cost (randomized as in R&P)."""
    M = prob.M
    gains = []
    for r, k in sol.where.items():
        p, d = prob.req_p[r], prob.req_d[r]
        rest = tuple(s for s in sol.routes[k] if s != p and s != d)
        gains.append((sol.costs[k] - simulate(M, prob.starts[k], rest)[0], r))
    gains.sort(reverse=True)
    lst = [r for _, r in gains]
    out = []
    while lst and len(out) < q:
        idx = int(len(lst) * rng.random() ** p_worst)
        out.append(lst.pop(idx))
    for r in out:
        remove_request(prob, sol, r)
    return out


def shaw_removal(prob, sol, q, rng, p_shaw):
    """Related removal: requests close in pickup/delivery location and in pickup/delivery time."""
    M = prob.M
    reqs = list(sol.where)
    if not reqs:
        return []
    feat, by_route = {}, {}
    for r in reqs:
        by_route.setdefault(sol.where[r], []).append(r)
    for k, rs in by_route.items():
        e = prob.ev(k, sol.routes[k])
        pos = {s: n for n, s in enumerate(e.stops)}
        arr = e._raw[0]
        for r in rs:
            p, d = prob.req_p[r], prob.req_d[r]
            dd = pos[d]
            if p >= 0:
                pp = pos[p]
                feat[r] = (M.sx[p], M.sy[p], M.sx[d], M.sy[d], arr[pp], arr[dd])
            else:
                feat[r] = (M.sx[d], M.sy[d], M.sx[d], M.sy[d], arr[dd], arr[dd])
    F = np.array([feat[r] for r in reqs])
    dn = max(1.0, float(np.ptp(F[:, [0, 2]]) + np.ptp(F[:, [1, 3]])))
    tn = max(1.0, float(np.ptp(F[:, 4:])))
    remaining = list(range(len(reqs)))
    first = remaining.pop(rng.randrange(len(remaining)))
    chosen = [first]
    while remaining and len(chosen) < q:
        a = F[rng.choice(chosen)]
        R = F[remaining]
        rel = (9.0 * (np.hypot(R[:, 0] - a[0], R[:, 1] - a[1]) + np.hypot(R[:, 2] - a[2], R[:, 3] - a[3])) / dn
               + 3.0 * (np.abs(R[:, 4] - a[4]) + np.abs(R[:, 5] - a[5])) / tn)
        order = np.argsort(rel, kind="stable")
        idx = int(len(remaining) * rng.random() ** p_shaw)
        chosen.append(remaining.pop(int(order[idx])))
    out = [reqs[i] for i in chosen]
    for r in out:
        remove_request(prob, sol, r)
    return out


def repair(prob, sol, reqs, k_regret, rng=None):
    """Greedy (k_regret=1) or regret-k insertion of `reqs`. Requests with no feasible slot stay unassigned."""
    pending = [r for r in reqs]
    sol.unassigned.difference_update(pending)
    opts = {}
    for r in pending:
        opts[r] = {k: prob.options(r, k, sol.routes[k]) for k in prob.cand[r]}
    BIG = 1e7
    while pending:
        best, best_key = None, None
        for r in pending:
            flat = sorted(x for v in opts[r].values() for x in v)
            if not flat:
                continue
            if k_regret <= 1:
                key = (flat[0][0],)
            else:
                reg = sum((flat[h][0] if h < len(flat) else BIG) - flat[0][0] for h in range(1, k_regret))
                key = (-reg, flat[0][0])
            if best_key is None or key < best_key:
                best, best_key, best_opt = r, key, flat[0]
        if best is None:
            break
        _, k, i, j = best_opt
        insert_request(prob, sol, best, k, i, j)
        pending.remove(best)
        st = sol.routes[k]
        for r in pending:
            if k in opts[r]:
                opts[r][k] = prob.options(r, k, st)
    sol.unassigned.update(pending)


# ---------------------------------------------------------------------------------------------- main loop
class ALNS:
    """Adaptive operator selection + simulated annealing. Weights persist across calls to run(), which is how
    the online version carries learning from one epoch to the next."""

    DESTROY = ("random", "worst", "related")
    REPAIR = ("greedy", "regret2", "regret3")

    def __init__(self, params=None, seed=0):
        self.P = dict(DEFAULTS, **(params or {}))
        self.rng = random.Random(seed)
        self.wd = [1.0] * len(self.DESTROY)
        self.wr = [1.0] * len(self.REPAIR)
        self.stats = {"iters": 0}

    def _pick(self, w):
        x = self.rng.random() * sum(w)
        for i, v in enumerate(w):
            x -= v
            if x <= 0:
                return i
        return len(w) - 1

    def _destroy(self, op, prob, sol, q):
        if op == 0:
            return random_removal(prob, sol, q, self.rng)
        if op == 1:
            return worst_removal(prob, sol, q, self.rng, self.P["p_worst"])
        return shaw_removal(prob, sol, q, self.rng, self.P["p_shaw"])

    def run(self, prob, sol, time_limit=None, max_iters=None, sunk=0.0):
        """Improves `sol` in place of a copy; returns the best solution found. `sunk` is the part of the
        objective no decision can change (used only to scale the start temperature)."""
        P, rng = self.P, self.rng
        t0 = time.perf_counter()
        cur, best = sol, sol.copy()
        f_cur = f_best = cur.objective()
        n_req = len(prob.req_d)
        if n_req == 0 or (time_limit is None and max_iters is None):
            return best
        T0 = max(1e-6, P["w_start"] * max(1.0, cur.total - sunk) / LN2)
        seen = {cur.key()}
        sd, sr = [0.0] * len(self.wd), [0.0] * len(self.wr)
        nd, nr = [0] * len(self.wd), [0] * len(self.wr)
        it = last_best = 0
        s1, s2, s3 = P["sigma"]
        while True:
            if P["patience"] is not None and it - last_best >= P["patience"]:
                break
            el = time.perf_counter() - t0
            prog = 0.0
            if time_limit is not None:
                if el >= time_limit:
                    break
                prog = el / time_limit
            if max_iters is not None:
                if it >= max_iters:
                    break
                prog = max(prog, it / max_iters)
            T = T0 * P["cool"] ** prog
            n_in = len(cur.where)
            if n_in == 0 and not cur.unassigned:
                break
            qmax = min(P["q_max"], max(min(4, n_in), int(round(P["xi"] * n_in))))
            q = rng.randint(1, qmax) if n_in else 0
            a, b = self._pick(self.wd), self._pick(self.wr)
            new = cur.copy()
            removed = self._destroy(a, prob, new, q) if q else []
            repair(prob, new, removed + list(new.unassigned), b + 1)
            f_new = new.objective()
            it += 1
            nd[a] += 1
            nr[b] += 1
            key = new.key()
            fresh = key not in seen
            score = 0
            if f_new < f_best - 1e-9:
                best, f_best, score, last_best = new.copy(), f_new, s1, it
                cur, f_cur = new, f_new
            elif f_new < f_cur - 1e-9:
                if fresh:
                    score = s2
                cur, f_cur = new, f_new
            elif f_new <= f_cur + 1e-9 or rng.random() < math.exp(-(f_new - f_cur) / T):
                if fresh and f_new > f_cur + 1e-9:
                    score = s3
                cur, f_cur = new, f_new
            if fresh:
                seen.add(key)
            sd[a] += score
            sr[b] += score
            if it % P["segment"] == 0:
                rr = P["reaction"]
                for w, s, n in ((self.wd, sd, nd), (self.wr, sr, nr)):
                    for h in range(len(w)):
                        if n[h]:
                            w[h] = max(0.05, (1 - rr) * w[h] + rr * s[h] / n[h])
                        s[h], n[h] = 0.0, 0
        self.stats["iters"] += it
        self.stats["last_iters"] = it
        return best


# ------------------------------------------------------------------------------------- single-courier chunks
def chunk_model(ch):
    """Stops follow chunk_env's indexing: s < n is the pickup of order s, s >= n its delivery."""
    o = ch["orders"]
    n = len(o)
    sx = [a["rx"] for a in o] + [a["cx"] for a in o]
    sy = [a["ry"] for a in o] + [a["cy"] for a in o]
    tw = [a["ready"] for a in o] + [a["placement"] for a in o]
    M = Model(sx, sy, [0] * n + [1] * n, tw, ch["speed"], ch["ps"], ch["ds"], ch["target"])
    start = (float(ch["courier"][0]), float(ch["courier"][1]), 0.0, INF)
    req = list(range(n))
    return Problem(M, [start], {r: r for r in req}, {r: r + n for r in req}, {r: [0] for r in req})


def chunk_insertion(prob):
    """Cheapest insertion in placement order: the shared 'insertion' baseline, evaluated with RouteEval."""
    n = len(prob.req_d)
    order = sorted(range(n), key=lambda r: prob.M.tw[prob.req_d[r]])
    st = ()
    for r in order:
        e = prob.ev(0, st)
        c = e.insert_full(prob.M, prob.req_p[r], prob.req_d[r])
        i, j = divmod(int(np.argmin(c)), c.shape[1])
        st = st[:i] + (r,) + st[i:j] + (r + n,) + st[j:]
    return st


def solve_chunk(ch, time_limit=None, max_iters=None, seed=0, params=None, init=None):
    """ALNS for one courier's batch. Starts from cheapest insertion (or `init`, a stop sequence)."""
    t_start = time.perf_counter()
    prob = chunk_model(ch)
    st = tuple(init) if init is not None else chunk_insertion(prob)
    sol = build_solution(prob, [st])
    alns = ALNS(params, seed)
    rem = None if time_limit is None else max(0.0, time_limit - (time.perf_counter() - t_start))
    best = alns.run(prob, sol, time_limit=rem, max_iters=max_iters)
    return list(best.routes[0]), alns.stats["iters"]
