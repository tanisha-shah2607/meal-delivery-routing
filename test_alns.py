"""Correctness checks for the ALNS engine. Run: python test_alns.py"""
import random

import numpy as np

import alns as A
import chunk_env as E


def brute_full(M, start, stops, p, d):
    L = len(stops)
    out = np.full((L + 1, L + 1), A.INF)
    for i in range(L + 1):
        for j in range(i, L + 1):
            c, ok = A.simulate(M, start, stops[:i] + (p,) + stops[i:j] + (d,) + stops[j:])
            if ok:
                out[i, j] = c
    return out


def brute_delivery(M, start, stops, d):
    out = np.full(len(stops) + 1, A.INF)
    for j in range(len(stops) + 1):
        c, ok = A.simulate(M, start, stops[:j] + (d,) + stops[j:])
        if ok:
            out[j] = c
    return out


def same(a, b):
    fa, fb = np.isfinite(a), np.isfinite(b)
    return bool((fa == fb).all() and np.allclose(a[fa], b[fb], atol=1e-7))


def test_simulate_matches_chunk_env():
    rng = random.Random(1)
    for n in (5, 10, 20):
        for ch in E.load_chunks(n)[:30]:
            prob = A.chunk_model(ch)
            seq = E.random_route(ch, rng)
            c, ok = A.simulate(prob.M, prob.starts[0], seq)
            assert ok and abs(c - E.route_cost(ch, seq)) < 1e-7
            assert abs(A.RouteEval(prob.M, prob.starts[0], tuple(seq)).cost - c) < 1e-7
    print("simulate == chunk_env.route_cost: OK")


def test_insertion_matches_brute_force():
    rng = random.Random(2)
    checked = 0
    for n in (5, 10, 20):
        for ch in E.load_chunks(n)[:25]:
            prob = A.chunk_model(ch)
            M = prob.M
            seq = E.random_route(ch, rng)
            r = rng.randrange(n)
            base = tuple(s for s in seq if s not in (r, r + n))
            for off in (A.INF, None):
                x, y, t, _ = prob.starts[0]
                t = float(rng.randint(-30, 30))
                if off is None:      # a shift end that binds somewhere in the route
                    full = A.RouteEval(M, (x, y, t, A.INF), base)
                    picks = [full._raw[1][k] for k, s in enumerate(base) if M.kind[s] == 0]
                    off = float(rng.choice(picks)) + rng.randint(0, 15) if picks else A.INF
                start = (x, y, t, off)
                e = A.RouteEval(M, start, base)
                if not e.ok:
                    continue
                bf = brute_full(M, start, base, r, r + n)
                assert same(e.insert_full(M, r, r + n), bf), (n, off)
                bd = brute_delivery(M, start, base, r + n)
                # the same order treated as already picked up (delivery-only request)
                assert same(e.insert_delivery(M, r + n), bd), (n, off)
                # pure-Python path used for short routes: same optimum, and its slot achieves it
                c, i, j = e.best_full_py(M, r, r + n)
                if np.isfinite(bf).any():
                    assert abs(c - bf.min()) < 1e-7 and abs(bf[i, j] - c) < 1e-7, (n, off)
                else:
                    assert c == A.INF
                c, j = e.best_delivery_py(M, r + n)
                assert (c == A.INF and not np.isfinite(bd).any()) or abs(c - bd.min()) < 1e-7 and abs(bd[j] - c) < 1e-7
                checked += 1
    print(f"vectorized insertion == brute force on {checked} routes (with and without shift ends): OK")


def test_solver():
    for n in (5, 10, 20):
        chs = E.load_chunks(n)[:10]
        for ch in chs:
            seq, _ = A.solve_chunk(ch, max_iters=200, seed=0)
            assert E.valid(ch, seq)
            ins = E.route_cost(ch, A.chunk_insertion(A.chunk_model(ch)))
            assert E.route_cost(ch, seq) <= ins + 1e-9
    for ch in E.load_chunks(5)[:10]:
        seq, _ = A.solve_chunk(ch, max_iters=500, seed=0)
        opt = E.optimal_route(ch)
        assert E.route_cost(ch, seq) >= E.route_cost(ch, opt) - 1e-9
    print("solve_chunk returns valid routes, never worse than insertion, never below the exact optimum: OK")


def test_online(inst="mdrplib/public_instances/0o50t100s2p100", out="solutions/_test"):
    """With ALNS off, every courier a candidate and no lazy departures, the online simulator must reproduce
    the greedy-insertion baseline exactly; with ALNS on, the official evaluator must accept the solution."""
    import os
    import baselines
    from full_eval import official_check, metrics
    from online_alns import OnlineALNS
    ctx, routes, _ = baselines.run(inst, "greedy")
    baselines.write_solution(ctx, routes, os.path.join(out, "greedy"))
    OnlineALNS(inst, budget=0, interval=0, lazy=False, k_near=None).run().write_solution(os.path.join(out, "sim"))
    for f in ("solution_info_couriers.txt", "solution_info_orders.txt"):
        assert open(os.path.join(out, "greedy", f)).read() == open(os.path.join(out, "sim", f)).read(), f
    OnlineALNS(inst, budget=0.2, seed=0).run().write_solution(os.path.join(out, "alns"))
    ok, _ = official_check(inst, os.path.join(out, "alns"))
    assert ok, "official evaluator rejected the ALNS solution"
    g, a = metrics(inst, os.path.join(out, "greedy")), metrics(inst, os.path.join(out, "alns"))
    print(f"online: simulator == greedy baseline; ALNS solution FEASIBLE, objective {a['objective']:.1f} "
          f"vs greedy {g['objective']:.1f}: OK")


if __name__ == "__main__":
    test_simulate_matches_chunk_env()
    test_insertion_matches_brute_force()
    test_solver()
    test_online()
