import chunk_env


def polish(ch, seq, max_pass=3):
    """Relocate one stop at a time (pickup or delivery) to its best feasible position until no move helps.

    Keeps every pickup before its delivery, so the route stays valid.
    """
    n = len(ch["orders"])
    best = list(seq)
    best_cost = chunk_env.route_cost(ch, best)
    for _ in range(max_pass):
        improved = False
        for s in list(best):
            base = [x for x in best if x != s]
            if s < n:                                  # pickup: must stay before its delivery
                lo, hi = 0, base.index(s + n)
            else:                                      # delivery: must stay after its pickup
                lo, hi = base.index(s - n) + 1, len(base)
            for k in range(lo, hi + 1):
                cand = base[:k] + [s] + base[k:]
                c = chunk_env.route_cost(ch, cand)
                if c < best_cost - 1e-9:
                    best, best_cost, improved = cand, c, True
        if not improved:
            break
    return best