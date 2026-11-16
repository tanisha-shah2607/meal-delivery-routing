import os
import sys
import time
from functools import lru_cache

from mdrp_data import Instance

# Objective weights (tunable): delivery time, lateness over target, food waiting, courier travel.
W_DELIVERY, W_LATE, W_WAIT, W_TRAVEL = 1.0, 1.0, 0.5, 0.2


class Ctx:
    """Fast lookups plus route scheduling. Shared by baselines and the attention model later."""

    def __init__(self, inst):
        self.inst = inst
        o = inst.orders
        self.placement = o.placement_time.astype(int).to_dict()
        self.ready = o.ready_time.astype(int).to_dict()
        self.rest = o.restaurant.to_dict()
        self.ps = int(inst.pickup_service)
        self.ds = int(inst.dropoff_service)
        self.target = inst.target_c2d
        self.tt = lru_cache(maxsize=None)(inst.travel)
        self.on = inst.couriers.on_time.astype(int).to_dict()
        self.off = inst.couriers.off_time.astype(int).to_dict()

    def schedule(self, courier, stops, loc=None, free=None):
        """stops = [('P', order) | ('D', order)]. Returns [(depart, arrive, done, node)]."""
        if loc is None:
            loc, free = courier, self.on[courier]
        out = []
        for kind, o in stops:
            if kind == "P":
                node = self.rest[o]
                dep = max(free, self.placement[o])   # cannot be assigned before it is placed
                arr = dep + self.tt(loc, node)
                done = max(arr, self.ready[o]) + self.ps
            else:
                node = o
                dep = free
                arr = dep + self.tt(loc, node)
                done = arr + self.ds
            out.append((dep, arr, done, node))
            loc, free = node, done
        return out

    def cost_ok(self, courier, stops, loc=None, free=None):
        sched = self.schedule(courier, stops, loc, free)
        cost, ok = 0.0, True
        for (kind, o), (dep, arr, done, node) in zip(stops, sched):
            if kind == "P":
                if done > self.off[courier]:
                    ok = False                      # pickup after the courier's shift ends
                cost += W_WAIT * max(0, done - self.ps - self.ready[o])
            else:
                c2d = done - self.placement[o]
                cost += W_DELIVERY * c2d + W_LATE * max(0, c2d - self.target)
            cost += W_TRAVEL * (arr - dep)
        return cost, ok


def nn_choose(ctx, routes, state, o, t):
    """Nearest courier: whoever can reach the restaurant earliest."""
    r = ctx.rest[o]
    best = None
    for c in routes:
        loc, free = state[c]
        arr = max(free, t) + ctx.tt(loc, r)
        done = max(arr, ctx.ready[o]) + ctx.ps
        if done > ctx.off[c]:
            continue
        key = (arr, done)
        if best is None or key < best[0]:
            best = (key, c)
    return None if best is None else (best[1], routes[best[1]] + [("P", o), ("D", o)])


def greedy_choose(ctx, routes, scheds, o, t):
    """Cheapest insertion of (pickup, delivery) into any courier's not-yet-started stops."""
    best = None
    for c, stops in routes.items():
        sched = scheds[c]
        nf = 0
        while nf < len(sched) and sched[nf][0] < t:   # stops already under way are frozen
            nf += 1
        loc, free = (c, ctx.on[c]) if nf == 0 else (sched[nf - 1][3], sched[nf - 1][2])
        suffix = stops[nf:]
        base = ctx.cost_ok(c, suffix, loc, free)[0]
        L = len(suffix)
        for i in range(L + 1):
            for j in range(i, L + 1):
                new = suffix[:i] + [("P", o)] + suffix[i:j] + [("D", o)] + suffix[j:]
                cost, ok = ctx.cost_ok(c, new, loc, free)
                if ok and (best is None or cost - base < best[0]):
                    best = (cost - base, c, stops[:nf] + new)
    return None if best is None else (best[1], best[2])


def run(instance_dir, method):
    inst = Instance(instance_dir)
    ctx = Ctx(inst)
    routes = {c: [] for c in inst.couriers.index}
    scheds = {c: [] for c in routes}
    state = {c: (c, ctx.on[c]) for c in routes}
    unserved = []
    order_list = sorted(inst.orders.index, key=lambda o: (ctx.placement[o], int(o[1:])))
    for o in order_list:
        t = ctx.placement[o]
        choice = nn_choose(ctx, routes, state, o, t) if method == "nn" \
            else greedy_choose(ctx, routes, scheds, o, t)
        if choice is None:
            unserved.append(o)
            continue
        c, new_stops = choice
        routes[c] = new_stops
        scheds[c] = ctx.schedule(c, new_stops)
        state[c] = (scheds[c][-1][3], scheds[c][-1][2])
    return ctx, routes, unserved


def write_solution(ctx, routes, out_dir):
    """Writes the three files the official evaluator expects. Returns per-order results."""
    os.makedirs(out_dir, exist_ok=True)
    pick, drop, who, assign, moves = {}, {}, {}, [], []
    for c, stops in routes.items():
        loc = c
        for (kind, o), (dep, arr, done, node) in zip(stops, ctx.schedule(c, stops)):
            if node != loc:
                moves.append(f"{c} {dep} {loc} {node}")
            if kind == "P":
                pick[o], who[o] = done, c
                assign.append(f"{ctx.placement[o]} {done} {c} {o}")
            else:
                drop[o] = done
            loc = node
    with open(os.path.join(out_dir, "solution_info_assignments.txt"), "w") as f:
        f.write("assignment_time pickup_time courier bundle\n" + "\n".join(assign) + "\n")
    with open(os.path.join(out_dir, "solution_info_couriers.txt"), "w") as f:
        f.write("courier departure_time origin destination\n" + "\n".join(moves) + "\n")
    with open(os.path.join(out_dir, "solution_info_orders.txt"), "w") as f:
        f.write("order placement_time ready_time pickup_time dropoff_time courier\n")
        for o in ctx.placement:
            if o in drop:
                f.write(f"{o} {ctx.placement[o]} {ctx.ready[o]} {pick[o]} {drop[o]} {who[o]}\n")
            else:
                f.write(f"{o} {ctx.placement[o]} {ctx.ready[o]} NA NA NA\n")
    return pick, drop


if __name__ == "__main__":
    inst_dir, method, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    ctx, routes, unserved = run(inst_dir, method)
    secs = time.time() - t0
    pick, drop = write_solution(ctx, routes, out_dir)
    c2d = [drop[o] - ctx.placement[o] for o in drop]
    print(f"method={method} delivered={len(drop)}/{len(ctx.placement)} unserved={len(unserved)}")
    print(f"mean click-to-door={sum(c2d)/len(c2d):.1f} min | "
          f"late (> {ctx.target:.0f} min)={sum(x > ctx.target for x in c2d)} | runtime={secs:.1f}s")
    print("solution written to:", out_dir)