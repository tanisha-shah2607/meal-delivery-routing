import argparse
import copy
import random
import time
import numpy as np
import torch

import chunk_env
from attention_model import AttentionRouter, make_batch
from mdrp_chunks import SyntheticGen

SIZES = (5, 10, 20)


@torch.no_grad()
def greedy_cost(model, chs):
    return model(make_batch(chs), greedy=True)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="attn.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    gen = SyntheticGen(seed=args.seed)
    vgen = SyntheticGen(seed=12345)                      # fixed synthetic validation set (never real test data)
    val = {n: [vgen.sample(n) for _ in range(128)] for n in SIZES}
    print("reference cost/order on the synthetic validation set (python baselines):")
    for n in SIZES:
        nn_c = np.mean([chunk_env.route_cost(c, chunk_env.nn_route(c)) / n for c in val[n]])
        in_c = np.mean([chunk_env.route_cost(c, chunk_env.insertion_route(c)) / n for c in val[n]])
        print(f"   n={n:2d}  nearest-nbr {nn_c:7.2f} | insertion {in_c:7.2f}")

    model = AttentionRouter()
    base = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        n = random.choice(SIZES)
        b = make_batch([gen.sample(n) for _ in range(args.bs)])
        _, logp, cost = model(b)
        with torch.no_grad():
            bl = base(b, greedy=True)[2]
        loss = ((cost - bl).float() * logp).mean()       # REINFORCE with greedy-rollout baseline
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0:
            with torch.no_grad():
                pol = {n: greedy_cost(model, val[n]) for n in SIZES}
                old = {n: greedy_cost(base, val[n]) for n in SIZES}
            diff = torch.cat([old[n] - pol[n] for n in SIZES])
            tstat = diff.mean() / (diff.std() / diff.numel() ** 0.5 + 1e-9)
            updated = tstat.item() > 1.645                 # one-sided paired t-test, as in Kool et al.
            if updated:
                base.load_state_dict(model.state_dict())
            torch.save({"state": model.state_dict()}, args.out)
            row = " | ".join(f"n={n}: {pol[n].mean():7.2f}" for n in SIZES)
            print(f"step {step:5d} | val greedy {row} | baseline {'UPDATED' if updated else 'kept   '} "
                  f"| {time.time() - t0:6.0f}s", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()