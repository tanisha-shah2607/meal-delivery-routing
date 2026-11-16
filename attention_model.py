import math
import numpy as np
import torch
import torch.nn as nn
from baselines import W_DELIVERY, W_LATE, W_WAIT, W_TRAVEL

F64 = torch.float64


def make_batch(chs):
    """Turns a list of same-size chunks into tensors. Stop s < n: pickup of order s; s >= n: its delivery."""
    B, n = len(chs), chs[0]["n"]
    N = 2 * n
    sxy = np.zeros((B, N, 2))
    place, ready = np.zeros((B, n)), np.zeros((B, n))
    cour = np.zeros((B, 2))
    speed, ps, ds, target = (np.zeros(B) for _ in range(4))
    for b, ch in enumerate(chs):
        o = ch["orders"]
        sxy[b, :n] = [(a["rx"], a["ry"]) for a in o]
        sxy[b, n:] = [(a["cx"], a["cy"]) for a in o]
        place[b] = [a["placement"] for a in o]
        ready[b] = [a["ready"] for a in o]
        cour[b] = ch["courier"]
        speed[b], ps[b], ds[b], target[b] = ch["speed"], ch["ps"], ch["ds"], ch["target"]
    T = lambda a: torch.as_tensor(a, dtype=F64)
    d = dict(n=n, sxy=T(sxy), place=T(place), ready=T(ready), cour=T(cour),
             speed=T(speed), ps=T(ps), ds=T(ds), target=T(target))
    scale = (d["speed"] * 10.0)[:, None, None]              # coordinates in units of 10 travel-minutes
    rel = (d["sxy"] - d["cour"][:, None, :]) / scale
    prel = (torch.roll(d["sxy"], shifts=n, dims=1) - d["sxy"]) / scale   # vector to the paired stop
    flag = torch.cat([torch.zeros(B, n, dtype=F64), torch.ones(B, n, dtype=F64)], 1)[..., None]
    pr = torch.cat([d["place"], d["place"]], 1)[..., None] / 30.0
    rd = torch.cat([d["ready"], d["ready"]], 1)[..., None] / 30.0
    d["feat"] = torch.cat([rel, prel, flag, pr, rd], -1).float()          # (B, 2n, 7)
    return d


class AttentionRouter(nn.Module):
    """Encoder-decoder attention policy (Kool et al., 2019) adapted to pickup-and-delivery with time."""

    def __init__(self, d=128, heads=8, layers=3, ff=512, clip=10.0):
        super().__init__()
        self.d, self.heads, self.clip = d, heads, clip
        self.embed = nn.Linear(7, d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.start = nn.Parameter(torch.randn(d) * 0.1)      # stands in for "last stop" at step 0
        self.ctx = nn.Linear(2 * d + 3, d)                   # [graph, last stop, time, x, y]
        self.dyn = nn.Linear(3, d)                           # per-stop dynamic features
        self.wk = nn.Linear(d, 3 * d, bias=False)            # glimpse key, glimpse value, logit key
        self.wq = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.wq2 = nn.Linear(d, d, bias=False)

    def glimpse(self, q, gk, gv, avail):
        B, N, d = gk.shape
        H, dh = self.heads, d // self.heads
        qh = q.view(B, H, 1, dh)
        kh = gk.view(B, N, H, dh).permute(0, 2, 1, 3)
        vh = gv.view(B, N, H, dh).permute(0, 2, 1, 3)
        att = (qh @ kh.transpose(-1, -2)) / math.sqrt(dh)
        att = att.masked_fill(~avail[:, None, None, :], float("-inf")).softmax(-1)
        return self.wo((att @ vh).transpose(1, 2).reshape(B, d))

    def forward(self, b, greedy=False):
        """Builds one route per chunk. Returns (stop sequence, sum of log-probs, cost per order)."""
        n, feat = b["n"], b["feat"]
        B, N = feat.shape[0], 2 * n
        h = self.enc(self.embed(feat))
        graph = h.mean(1)
        gk, gv, lk = self.wk(h).chunk(3, dim=-1)
        sxy, place, ready, cour = b["sxy"], b["place"], b["ready"], b["cour"]
        speed, ps, ds, target = b["speed"], b["ps"], b["ds"], b["target"]
        place2, ready2 = torch.cat([place, place], 1), torch.cat([ready, ready], 1)
        ispick = (torch.arange(N) < n)[None, :]
        ar = torch.arange(B)
        scale = speed * 10.0
        x, y = cour[:, 0].clone(), cour[:, 1].clone()
        t = torch.zeros(B, dtype=F64)
        visited = torch.zeros(B, N, dtype=torch.bool)
        last = self.start.expand(B, -1)
        cost = torch.zeros(B, dtype=F64)
        logp = torch.zeros(B)
        seq = []
        for _ in range(N):
            ttm = torch.hypot(sxy[..., 0] - x[:, None], sxy[..., 1] - y[:, None]) / speed[:, None]
            wait = torch.where(ispick, (ready2 - (t[:, None] + ttm)).clamp(min=0), torch.zeros_like(ttm))
            dk = self.dyn(torch.stack([ttm / 10, wait / 30, (t[:, None] + ttm - place2) / 30], -1).float())
            avail = torch.cat([~visited[:, :n], visited[:, :n] & ~visited[:, n:]], 1)
            ctx = torch.cat([graph, last, (t / 30).float()[:, None],
                             ((x - cour[:, 0]) / scale).float()[:, None],
                             ((y - cour[:, 1]) / scale).float()[:, None]], -1)
            g = self.glimpse(self.wq(self.ctx(ctx)), gk + dk, gv, avail)
            logits = self.clip * torch.tanh((self.wq2(g)[:, None, :] * (lk + dk)).sum(-1) / math.sqrt(self.d))
            logps = torch.log_softmax(logits.masked_fill(~avail, float("-inf")), -1)
            a = logps.argmax(-1) if greedy else torch.multinomial(logps.exp(), 1).squeeze(1)
            logp = logp + logps.gather(1, a[:, None]).squeeze(1)
            # advance the simulation exactly like chunk_env.step
            ax, ay = sxy[ar, a, 0], sxy[ar, a, 1]
            tt = torch.ceil(torch.hypot(ax - x, ay - y) / speed)
            oid = a % n
            rdy, plc, pk = ready[ar, oid], place[ar, oid], a < n
            arr = t + tt
            done = torch.where(pk, torch.maximum(arr, rdy) + ps, arr + ds)
            c2d = done - plc
            cost = cost + torch.where(pk, W_WAIT * (done - ps - rdy).clamp(min=0),
                                      W_DELIVERY * c2d + W_LATE * (c2d - target).clamp(min=0)) + W_TRAVEL * tt
            t, x, y = done, ax, ay
            visited = visited.clone()
            visited[ar, a] = True
            last = h[ar, a]
            seq.append(a)
        return torch.stack(seq, 1), logp, cost / n