"""Quality-vs-time figure for the chunk benchmark (results_alns_chunks.json -> figures/chunks_quality_time.png)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results_alns_chunks.json"))
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
style = {"nearest-nbr": ("#9e9e9e", "s"), "insertion": ("#424242", "D"), "insertion + polish": ("#424242", "d"),
         "attention (greedy)": ("#1f77b4", "o"), "attention (best of 32)": ("#1f77b4", "^"),
         "attention + polish": ("#1f77b4", "v")}
for ax, n in zip(axes, ("5", "10", "20")):
    rows = R[n]
    alns = sorted((v["ms_per_chunk"], v["gap_vs_insertion_pct"], v["ci95"]) for k, v in rows.items() if k.startswith("ALNS"))
    ax.plot([a[0] for a in alns], [a[1] for a in alns], "-o", color="#d62728", label="ALNS (10 ms .. 1 s budget)", zorder=3)
    for ms, g, ci in alns:
        ax.plot([ms, ms], ci, color="#d62728", lw=1, alpha=0.6)
    for k, (c, m) in style.items():
        v = rows[k]
        if k == "nearest-nbr":
            continue                                    # +11% to +32%, off the scale; reported in the table
        ax.errorbar(v["ms_per_chunk"], v["gap_vs_insertion_pct"],
                    yerr=[[v["gap_vs_insertion_pct"] - v["ci95"][0]], [v["ci95"][1] - v["gap_vs_insertion_pct"]]],
                    fmt=m, color=c, mfc="white" if "polish" in k or "32" in k else c, ms=7, capsize=2, label=k)
    if "exact optimum" in rows:
        go = 100 * (rows["exact optimum"]["cost_per_order"] / rows["insertion"]["cost_per_order"] - 1)
        ax.axhline(go, ls="--", color="green", lw=1, label="exact optimum")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("log")
    ax.set_title(f"{n} orders per courier")
    ax.set_xlabel("ms per chunk (log scale)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("cost vs cheapest insertion (%)\n(lower is better, bars = 95% CI)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.text(0.99, 0.01, "nearest neighbor (+11% to +32%) is off the scale", ha="right", fontsize=7, color="#666")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("figures/chunks_quality_time.png", dpi=150)
print("saved figures/chunks_quality_time.png")
