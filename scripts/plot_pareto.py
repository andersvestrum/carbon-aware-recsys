#!/usr/bin/env python
"""Plot Pareto frontier: NDCG@10 vs avg carbon, BPR vs NeuMF_pretrained."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "results"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "BPR": ("electronics_BPR_reranking_metrics.json", "tab:blue", "o"),
    "NeuMF (pretrained)": (
        "electronics_NeuMF_pretrained_reranking_metrics.json",
        "tab:red",
        "s",
    ),
}

fig, ax = plt.subplots(figsize=(7, 5))

for label, (fname, color, marker) in MODELS.items():
    data = json.loads((RESULTS / fname).read_text())
    per_lam = data["per_lambda"]
    carbon = [m["avg_carbon_kg"] for m in per_lam]
    ndcg = [m["NDCG@10"] for m in per_lam]
    lams = [m["lambda"] for m in per_lam]

    ax.plot(carbon, ndcg, color=color, lw=1.5, alpha=0.7)
    ax.scatter(carbon, ndcg, color=color, marker=marker, s=40, label=label, zorder=3)

    # Annotate λ=0, 0.5, 1
    for lam_mark in (0.0, 0.5, 1.0):
        try:
            idx = lams.index(lam_mark)
            ax.annotate(
                f"λ={lam_mark}",
                (carbon[idx], ndcg[idx]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color=color,
            )
        except ValueError:
            pass

ax.set_xlabel("Avg carbon footprint (kg CO₂e)")
ax.set_ylabel("NDCG@10")
ax.set_title("Carbon–Accuracy Pareto Frontier — Electronics\n"
             "BPR (k-core) vs NeuMF (pretrained)")
ax.invert_xaxis()  # lower carbon is better → left is "greener"
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")

out_path = OUT / "pareto_electronics_bpr_vs_neumf_pretrained.png"
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
