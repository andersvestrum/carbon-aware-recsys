"""
Diagnostic plots for NeuMF's sharper Pareto curve.

Consumes raw top-K relevance scores saved by ``train_and_score`` at
``output/results/{dataset}_{model}_scores.parquet`` and produces three
diagnostic figures written to ``output/figures/diagnostics/``:

  1. Normalized rank-score curves (mean ± IQR across users), one line
     per model. Directly visualizes distribution shape — flat curves
     resist carbon re-weighting; peaked curves collapse quickly.

  2. Top-1 vs rest gap: fraction of per-user score mass concentrated in
     rank 1. High values mean min-max normalization squashes ranks 2..K
     near zero, so even tiny λ lets carbon dominate.

  3. Within-user PCF std on the top-K candidate pool. Low std → little
     room for the re-ranker to find a greener alternative.

Usage:
    python scripts/diagnose_scores.py --category electronics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
FIG_DIR = PROJECT_ROOT / "output" / "figures" / "diagnostics"


def load_scores(category: str) -> dict[str, pd.DataFrame]:
    """Find all {category}_{model}_scores.parquet files."""
    pattern = f"{category}_*_scores.parquet"
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No score files matching {RESULTS_DIR}/{pattern}. "
            f"Run the pipeline notebook first."
        )
    out: dict[str, pd.DataFrame] = {}
    for f in files:
        model = f.stem.replace(f"{category}_", "").replace("_scores", "")
        out[model] = pd.read_parquet(f)
    return out


def load_carbon(category: str) -> pd.DataFrame:
    from src.recommender.recbole_formatter import _concat_interim_splits
    df = _concat_interim_splits(category)
    return df[["parent_asin", "pcf"]].dropna().drop_duplicates("parent_asin")


# ─── Plot 1: normalized rank-score curves ───────────────────────────────────

def plot_rank_score_curves(scores: dict[str, pd.DataFrame], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, df in scores.items():
        # Per user: sort desc, min-max normalize, align by rank
        def _user_curve(g: pd.DataFrame) -> np.ndarray:
            s = g["relevance_score"].sort_values(ascending=False).values
            if s.max() == s.min():
                return np.full_like(s, 0.5, dtype=float)
            return (s - s.min()) / (s.max() - s.min())

        curves = (
            df.groupby("user_id", sort=False)
            .apply(_user_curve, include_groups=False)
            .tolist()
        )
        # Pad/truncate to common length K
        K = min(len(c) for c in curves)
        mat = np.stack([c[:K] for c in curves])

        ranks = np.arange(1, K + 1)
        mean = mat.mean(axis=0)
        q25 = np.quantile(mat, 0.25, axis=0)
        q75 = np.quantile(mat, 0.75, axis=0)

        line, = ax.plot(ranks, mean, label=model, linewidth=2)
        ax.fill_between(ranks, q25, q75, color=line.get_color(), alpha=0.15)

    ax.set_xlabel("Rank within user's top-K candidates")
    ax.set_ylabel("Min-max normalized engagement score")
    ax.set_title("Per-user score-vs-rank distribution shape\n(mean with IQR band)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── Plot 2: top-1 concentration ────────────────────────────────────────────

def plot_top1_gap(scores: dict[str, pd.DataFrame], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, df in scores.items():
        gaps = []
        for _, g in df.groupby("user_id", sort=False):
            s = np.sort(g["relevance_score"].values)[::-1]
            if len(s) < 2 or (s[0] - s[-1]) == 0:
                continue
            # Gap between rank-1 and rank-2 as fraction of full range
            gaps.append((s[0] - s[1]) / (s[0] - s[-1]))
        ax.hist(gaps, bins=40, alpha=0.5, label=f"{model} (n={len(gaps)})", density=True)

    ax.set_xlabel("(score₁ − score₂) / (score₁ − score_K)")
    ax.set_ylabel("Density")
    ax.set_title("Top-1 concentration: fraction of range consumed by the rank-1 gap")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── Plot 3: within-user PCF std ────────────────────────────────────────────

def plot_pcf_std(scores: dict[str, pd.DataFrame], carbon_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, df in scores.items():
        merged = df.merge(carbon_df, on="parent_asin", how="left").dropna(subset=["pcf"])
        stds = merged.groupby("user_id")["pcf"].std().dropna()
        ax.hist(stds, bins=40, alpha=0.5, label=f"{model} (median={stds.median():.1f})",
                density=True)

    ax.set_xlabel("Std of PCF across a user's top-K candidates (kg CO₂e)")
    ax.set_ylabel("Density")
    ax.set_title("How much PCF diversity does each model's candidate pool expose?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="electronics")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores for category='{args.category}'...")
    scores = load_scores(args.category)
    for m, df in scores.items():
        print(f"  {m}: {len(df):,} rows, {df['user_id'].nunique():,} users")

    print("Loading carbon predictions...")
    carbon = load_carbon(args.category)
    print(f"  {len(carbon):,} items with PCF")

    print("Plotting...")
    plot_rank_score_curves(scores, FIG_DIR / f"{args.category}_rank_score_curves.png")
    plot_top1_gap(scores, FIG_DIR / f"{args.category}_top1_gap.png")
    plot_pcf_std(scores, carbon, FIG_DIR / f"{args.category}_pcf_std.png")

    # Print summary table
    print("\nSummary:")
    print(f"  {'model':<12} {'median top1-gap':>16} {'median pcf-std':>16}")
    for model, df in scores.items():
        gaps = []
        for _, g in df.groupby("user_id", sort=False):
            s = np.sort(g["relevance_score"].values)[::-1]
            if len(s) >= 2 and (s[0] - s[-1]) > 0:
                gaps.append((s[0] - s[1]) / (s[0] - s[-1]))
        merged = df.merge(carbon, on="parent_asin", how="left").dropna(subset=["pcf"])
        pcf_std = merged.groupby("user_id")["pcf"].std().dropna().median()
        print(f"  {model:<12} {np.median(gaps):>16.3f} {pcf_std:>16.1f}")


if __name__ == "__main__":
    main()
