"""
Evaluation Metrics — Pipeline Step 4

Analyses the engagement vs carbon footprint trade-off across λ values.

Core capabilities:
    • Pareto frontier extraction (engagement vs carbon)
    • Summary statistics table across all λ
    • Visualisation: trade-off curve with Pareto front highlighted
    • Per-category and cross-category comparison

Metrics:
    - NDCG@k, Recall@k, MRR  (ranking quality)
    - avg_carbon_kg            (mean PCF of recommended items, shown as AvgPCF@k)
    - carbon_reduction_pct     (vs λ=0 baseline)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def _infer_top_k(metric_key: str, default_k: int = 10) -> int:
    """Infer @k from a metric name like ``NDCG@10``."""
    match = re.search(r"@(\d+)", metric_key)
    return int(match.group(1)) if match else default_k


def _carbon_axis_label(carbon_key: str, engagement_key: str) -> str:
    """Return report-consistent axis label for carbon metrics."""
    if carbon_key == "avg_carbon_kg":
        k = _infer_top_k(engagement_key)
        return f"AvgPCF@{k} (kg CO₂e)"
    if carbon_key.startswith("AvgPCF@"):
        return f"{carbon_key} (kg CO₂e)"
    return carbon_key


# ─── Pareto frontier ─────────────────────────────────────────────────────────

def pareto_frontier(
    metrics: list[dict],
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
) -> list[dict]:
    """Extract the Pareto-optimal operating points.

    A point is Pareto-optimal if no other point has *both*
    higher engagement *and* lower carbon.

    Args:
        metrics: List of per-λ metric dicts (from ``03_rerank.py``).
        engagement_key: Key for the engagement metric (to maximise).
        carbon_key: Key for the carbon metric (to minimise).

    Returns:
        Subset of ``metrics`` that lie on the Pareto front,
        sorted by ascending carbon.
    """
    # Sort by carbon ascending (primary), engagement descending (secondary)
    sorted_pts = sorted(
        metrics,
        key=lambda m: (m[carbon_key], -m[engagement_key]),
    )

    pareto: list[dict] = []
    best_engagement = -float("inf")

    for pt in sorted_pts:
        if pt[engagement_key] > best_engagement:
            pareto.append(pt)
            best_engagement = pt[engagement_key]

    log.info(
        "Pareto frontier: %d / %d points",
        len(pareto), len(metrics),
    )
    return pareto


def pareto_frontier_df(
    metrics: list[dict],
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
) -> pd.DataFrame:
    """Return the Pareto front as a tidy DataFrame."""
    front = pareto_frontier(metrics, engagement_key, carbon_key)
    df = pd.DataFrame(front)
    df["pareto_optimal"] = True
    return df


# ─── Summary table ───────────────────────────────────────────────────────────

def build_summary_table(
    metrics: list[dict],
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
) -> pd.DataFrame:
    """Build a summary DataFrame from per-λ metrics.

    Adds columns:
        - ``pareto_optimal``: whether the point is on the Pareto front
        - ``carbon_reduction_pct``: % reduction vs λ=0 baseline
        - ``engagement_retention_pct``: % of λ=0 engagement retained
    """
    df = pd.DataFrame(metrics)

    # Mark Pareto-optimal points
    front = pareto_frontier(metrics, engagement_key, carbon_key)
    front_lambdas = {m["lambda"] for m in front}
    df["pareto_optimal"] = df["lambda"].isin(front_lambdas)

    # Baseline (λ=0)
    baseline = df.loc[df["lambda"] == 0.0].iloc[0]
    baseline_carbon = baseline[carbon_key]
    baseline_engagement = baseline[engagement_key]

    df["carbon_reduction_pct"] = (
        100.0 * (baseline_carbon - df[carbon_key]) / baseline_carbon
        if baseline_carbon > 0 else 0.0
    )
    df["engagement_retention_pct"] = (
        100.0 * df[engagement_key] / baseline_engagement
        if baseline_engagement > 0 else 0.0
    )

    return df.sort_values("lambda").reset_index(drop=True)


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_tradeoff_curve(
    metrics: list[dict],
    category: str,
    model_name: str = "BPR",
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot engagement vs carbon trade-off with Pareto front highlighted.

    Args:
        metrics: Per-λ metric dicts.
        category: Category name (for title).
        model_name: Model name (for title).
        engagement_key: Engagement metric key.
        carbon_key: Carbon metric key.
        save: Save to ``output/figures/``.
        show: Display the plot interactively.

    Returns:
        matplotlib Figure.
    """
    df = pd.DataFrame(metrics)
    front = pareto_frontier(metrics, engagement_key, carbon_key)
    front_df = pd.DataFrame(front).sort_values(carbon_key)

    fig, ax = plt.subplots(figsize=(10, 6))

    # All operating points
    ax.scatter(
        df[carbon_key], df[engagement_key],
        c="steelblue", s=60, alpha=0.6, zorder=3,
        label="Operating points",
    )

    # Annotate λ values on all operating points
    for _, row in df.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.2f}",
            (row[carbon_key], row[engagement_key]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, color="gray", alpha=0.7,
        )

    # Pareto front
    ax.plot(
        front_df[carbon_key], front_df[engagement_key],
        "o-", color="crimson", markersize=8, linewidth=2, zorder=4,
        label="Pareto frontier",
    )

    # Annotate λ values on Pareto points
    for _, row in front_df.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.2f}",
            (row[carbon_key], row[engagement_key]),
            textcoords="offset points", xytext=(8, 6),
            fontsize=8, color="crimson", fontweight="bold",
        )

    # Baseline marker (λ=0)
    baseline = df.loc[df["lambda"] == 0.0]
    if not baseline.empty:
        ax.scatter(
            baseline[carbon_key], baseline[engagement_key],
            marker="*", s=200, c="gold", edgecolors="black",
            zorder=5, label="Baseline (λ=0)",
        )

    ax.set_xlabel(_carbon_axis_label(carbon_key, engagement_key), fontsize=12)
    ax.set_ylabel(engagement_key, fontsize=12)
    ax.set_title(
        f"Engagement–Emissions Pareto Frontier — {category.replace('_', ' ').title()} ({model_name})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"{category}_{model_name}_tradeoff.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        log.info("Saved trade-off plot → %s", fig_path)

    if show:
        plt.show()

    return fig


def plot_multi_category(
    all_metrics: dict[str, list[dict]],
    model_name: str = "BPR",
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot trade-off curves for multiple categories on one figure.

    Args:
        all_metrics: Dict mapping category → per-λ metrics list.
        model_name: Model name (for title).
        engagement_key: Engagement metric key.
        carbon_key: Carbon metric key.
        save: Save to ``output/figures/``.
        show: Display interactively.

    Returns:
        matplotlib Figure.
    """
    colors = {"electronics": "steelblue", "home_and_kitchen": "seagreen",
              "sports_and_outdoors": "coral"}

    fig, ax = plt.subplots(figsize=(12, 7))

    for cat, metrics in all_metrics.items():
        df = pd.DataFrame(metrics).sort_values(carbon_key)
        front = pareto_frontier(metrics, engagement_key, carbon_key)
        front_df = pd.DataFrame(front).sort_values(carbon_key)
        color = colors.get(cat, "gray")
        label = cat.replace("_", " ").title()

        ax.scatter(
            df[carbon_key], df[engagement_key],
            c=color, s=40, alpha=0.4,
        )
        ax.plot(
            front_df[carbon_key], front_df[engagement_key],
            "o-", color=color, markersize=6, linewidth=2,
            label=f"{label} (Pareto)",
        )

    ax.set_xlabel(_carbon_axis_label(carbon_key, engagement_key), fontsize=12)
    ax.set_ylabel(engagement_key, fontsize=12)
    ax.set_title(
        f"Engagement–Emissions Pareto Frontier — All Categories ({model_name})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"all_categories_{model_name}_tradeoff.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        log.info("Saved multi-category plot → %s", fig_path)

    if show:
        plt.show()

    return fig


def plot_lambda_sensitivity(
    metrics: list[dict],
    category: str,
    model_name: str = "BPR",
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot engagement and carbon as functions of λ (dual y-axis).

    Shows how both metrics change as λ increases from 0 → 1.
    """
    df = pd.DataFrame(metrics).sort_values("lambda")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Engagement (left axis)
    color_eng = "steelblue"
    ax1.plot(
        df["lambda"], df[engagement_key],
        "o-", color=color_eng, linewidth=2, markersize=6,
    )
    ax1.set_xlabel("λ (carbon weight)", fontsize=12)
    ax1.set_ylabel(engagement_key, fontsize=12, color=color_eng)
    ax1.tick_params(axis="y", labelcolor=color_eng)

    # Carbon (right axis)
    ax2 = ax1.twinx()
    color_carb = "coral"
    ax2.plot(
        df["lambda"], df[carbon_key],
        "s--", color=color_carb, linewidth=2, markersize=6,
    )
    ax2.set_ylabel(
        _carbon_axis_label(carbon_key, engagement_key),
        fontsize=12,
        color=color_carb,
    )
    ax2.tick_params(axis="y", labelcolor=color_carb)

    ax1.set_title(
        f"λ Sensitivity — {category.replace('_', ' ').title()} ({model_name})",
        fontsize=14, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"{category}_{model_name}_lambda_sensitivity.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        log.info("Saved λ-sensitivity plot → %s", fig_path)

    if show:
        plt.show()

    return fig


# ─── Full evaluation runner ─────────────────────────────────────────────────

def evaluate_category(
    category: str,
    model_name: str = "BPR",
    engagement_key: str = "NDCG@10",
    carbon_key: str = "avg_carbon_kg",
) -> dict[str, Any]:
    """Run full evaluation for one category.

    Loads the reranking metrics JSON produced by ``03_rerank.py``,
    computes the Pareto frontier, builds a summary table, and
    generates all plots.

    Returns:
        Dict with ``summary_table``, ``pareto_points``, ``baseline``,
        ``best_tradeoff``.
    """
    metrics_path = RESULTS_DIR / f"{category}_{model_name}_reranking_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Reranking metrics not found at {metrics_path}. "
            "Run 03_rerank.py first."
        )

    with open(metrics_path) as f:
        data = json.load(f)

    per_lambda = data["per_lambda"]

    # Summary table
    summary_df = build_summary_table(per_lambda, engagement_key, carbon_key)

    # Pareto frontier
    front = pareto_frontier(per_lambda, engagement_key, carbon_key)

    # Best trade-off: Pareto point with highest engagement that still
    # achieves ≥ 10% carbon reduction
    baseline_carbon = summary_df.loc[
        summary_df["lambda"] == 0.0, carbon_key
    ].iloc[0]

    best_tradeoff = None
    for pt in sorted(front, key=lambda p: -p[engagement_key]):
        reduction = (baseline_carbon - pt[carbon_key]) / baseline_carbon
        if reduction >= 0.10:
            best_tradeoff = pt
            break
    # If no point achieves 10%, pick the best Pareto point after baseline
    if best_tradeoff is None and len(front) > 1:
        best_tradeoff = sorted(front, key=lambda p: -p[engagement_key])[1]

    # Plots
    plot_tradeoff_curve(per_lambda, category, model_name, engagement_key, carbon_key)
    plot_lambda_sensitivity(per_lambda, category, model_name, engagement_key, carbon_key)

    # Save summary table
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = RESULTS_DIR / f"{category}_{model_name}_evaluation_summary.csv"
    summary_df.to_csv(table_path, index=False)
    log.info("Saved evaluation summary → %s", table_path)

    # Save Pareto points
    pareto_path = RESULTS_DIR / f"{category}_{model_name}_pareto.json"
    with open(pareto_path, "w") as f:
        json.dump(front, f, indent=2)
    log.info("Saved Pareto frontier → %s", pareto_path)

    return {
        "summary_table": summary_df,
        "pareto_points": front,
        "baseline": per_lambda[0] if per_lambda else {},
        "best_tradeoff": best_tradeoff,
    }
