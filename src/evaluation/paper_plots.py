"""
Paper-ready plots for the Results section in docs/main.tex.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(
    os.environ.get(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "carbon-aware-recsys-mpl"),
    )
)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import build_summary_table, pareto_frontier

log = logging.getLogger(__name__)

CATEGORY_ORDER = ["electronics", "home_and_kitchen", "sports_and_outdoors"]
MODEL_ORDER = ["BPR", "LightGCN", "NeuMF"]
METHOD_ORDER = ["neighbor_average", "zero_shot_llm", "few_shot_llm"]

CATEGORY_LABELS = {
    "electronics": "Electronics",
    "home_and_kitchen": "Home and Kitchen",
    "sports_and_outdoors": "Sports and Outdoors",
}
MODEL_COLORS = {
    "BPR": "#2f6f6f",
    "LightGCN": "#d96c06",
    "NeuMF": "#8d5a97",
}
CATEGORY_COLORS = {
    "electronics": "#2f6f6f",
    "home_and_kitchen": "#d96c06",
    "sports_and_outdoors": "#8d5a97",
}
METHOD_LABELS = {
    "neighbor_average": "Neighbour average",
    "zero_shot_llm": "Zero-shot LLM",
    "few_shot_llm": "Few-shot LLM",
}
METHOD_FIGURE_SUFFIX = {
    "neighbor_average": "neighbour",
    "zero_shot_llm": "zeroshot",
    "few_shot_llm": "fewshot",
}
SECTOR_COLUMN = "*Company's sector"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def _best_pareto_point(
    summary_df: pd.DataFrame,
    *,
    min_carbon_reduction_pct: float = 10.0,
    engagement_key: str = "NDCG@10",
) -> pd.Series | None:
    pareto_df = summary_df[summary_df["pareto_optimal"]].copy()
    if pareto_df.empty:
        return None

    eligible = pareto_df[pareto_df["carbon_reduction_pct"] >= min_carbon_reduction_pct]
    if not eligible.empty:
        return eligible.sort_values(
            [engagement_key, "carbon_reduction_pct"],
            ascending=[False, False],
        ).iloc[0]
    return pareto_df.sort_values(engagement_key, ascending=False).iloc[0]


def _cliff_lambda(summary_df: pd.DataFrame, *, max_ndcg_drop_pct: float = 5.0) -> float | None:
    threshold = 100.0 - max_ndcg_drop_pct
    cliff_rows = summary_df[summary_df["engagement_retention_pct"] < threshold]
    if cliff_rows.empty:
        return None
    return float(cliff_rows.iloc[0]["lambda"])


def _max_carbon_reduction_under_cap(
    summary_df: pd.DataFrame,
    *,
    max_ndcg_drop_pct: float = 5.0,
) -> float:
    threshold = 100.0 - max_ndcg_drop_pct
    eligible = summary_df[summary_df["engagement_retention_pct"] >= threshold]
    if eligible.empty:
        return 0.0
    return float(eligible["carbon_reduction_pct"].max())


def load_reranking_results(
    results_dir: Path,
    categories: list[str] | None = None,
    models: list[str] | None = None,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    categories = categories or CATEGORY_ORDER
    models = models or MODEL_ORDER
    metrics_by_combo: dict[tuple[str, str], pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []

    for category in categories:
        for model in models:
            metrics_path = Path(results_dir) / f"{category}_{model}_reranking_metrics.json"
            if not metrics_path.exists():
                log.warning("Missing reranking metrics: %s", metrics_path)
                continue

            with metrics_path.open() as handle:
                payload = json.load(handle)
            summary_df = build_summary_table(payload["per_lambda"])
            summary_df.insert(0, "model", model)
            summary_df.insert(0, "category", category)
            metrics_by_combo[(category, model)] = summary_df
            all_rows.append(summary_df)

    if not all_rows:
        raise FileNotFoundError(f"No reranking metrics were found in {results_dir}.")

    return metrics_by_combo, pd.concat(all_rows, ignore_index=True)


def _build_best_points_df(all_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (category, model), combo_df in all_results.groupby(["category", "model"]):
        best = _best_pareto_point(combo_df)
        if best is None:
            continue
        rows.append(
            {
                "category": category,
                "model": model,
                "lambda": float(best["lambda"]),
                "NDCG@10": float(best["NDCG@10"]),
                "Recall@10": float(best["Recall@10"]),
                "avg_carbon_kg": float(best["avg_carbon_kg"]),
                "carbon_reduction_pct": float(best["carbon_reduction_pct"]),
            }
        )
    return pd.DataFrame(rows)


def plot_pcf_method_comparison(metrics_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    available = metrics_df.copy()
    if "available" in available.columns:
        available = available[available["available"].fillna(False)].copy()
    if available.empty:
        return None

    figure_dir = _ensure_dir(figure_dir)
    available["label"] = available["method"].map(METHOD_LABELS)
    available["color"] = available["method"].map(
        {
            "neighbor_average": "#2f6f6f",
            "zero_shot_llm": "#d96c06",
            "few_shot_llm": "#8d5a97",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("rmse", "RMSE", True),
        ("mae", "MAE", True),
        ("spearman", "Spearman", False),
    ]

    for ax, (column, title, log_scale) in zip(axes, metrics):
        values = available[column].astype(float)
        ax.bar(available["label"], values, color=available["color"])
        if log_scale:
            ax.set_yscale("log")
        winner_idx = values.idxmin() if column != "spearman" else values.idxmax()
        winner = available.loc[winner_idx]
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.text(
            available.index.get_loc(winner_idx),
            values.loc[winner_idx],
            "best",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    fig.suptitle("PCF Estimation Accuracy Across Methods", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = figure_dir / "pcf_method_comparison_bar.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pcf_scatter(predictions_df: pd.DataFrame, method: str, figure_dir: Path) -> Path | None:
    pred_col = {
        "neighbor_average": "neighbor_average_pcf",
        "zero_shot_llm": "zero_shot_llm_pcf",
        "few_shot_llm": "few_shot_llm_pcf",
    }[method]
    if pred_col not in predictions_df.columns or "pcf" not in predictions_df.columns:
        return None

    df = predictions_df[["pcf", pred_col] + ([SECTOR_COLUMN] if SECTOR_COLUMN in predictions_df.columns else [])].copy()
    df = df.rename(columns={pred_col: "prediction"})
    df = df.dropna(subset=["pcf", "prediction"])
    df = df[(df["pcf"] > 0) & (df["prediction"] > 0)]
    if df.empty:
        return None

    figure_dir = _ensure_dir(figure_dir)
    fig, ax = plt.subplots(figsize=(5, 5))

    if SECTOR_COLUMN in df.columns:
        sectors = df[SECTOR_COLUMN].fillna("Unknown")
        top_sectors = sectors.value_counts().head(6).index
        colors = np.where(sectors.isin(top_sectors), sectors, "Other")
        for label in pd.unique(colors):
            subset = df[colors == label]
            ax.scatter(
                subset["pcf"],
                subset["prediction"],
                s=18,
                alpha=0.6,
                label=str(label),
            )
        ax.legend(fontsize=7, loc="upper left")
    else:
        ax.scatter(df["pcf"], df["prediction"], s=18, alpha=0.6, color="#2f6f6f")

    lo = min(df["pcf"].min(), df["prediction"].min())
    hi = max(df["pcf"].max(), df["prediction"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True PCF (kg CO$_2$e)")
    ax.set_ylabel("Predicted PCF (kg CO$_2$e)")
    ax.set_title(METHOD_LABELS[method])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figure_dir / f"pcf_scatter_{METHOD_FIGURE_SUFFIX[method]}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pcf_distribution(amazon_df: pd.DataFrame, category: str, figure_dir: Path) -> Path | None:
    source_col = "source_category" if "source_category" in amazon_df.columns else "main_category"
    subset = amazon_df[amazon_df[source_col] == category].copy()
    if "pcf" not in subset.columns:
        return None
    subset = subset[subset["pcf"].notna() & (subset["pcf"] > 0)]
    if subset.empty:
        return None

    figure_dir = _ensure_dir(figure_dir)
    pcf_values = subset["pcf"].astype(float)
    bins = np.logspace(np.log10(pcf_values.min()), np.log10(pcf_values.max()), 30)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pcf_values, bins=bins, color=CATEGORY_COLORS[category], alpha=0.8)
    ax.axvline(pcf_values.median(), color="black", linestyle="--", linewidth=1.5, label="Median")
    ax.axvline(pcf_values.quantile(0.9), color="black", linestyle=":", linewidth=1.5, label="90th pct")
    ax.set_xscale("log")
    ax.set_xlabel("Estimated PCF (kg CO$_2$e)")
    ax.set_ylabel("Item count")
    ax.set_title(_category_label(category))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figure_dir / f"pcf_dist_{category}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lambda_sensitivity(summary_df: pd.DataFrame, category: str, model: str, figure_dir: Path) -> Path:
    figure_dir = _ensure_dir(figure_dir)
    df = summary_df.sort_values("lambda").copy()
    cliff_lambda = _cliff_lambda(df)
    plateau_end = cliff_lambda if cliff_lambda is not None else float(df["lambda"].max())

    fig, ax1 = plt.subplots(figsize=(5.2, 4))
    ax1.axvspan(0.0, plateau_end, color="#d8f0d2", alpha=0.5)
    ax1.plot(df["lambda"], df["NDCG@10"], color="#2f6f6f", marker="o", linewidth=2)
    if cliff_lambda is not None:
        ax1.axvline(cliff_lambda, color="#b22222", linestyle=":", linewidth=1.5)
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("NDCG@10", color="#2f6f6f")
    ax1.tick_params(axis="y", labelcolor="#2f6f6f")

    ax2 = ax1.twinx()
    ax2.plot(df["lambda"], df["avg_carbon_kg"], color="#d96c06", marker="s", linestyle="--", linewidth=2)
    ax2.set_ylabel("AvgPCF@10 (kg CO$_2$e)", color="#d96c06")
    ax2.tick_params(axis="y", labelcolor="#d96c06")

    ax1.set_title(f"{_category_label(category)} — {model}")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figure_dir / f"lambda_sensitivity_{category}_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pareto(summary_df: pd.DataFrame, category: str, model: str, figure_dir: Path) -> Path:
    figure_dir = _ensure_dir(figure_dir)
    df = summary_df.sort_values("lambda").copy()
    front_df = pd.DataFrame(
        pareto_frontier(df.to_dict("records"), engagement_key="NDCG@10", carbon_key="avg_carbon_kg")
    ).sort_values("avg_carbon_kg")
    baseline = df.loc[df["lambda"] == 0.0].iloc[0]
    best = _best_pareto_point(df)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(df["avg_carbon_kg"], df["NDCG@10"], color="#7aa6a6", s=36, alpha=0.7)
    ax.plot(front_df["avg_carbon_kg"], front_df["NDCG@10"], color="#b22222", marker="o", linewidth=2)
    ax.scatter(
        [baseline["avg_carbon_kg"]],
        [baseline["NDCG@10"]],
        marker="*",
        s=180,
        color="gold",
        edgecolors="black",
        zorder=5,
    )
    if best is not None:
        ax.annotate(
            f"$\\lambda$={best['lambda']:.2f}",
            (best["avg_carbon_kg"], best["NDCG@10"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlabel("AvgPCF@10 (kg CO$_2$e)")
    ax.set_ylabel("NDCG@10")
    ax.set_title(f"{_category_label(category)} — {model}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figure_dir / f"pareto_{category}_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multimodel_pareto(all_results: pd.DataFrame, category: str, figure_dir: Path) -> Path:
    figure_dir = _ensure_dir(figure_dir)
    fig, ax = plt.subplots(figsize=(5.2, 4))

    for model in MODEL_ORDER:
        combo_df = all_results[(all_results["category"] == category) & (all_results["model"] == model)]
        if combo_df.empty:
            continue
        front_df = pd.DataFrame(
            pareto_frontier(combo_df.to_dict("records"), engagement_key="NDCG@10", carbon_key="avg_carbon_kg")
        ).sort_values("avg_carbon_kg")
        ax.scatter(combo_df["avg_carbon_kg"], combo_df["NDCG@10"], color=MODEL_COLORS[model], s=20, alpha=0.25)
        ax.plot(
            front_df["avg_carbon_kg"],
            front_df["NDCG@10"],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("AvgPCF@10 (kg CO$_2$e)")
    ax.set_ylabel("NDCG@10")
    ax.set_title(_category_label(category))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = figure_dir / f"multimodel_pareto_{category}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cross_category_pareto(all_results: pd.DataFrame, model: str, figure_dir: Path) -> Path:
    figure_dir = _ensure_dir(figure_dir)
    fig, ax = plt.subplots(figsize=(5.2, 4))

    for category in CATEGORY_ORDER:
        combo_df = all_results[(all_results["category"] == category) & (all_results["model"] == model)]
        if combo_df.empty:
            continue
        front_df = pd.DataFrame(
            pareto_frontier(combo_df.to_dict("records"), engagement_key="NDCG@10", carbon_key="avg_carbon_kg")
        ).sort_values("avg_carbon_kg")
        ax.scatter(combo_df["avg_carbon_kg"], combo_df["NDCG@10"], color=CATEGORY_COLORS[category], s=20, alpha=0.25)
        ax.plot(
            front_df["avg_carbon_kg"],
            front_df["NDCG@10"],
            color=CATEGORY_COLORS[category],
            marker="o",
            linewidth=2,
            label=_category_label(category),
        )

    ax.set_xlabel("AvgPCF@10 (kg CO$_2$e)")
    ax.set_ylabel("NDCG@10")
    ax.set_title(model)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = figure_dir / f"cross_category_pareto_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_best_pareto_summary(best_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    if best_df.empty:
        return None

    figure_dir = _ensure_dir(figure_dir)
    x = np.arange(len(CATEGORY_ORDER))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for idx, model in enumerate(MODEL_ORDER):
        subset = best_df[best_df["model"] == model].set_index("category").reindex(CATEGORY_ORDER)
        axes[0].bar(
            x + (idx - 1) * width,
            subset["NDCG@10"],
            width=width,
            color=MODEL_COLORS[model],
            label=model,
        )
        axes[1].bar(
            x + (idx - 1) * width,
            subset["carbon_reduction_pct"],
            width=width,
            color=MODEL_COLORS[model],
            label=model,
        )

    axes[0].set_ylabel("NDCG@10")
    axes[1].set_ylabel("Carbon reduction (%)")
    axes[0].set_title("Best Pareto point")
    axes[1].set_title("Best Pareto point")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([_category_label(cat) for cat in CATEGORY_ORDER], rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()

    out_path = figure_dir / "best_pareto_summary_bar.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_carbon_reduction_heatmap(all_results: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir = _ensure_dir(figure_dir)
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        for model in MODEL_ORDER:
            combo_df = all_results[(all_results["category"] == category) & (all_results["model"] == model)]
            if combo_df.empty:
                continue
            rows.append(
                {
                    "category": _category_label(category),
                    "model": model,
                    "value": _max_carbon_reduction_under_cap(combo_df),
                }
            )
    if not rows:
        return None
    heatmap_df = pd.DataFrame(rows).pivot(index="category", columns="model", values="value")
    heatmap_df = heatmap_df.reindex([_category_label(cat) for cat in CATEGORY_ORDER])
    heatmap_df = heatmap_df.reindex(columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        cbar_kws={"label": "Carbon reduction (%)"},
        ax=ax,
    )
    ax.set_title("Max Carbon Reduction Subject to ≤5% NDCG Drop")
    fig.tight_layout()

    out_path = figure_dir / "carbon_reduction_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_paper_plots(
    *,
    results_dir: Path,
    figure_dir: Path,
    carbon_metrics_path: Path | None = None,
    carbon_eval_predictions_path: Path | None = None,
    amazon_predictions_path: Path | None = None,
    categories: list[str] | None = None,
    models: list[str] | None = None,
    summary_output_dir: Path | None = None,
) -> dict[str, Any]:
    categories = categories or CATEGORY_ORDER
    models = models or MODEL_ORDER
    figure_dir = _ensure_dir(Path(figure_dir))
    summary_output_dir = _ensure_dir(Path(summary_output_dir or results_dir))

    metrics_by_combo, all_results = load_reranking_results(Path(results_dir), categories, models)
    best_df = _build_best_points_df(all_results)
    all_results.to_csv(summary_output_dir / "paper_reranking_summary.csv", index=False)
    best_df.to_csv(summary_output_dir / "paper_best_pareto_points.csv", index=False)

    generated: dict[str, Any] = {
        "figures": [],
        "summary_csv": str(summary_output_dir / "paper_reranking_summary.csv"),
        "best_points_csv": str(summary_output_dir / "paper_best_pareto_points.csv"),
    }

    if carbon_metrics_path is not None and Path(carbon_metrics_path).exists():
        metrics_df = pd.read_csv(carbon_metrics_path)
        path = plot_pcf_method_comparison(metrics_df, figure_dir)
        if path is not None:
            generated["figures"].append(str(path))

    if carbon_eval_predictions_path is not None and Path(carbon_eval_predictions_path).exists():
        predictions_df = pd.read_csv(carbon_eval_predictions_path)
        for method in METHOD_ORDER:
            path = plot_pcf_scatter(predictions_df, method, figure_dir)
            if path is not None:
                generated["figures"].append(str(path))

    if amazon_predictions_path is not None and Path(amazon_predictions_path).exists():
        amazon_df = pd.read_csv(amazon_predictions_path)
        for category in categories:
            path = plot_pcf_distribution(amazon_df, category, figure_dir)
            if path is not None:
                generated["figures"].append(str(path))

    for (category, model), combo_df in metrics_by_combo.items():
        generated["figures"].append(str(plot_lambda_sensitivity(combo_df, category, model, figure_dir)))
        generated["figures"].append(str(plot_pareto(combo_df, category, model, figure_dir)))

    for category in categories:
        generated["figures"].append(str(plot_multimodel_pareto(all_results, category, figure_dir)))
    for model in models:
        generated["figures"].append(str(plot_cross_category_pareto(all_results, model, figure_dir)))

    summary_bar = plot_best_pareto_summary(best_df, figure_dir)
    if summary_bar is not None:
        generated["figures"].append(str(summary_bar))
    heatmap_path = plot_carbon_reduction_heatmap(all_results, figure_dir)
    if heatmap_path is not None:
        generated["figures"].append(str(heatmap_path))

    manifest_path = summary_output_dir / "paper_plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(generated, handle, indent=2)
    generated["manifest"] = str(manifest_path)
    return generated
