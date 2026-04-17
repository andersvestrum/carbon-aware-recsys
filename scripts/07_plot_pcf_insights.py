#!/usr/bin/env python
"""
Generate PCF insight plots for category-level and distribution analysis.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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


def _set_plot_style(paper_style: bool) -> None:
    if paper_style:
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update(
            {
                "figure.dpi": 140,
                "savefig.dpi": 300,
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )
    else:
        sns.set_theme(style="whitegrid", context="notebook")


def _resolve_category_column(df: pd.DataFrame) -> str | None:
    candidates = ["source_category", "main_category", "*Company's sector"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _ensure_predicted_pcf_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Column used for plots should reflect *predicted* PCF (downstream signal), not
    ground-truth `pcf` in Carbon Catalogue evaluation files.
    """
    out = df.copy()
    if "few_shot_llm_pcf" in out.columns and "neighbor_average_pcf" in out.columns:
        out["_plot_predicted_pcf"] = out["few_shot_llm_pcf"].where(
            out["few_shot_llm_pcf"].notna(),
            out["neighbor_average_pcf"],
        )
        return out, "_plot_predicted_pcf"
    if "pcf" in out.columns and "few_shot_llm_pcf" not in out.columns:
        return out, "pcf"
    for col in ("few_shot_llm_pcf", "neighbor_average_pcf", "zero_shot_llm_pcf"):
        if col in out.columns:
            return out, col
    raise ValueError("Could not resolve a predicted-PCF column for plotting.")


def plot_least_green_categories(
    df: pd.DataFrame,
    *,
    category_col: str,
    pcf_col: str,
    out_path: Path,
    top_n: int,
    paper_style: bool,
) -> None:
    summary = (
        df[[category_col, pcf_col]]
        .dropna(subset=[category_col, pcf_col])
        .groupby(category_col, as_index=False)
        .agg(
            avg_pcf=(pcf_col, "mean"),
            median_pcf=(pcf_col, "median"),
            count=(pcf_col, "size"),
        )
        .sort_values("avg_pcf", ascending=False)
        .head(top_n)
    )
    if summary.empty:
        raise ValueError("No valid rows for least-green category plot.")

    fig_size = (9.2, 5.0) if paper_style else (11, 6)
    bar_color = "#b74d00" if paper_style else "#d96c06"
    plt.figure(figsize=fig_size)
    ax = sns.barplot(data=summary, y=category_col, x="avg_pcf", color=bar_color)
    ax.set_title("Least-green categories by predicted average PCF")
    ax.set_xlabel("Average predicted PCF (kg CO2e)")
    ax.set_ylabel("Category")

    for idx, row in summary.reset_index(drop=True).iterrows():
        ax.text(
            row["avg_pcf"],
            idx,
            f" n={int(row['count'])}",
            va="center",
            ha="left",
            fontsize=8 if paper_style else 9,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pcf_distribution(
    df: pd.DataFrame,
    *,
    pcf_col: str,
    out_path: Path,
    paper_style: bool,
    upper_quantile: float,
) -> None:
    series = pd.to_numeric(df[pcf_col], errors="coerce")
    series = series[series.notna() & (series > 0)]
    if series.empty:
        raise ValueError("No positive PCF values found for distribution plot.")

    # Cap extreme outliers so the main body of the distribution is visible.
    cap = float(series.quantile(upper_quantile))
    clipped = series[series <= cap]
    if clipped.empty:
        clipped = series

    fig_size = (8.2, 4.8) if paper_style else (10, 5)
    hist_color = "#1f5d5d" if paper_style else "#2f6f6f"
    bins = 40 if paper_style else 50
    log_bins = np.logspace(np.log10(clipped.min()), np.log10(clipped.max()), bins)
    plt.figure(figsize=fig_size)
    ax = sns.histplot(clipped, bins=log_bins, color=hist_color)
    ax.set_xscale("log")
    ax.set_title(f"Predicted PCF distribution (up to {int(upper_quantile * 100)}th percentile)")
    ax.set_xlabel("Predicted PCF (kg CO2e, log scale)")
    ax.set_ylabel("Product count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pcf_boxplot_by_category(
    df: pd.DataFrame,
    *,
    category_col: str,
    pcf_col: str,
    out_path: Path,
    top_n: int,
    paper_style: bool,
) -> None:
    plot_df = df[[category_col, pcf_col]].copy()
    plot_df[pcf_col] = pd.to_numeric(plot_df[pcf_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[category_col, pcf_col])
    plot_df = plot_df[plot_df[pcf_col] > 0]
    if plot_df.empty:
        raise ValueError("No valid rows for category boxplot.")

    top_categories = (
        plot_df.groupby(category_col)[pcf_col]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = plot_df[plot_df[category_col].isin(top_categories)].copy()
    order = (
        plot_df.groupby(category_col)[pcf_col]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig_size = (10, 5.2) if paper_style else (12, 6)
    box_color = "#7e4b90" if paper_style else "#8d5a97"
    plt.figure(figsize=fig_size)
    ax = sns.boxplot(
        data=plot_df,
        y=category_col,
        x=pcf_col,
        order=order,
        color=box_color,
        showfliers=False,
    )
    ax.set_xscale("log")
    ax.set_title("Predicted PCF distribution by category (boxplot)")
    ax.set_xlabel("Predicted PCF (kg CO2e, log scale)")
    ax.set_ylabel("Category")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PCF insight plots")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("output/results/carbon/pcf_evaluation_predictions.csv"),
        help="CSV file with product-level PCF predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figures/pcf_insights"),
        help="Directory where plots will be saved",
    )
    parser.add_argument(
        "--top-n-categories",
        type=int,
        default=12,
        help="Number of highest-average-PCF categories to plot",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Use cleaner publication-oriented style settings",
    )
    parser.add_argument(
        "--distribution-upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile cap for readable PCF histogram (0-1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_plot_style(args.paper_style)
    df = pd.read_csv(args.predictions_path)

    df, pcf_col = _ensure_predicted_pcf_column(df)
    category_col = _resolve_category_column(df)
    if category_col is None:
        raise ValueError(
            "No category column found. Expected one of: source_category, main_category, *Company's sector"
        )

    plot_least_green_categories(
        df,
        category_col=category_col,
        pcf_col=pcf_col,
        out_path=args.output_dir
        / ("least_green_categories_paper.png" if args.paper_style else "least_green_categories.png"),
        top_n=args.top_n_categories,
        paper_style=args.paper_style,
    )
    plot_pcf_distribution(
        df,
        pcf_col=pcf_col,
        out_path=args.output_dir
        / ("pcf_distribution_paper.png" if args.paper_style else "pcf_distribution.png"),
        paper_style=args.paper_style,
        upper_quantile=args.distribution_upper_quantile,
    )
    plot_pcf_boxplot_by_category(
        df,
        category_col=category_col,
        pcf_col=pcf_col,
        out_path=args.output_dir
        / (
            "pcf_boxplot_by_category_paper.png"
            if args.paper_style
            else "pcf_boxplot_by_category.png"
        ),
        top_n=args.top_n_categories,
        paper_style=args.paper_style,
    )

    if args.paper_style:
        print(f"Saved: {args.output_dir / 'least_green_categories_paper.png'}")
        print(f"Saved: {args.output_dir / 'pcf_distribution_paper.png'}")
        print(f"Saved: {args.output_dir / 'pcf_boxplot_by_category_paper.png'}")
    else:
        print(f"Saved: {args.output_dir / 'least_green_categories.png'}")
        print(f"Saved: {args.output_dir / 'pcf_distribution.png'}")
        print(f"Saved: {args.output_dir / 'pcf_boxplot_by_category.png'}")


if __name__ == "__main__":
    main()
