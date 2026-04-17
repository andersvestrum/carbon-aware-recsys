#!/usr/bin/env python
"""
Figure 4 style outputs for the paper:
  (a) Predicted PCF distribution on Amazon catalog (selected PCF column).
  (b) Predicted PCF by Carbon Catalogue sector (boxplot, selected predictor).
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

SECTOR_COL = "*Company's sector"


def _selected_predicted(df: pd.DataFrame) -> pd.Series:
    if "few_shot_llm_pcf" in df.columns and "neighbor_average_pcf" in df.columns:
        return df["few_shot_llm_pcf"].where(
            df["few_shot_llm_pcf"].notna(),
            df["neighbor_average_pcf"],
        )
    if "pcf" in df.columns:
        return df["pcf"]
    raise ValueError("Cannot build selected predicted PCF series.")


def plot_amazon_distribution(
    df: pd.DataFrame,
    *,
    out_path: Path,
    upper_quantile: float,
) -> None:
    if df.empty or "pcf" not in df.columns:
        _placeholder(out_path, "Amazon PCF predictions missing or empty.\nRun predict_carbon.py to populate.")
        return
    s = pd.to_numeric(df["pcf"], errors="coerce")
    s = s[s.notna() & (s > 0)]
    if s.empty:
        _placeholder(out_path, "No positive Amazon PCF values to plot.")
        return

    cap = float(s.quantile(upper_quantile))
    clipped = s[s <= cap]
    if clipped.empty:
        clipped = s

    plt.figure(figsize=(8.2, 4.8))
    log_bins = np.logspace(np.log10(clipped.min()), np.log10(clipped.max()), 45)
    ax = sns.histplot(clipped, bins=log_bins, color="#1f5d5d")
    ax.set_xscale("log")
    ax.set_title(f"Amazon catalog: predicted PCF (≤{int(upper_quantile * 100)}th pct.)")
    ax.set_xlabel("Predicted PCF (kg CO2e, log scale)")
    ax.set_ylabel("Product count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_carbon_sector_boxplot(
    df: pd.DataFrame,
    *,
    out_path: Path,
    top_n: int,
) -> None:
    if SECTOR_COL not in df.columns:
        raise ValueError(f"Missing {SECTOR_COL} in Carbon Catalogue predictions.")
    plot_df = df.copy()
    plot_df["_pred"] = _selected_predicted(plot_df)
    plot_df = plot_df[[SECTOR_COL, "_pred"]].dropna()
    plot_df = plot_df[plot_df["_pred"] > 0]
    plot_df[SECTOR_COL] = plot_df[SECTOR_COL].fillna("Unknown")

    top_sectors = (
        plot_df.groupby(SECTOR_COL)["_pred"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = plot_df[plot_df[SECTOR_COL].isin(top_sectors)].copy()
    order = (
        plot_df.groupby(SECTOR_COL)["_pred"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(10, 5.2))
    ax = sns.boxplot(
        data=plot_df,
        y=SECTOR_COL,
        x="_pred",
        order=order,
        color="#7e4b90",
        showfliers=False,
    )
    ax.set_xscale("log")
    ax.set_title("Carbon Catalogue: predicted PCF by sector (top categories)")
    ax.set_xlabel("Predicted PCF (kg CO2e, log scale)")
    ax.set_ylabel("Sector")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _placeholder(out_path: Path, message: str) -> None:
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--amazon-path",
        type=Path,
        default=Path("data/processed/carbon/amazon_pcf_predictions.csv"),
    )
    p.add_argument(
        "--carbon-eval-path",
        type=Path,
        default=Path("output/pcf/pcf_evaluation_predictions.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figures/pcf_insights"),
    )
    p.add_argument("--top-n-sectors", type=int, default=12)
    p.add_argument("--distribution-upper-quantile", type=float, default=0.99)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="paper")

    if args.amazon_path.exists() and args.amazon_path.stat().st_size > 10:
        try:
            amazon_df = pd.read_csv(args.amazon_path)
        except (pd.errors.EmptyDataError, ValueError):
            amazon_df = pd.DataFrame()
    else:
        amazon_df = pd.DataFrame()
    carbon_df = pd.read_csv(args.carbon_eval_path)

    plot_amazon_distribution(
        amazon_df,
        out_path=args.output_dir / "amazon_pcf_distribution_paper.png",
        upper_quantile=args.distribution_upper_quantile,
    )
    plot_carbon_sector_boxplot(
        carbon_df,
        out_path=args.output_dir / "carbon_catalogue_pcf_by_sector_paper.png",
        top_n=args.top_n_sectors,
    )

    docs_fig = Path(__file__).resolve().parents[1] / "docs" / "figures"
    docs_fig.mkdir(parents=True, exist_ok=True)
    for name in ("amazon_pcf_distribution_paper.png", "carbon_catalogue_pcf_by_sector_paper.png"):
        src = args.output_dir / name
        if src.exists():
            import shutil

            shutil.copy2(src, docs_fig / name)

    print(f"Saved: {args.output_dir / 'amazon_pcf_distribution_paper.png'}")
    print(f"Saved: {args.output_dir / 'carbon_catalogue_pcf_by_sector_paper.png'}")
    print(f"Copied to docs/figures/ (if generated)")


if __name__ == "__main__":
    main()
