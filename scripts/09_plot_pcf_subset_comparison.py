#!/usr/bin/env python
"""
Plot full-holdout vs consumer-scale PCF evaluation metrics.
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
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot subset-aware PCF metrics.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("output/pcf/pcf_evaluation_metrics_by_subset.csv"),
        help="CSV with subset-aware PCF metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figures/pcf_subset_eval"),
        help="Directory for output figures.",
    )
    return parser.parse_args()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["available"].fillna(False)].copy()
    out = out[out["method"].isin(["neighbor_average", "zero_shot_llm", "few_shot_llm", "selected_pcf"])].copy()
    out["subset_label"] = out["subset"].map(
        {
            "full_holdout": "Full holdout",
            "consumer_scale_true_pcf_le_10000": "Consumer scale (<=10k)",
        }
    ).fillna(out["subset"])
    out["method_label"] = out["method"].map(
        {
            "neighbor_average": "Neighbor avg",
            "zero_shot_llm": "Zero-shot LLM",
            "few_shot_llm": "Few-shot LLM",
            "selected_pcf": "Selected PCF",
        }
    )
    return out


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, log_scale: bool = False) -> None:
    plt.figure(figsize=(9, 4.8))
    ax = sns.barplot(
        data=df,
        x="method_label",
        y=metric,
        hue="subset_label",
        palette=["#2f6f6f", "#d96c06"],
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: full holdout vs consumer-scale")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="paper")
    df = pd.read_csv(args.metrics_path)
    df = _clean(df)

    plot_metric(df, "rmse", "RMSE", args.output_dir / "pcf_subset_rmse.png", log_scale=True)
    plot_metric(df, "mae", "MAE", args.output_dir / "pcf_subset_mae.png", log_scale=True)
    plot_metric(df, "median_ae", "Median absolute error", args.output_dir / "pcf_subset_median_ae.png")
    plot_metric(df, "spearman", "Spearman", args.output_dir / "pcf_subset_spearman.png")

    print(f"Saved: {args.output_dir / 'pcf_subset_rmse.png'}")
    print(f"Saved: {args.output_dir / 'pcf_subset_mae.png'}")
    print(f"Saved: {args.output_dir / 'pcf_subset_median_ae.png'}")
    print(f"Saved: {args.output_dir / 'pcf_subset_spearman.png'}")


if __name__ == "__main__":
    main()
