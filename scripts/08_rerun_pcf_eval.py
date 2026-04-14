#!/usr/bin/env python
"""
Recompute PCF evaluation metrics for full and consumer-scale subsets.

This script is intentionally post-hoc and lightweight: it does not rerun LLM
inference. It recomputes metrics from an existing row-level predictions CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRUE_PCF_COL = "pcf"
CONSUMER_SCALE_MAX_KG = 10_000.0

METHOD_TO_COL = {
    "neighbor_average": "neighbor_average_pcf",
    "zero_shot_llm": "zero_shot_llm_pcf",
    "few_shot_llm": "few_shot_llm_pcf",
    "selected_pcf": "selected_pcf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute PCF evaluation metrics for full and consumer-scale subsets."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("output/pcf/pcf_evaluation_predictions.csv"),
        help="Row-level PCF predictions CSV (with true pcf and method columns).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("output/pcf/pcf_evaluation_metrics_by_subset.csv"),
        help="Output CSV for subset-aware method metrics.",
    )
    parser.add_argument(
        "--consumer-max-pcf",
        type=float,
        default=CONSUMER_SCALE_MAX_KG,
        help="Upper true-PCF threshold for consumer-scale subset.",
    )
    return parser.parse_args()


def _build_selected_pcf(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "few_shot_llm_pcf" not in result.columns or "neighbor_average_pcf" not in result.columns:
        raise ValueError(
            "Missing required columns for selected_pcf: need few_shot_llm_pcf and neighbor_average_pcf."
        )
    result["selected_pcf"] = result["few_shot_llm_pcf"].where(
        result["few_shot_llm_pcf"].notna(),
        result["neighbor_average_pcf"],
    )
    return result


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].astype(float)
    y_pred = y_pred[valid].astype(float)
    n = int(len(y_true))
    if n == 0:
        return {
            "n_examples": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "median_ae": np.nan,
            "spearman": np.nan,
            "available": False,
        }

    abs_err = (y_pred - y_true).abs()
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(abs_err.mean())
    median_ae = float(abs_err.median())
    spearman = float(y_true.rank(method="average").corr(y_pred.rank(method="average"), method="pearson"))
    return {
        "n_examples": n,
        "rmse": rmse,
        "mae": mae,
        "median_ae": median_ae,
        "spearman": spearman,
        "available": True,
    }


def _evaluate_subset(df: pd.DataFrame, subset_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, pred_col in METHOD_TO_COL.items():
        if pred_col not in df.columns:
            rows.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "n_examples": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "median_ae": np.nan,
                    "spearman": np.nan,
                    "available": False,
                }
            )
            continue

        stats = _compute_metrics(df[TRUE_PCF_COL], df[pred_col])
        rows.append(
            {
                "subset": subset_name,
                "method": method,
                **stats,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions_path)
    if TRUE_PCF_COL not in predictions.columns:
        raise ValueError(f"Missing required true PCF column: {TRUE_PCF_COL}")

    predictions = _build_selected_pcf(predictions)
    predictions[TRUE_PCF_COL] = pd.to_numeric(predictions[TRUE_PCF_COL], errors="coerce")

    full_df = predictions.copy()
    consumer_df = predictions[predictions[TRUE_PCF_COL] <= float(args.consumer_max_pcf)].copy()

    # Fair comparison: same rows for every method (consumer-scale AND all base predictions present).
    base_pred_cols = ["neighbor_average_pcf", "zero_shot_llm_pcf", "few_shot_llm_pcf"]
    if all(c in consumer_df.columns for c in base_pred_cols):
        intersection_mask = consumer_df[base_pred_cols].notna().all(axis=1)
        consumer_intersection_df = consumer_df[intersection_mask].copy()
    else:
        consumer_intersection_df = consumer_df.iloc[0:0].copy()

    rows: list[dict[str, object]] = []
    rows.extend(_evaluate_subset(full_df, "full_holdout"))
    rows.extend(_evaluate_subset(consumer_df, f"consumer_scale_true_pcf_le_{int(args.consumer_max_pcf)}"))
    if not consumer_intersection_df.empty:
        rows.extend(
            _evaluate_subset(
                consumer_intersection_df,
                "consumer_scale_intersection_all_methods",
            )
        )

    metrics = pd.DataFrame(rows)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    print(f"Saved: {args.metrics_output}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
