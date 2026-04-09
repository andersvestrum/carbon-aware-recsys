#!/usr/bin/env python
"""
Carbon-Aware Re-ranking (Pipeline Step 2)
==========================================
Re-ranks candidate items using RecBole relevance scores as the engagement signal:

    s(u, i; λ) = (1 − λ) · engagement_norm − λ · carbon_norm

Sweeps λ ∈ [0, 1] and computes ranking-quality + carbon metrics
at each operating point.

Pipeline context:
    1. RecBole → top-K candidates with relevance scores
    2. **This script** → carbon-aware re-ranked lists + metrics per λ
    3. Evaluation → engagement vs carbon footprint trade-off

Inputs:
    output/results/<category>_<model>_scores.parquet       (from 01_train_recommender.py)
    data/interim/{train,val,test}/<category>.csv           (has pcf column)

Outputs:
    output/results/<category>_<model>_reranked_<λ>.parquet  — re-ranked lists
    output/results/<category>_<model>_reranking_metrics.json — metrics per λ

Usage:
    python scripts/02_rerank.py                              # all categories
    python scripts/02_rerank.py --category electronics
    python scripts/02_rerank.py --model LightGCN
    python scripts/02_rerank.py --lambda-values 0.0 0.1 0.5  # specific λ values
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import SUPPORTED_MODELS, canonical_model_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "output" / "results"
CONFIG_DIR = PROJECT_ROOT / "configs" / "reranking"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

ALL_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]
DEFAULT_LAMBDA_VALUES = [
    0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
]


def load_config(config_path: str | Path | None = None) -> dict:
    """Load re-ranking config from YAML (or use defaults)."""
    if config_path is None:
        config_path = CONFIG_DIR / "default.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        log.info("Loaded reranking config from %s", config_path)
        return cfg

    # Fallback defaults
    return {
        "lambda_values": DEFAULT_LAMBDA_VALUES,
        "top_k": 10,
    }


def run_reranking(
    category: str,
    model_name: str = "BPR",
    lambda_values: list[float] | None = None,
    top_k: int = 10,
    results_dir: Path | None = None,
    interim_dir: Path | None = None,
) -> dict:
    """Run carbon-aware re-ranking for one category.

    Returns:
        Dict with ``metrics`` (list of dicts per λ) and ``summary``.
    """
    import pandas as pd
    from src.recommender.recbole_formatter import _concat_interim_splits, load_interim_split
    from src.reranking.carbon_reranker import (
        CarbonReranker,
        compute_reranking_metrics,
    )

    # ── Load relevance scores from step 1 (RecBole) ──────────────────
    model_name = canonical_model_name(model_name)
    results_dir = results_dir or RESULTS_DIR
    interim_dir = interim_dir or INTERIM_DIR
    scores_path = results_dir / f"{category}_{model_name}_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Relevance scores not found for {category}/{model_name}. "
            "Run 01_train_recommender.py first."
        )

    scores_df = pd.read_parquet(scores_path)

    if "relevance_score" in scores_df.columns:
        scores_df = scores_df.rename(columns={"relevance_score": "engagement_score"})

    log.info(
        "Loaded %s relevance scores from %s",
        f"{len(scores_df):,}",
        scores_path.name,
    )

    # ── Load item carbon footprints from interim data ─────────────────
    interactions = _concat_interim_splits(category, interim_dir=interim_dir)

    # Build carbon lookup from interim data (has pcf column)
    carbon_df = (
        interactions[["parent_asin", "pcf"]]
        .drop_duplicates(subset="parent_asin")
        .copy()
    )
    log.info(
        "Carbon data: %s items, median pcf = %.2f kg CO₂e",
        f"{len(carbon_df):,}",
        carbon_df["pcf"].median(),
    )

    # ── Build test set ────────────────────────────────────────────────
    test_interactions = (
        load_interim_split(category, "test", interim_dir=interim_dir)
        [["user_id", "parent_asin", "timestamp"]]
        .sort_values("timestamp")
        .drop_duplicates(subset="user_id", keep="last")
        [["user_id", "parent_asin"]]
    )
    log.info("Test set: %s users", f"{len(test_interactions):,}")

    # ── Default λ values ──────────────────────────────────────────────
    if lambda_values is None:
        lambda_values = DEFAULT_LAMBDA_VALUES

    # ── Sweep λ ───────────────────────────────────────────────────────
    reranker = CarbonReranker(top_k=top_k)
    all_metrics: list[dict] = []

    results_dir.mkdir(parents=True, exist_ok=True)

    for lam in lambda_values:
        ranked = reranker.rerank(scores_df, carbon_df, lam)

        metrics = compute_reranking_metrics(
            ranked, test_interactions, lam, k=top_k,
        )
        all_metrics.append(metrics)

        # Save re-ranked lists for each λ value
        lam_str = f"{lam:.3f}"
        out_path = results_dir / f"{category}_{model_name}_reranked_{lam_str}.parquet"
        ranked.to_parquet(out_path, index=False)

        log.info(
            "  λ=%5.3f  NDCG@%d=%.4f  carbon=%.2f kg  (%s users)",
            lam,
            top_k,
            metrics[f"NDCG@{top_k}"],
            metrics["avg_carbon_kg"],
            metrics["n_users"],
        )

    # ── Summary ───────────────────────────────────────────────────────
    baseline = next(m for m in all_metrics if m["lambda"] == 0.0)
    greenest = min(all_metrics, key=lambda m: m["avg_carbon_kg"])

    summary = {
        "category": category,
        "model": model_name,
        "baseline_carbon_kg": baseline["avg_carbon_kg"],
        f"baseline_NDCG@{top_k}": baseline[f"NDCG@{top_k}"],
        "greenest_lambda": greenest["lambda"],
        "greenest_carbon_kg": greenest["avg_carbon_kg"],
        f"greenest_NDCG@{top_k}": greenest[f"NDCG@{top_k}"],
        "carbon_reduction_pct": (
            100.0
            * (baseline["avg_carbon_kg"] - greenest["avg_carbon_kg"])
            / baseline["avg_carbon_kg"]
            if baseline["avg_carbon_kg"] > 0
            else 0.0
        ),
    }

    # Save metrics
    metrics_path = results_dir / f"{category}_{model_name}_reranking_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {"summary": summary, "per_lambda": all_metrics},
            f,
            indent=2,
        )
    log.info("Saved metrics → %s", metrics_path)

    return {"metrics": all_metrics, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Carbon-aware re-ranking (pipeline step 2)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=ALL_CATEGORIES,
        default=None,
        help="Category to re-rank (default: all three)",
    )
    parser.add_argument(
        "--model",
        type=canonical_model_name,
        choices=SUPPORTED_MODELS,
        default="BPR",
        help="Candidate generation model used in the paper (default: BPR)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to reranking YAML config",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit λ values to sweep (overrides config)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of items per user after re-ranking (default: from config)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing score files and reranking outputs",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_DIR,
        help="Directory containing interim train/val/test CSVs",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    lambda_values = args.lambda_values or cfg.get("lambda_values")
    top_k = args.top_k or cfg.get("top_k", 10)

    categories = [args.category] if args.category else ALL_CATEGORIES

    for cat in categories:
        log.info("=" * 60)
        log.info("Re-ranking: %s (model=%s)", cat, args.model)
        log.info("=" * 60)

        result = run_reranking(
            category=cat,
            model_name=args.model,
            lambda_values=lambda_values,
            top_k=top_k,
            results_dir=args.results_dir,
            interim_dir=args.interim_dir,
        )

        s = result["summary"]
        log.info(
            "  Baseline carbon: %.2f kg → Greenest (λ=%.3f): %.2f kg  "
            "(−%.1f%%)",
            s["baseline_carbon_kg"],
            s["greenest_lambda"],
            s["greenest_carbon_kg"],
            s["carbon_reduction_pct"],
        )

    log.info("Re-ranking complete. Run evaluation next.")


if __name__ == "__main__":
    main()
