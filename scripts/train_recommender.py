#!/usr/bin/env python
"""
Candidate Generation — RecBole (Pipeline Step 1)
=================================================
Trains a collaborative filtering model and extracts top-K relevance
scores per user.  These scores feed into the downstream pipeline:

  1. **RecBole** (this script) → top-K candidates with relevance scores
  2. DeepFM engagement prediction on candidates
  3. Carbon-aware re-ranking:  score = (1−λ)·engagement − λ·carbon
  4. Evaluation: engagement vs carbon footprint trade-off

Supported models: BPR, NeuMF, SASRec, LightGCN (any RecBole model).

Outputs:
  output/results/<category>_<model>_scores.parquet  — (user_id, parent_asin, relevance_score)
  output/results/<category>_<model>_eval.json       — NDCG@10, Recall@10, MRR

Run after:  preprocess.py  (which produces data/interim/ files)
Run before: train_deepfm.py

Usage:
  python scripts/train_recommender.py                       # all categories, BPR
  python scripts/train_recommender.py --category electronics
  python scripts/train_recommender.py --model NeuMF
  python scripts/train_recommender.py --model SASRec
  python scripts/train_recommender.py --model LightGCN
  python scripts/train_recommender.py --fallback             # numpy BPR (no RecBole)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "output" / "results"
CONFIG_DIR = PROJECT_ROOT / "configs" / "recbole"

ALL_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]


def run_recbole_pipeline(
    category: str,
    model_name: str = "BPR",
    config_file: str | Path | None = None,
    top_k: int = 100,
) -> tuple:
    """Format data and train with RecBole."""
    from src.recommender.recbole_formatter import format_category_for_recbole
    from src.recommender.trainer import train_and_score

    log.info("Formatting %s for RecBole …", category)
    _output_dir, dataset_name = format_category_for_recbole(category)

    if config_file is None:
        default_cfg = CONFIG_DIR / f"{model_name.lower()}.yaml"
        if default_cfg.exists():
            config_file = default_cfg

    log.info("Training %s on %s …", model_name, dataset_name)
    scores_df, eval_results = train_and_score(
        dataset_name=dataset_name,
        model_name=model_name,
        config_file=config_file,
        top_k=top_k,
    )
    return scores_df, eval_results


def run_fallback_pipeline(
    category: str,
    top_k: int = 100,
) -> tuple:
    """Load interim data and train with the numpy fallback."""
    from src.recommender.recbole_formatter import _concat_interim_splits
    from src.recommender.bpr_fallback import train_bpr_numpy, evaluate_bpr

    interactions = _concat_interim_splits(category)
    scores_df = train_bpr_numpy(interactions, top_k=top_k)
    eval_results = evaluate_bpr(scores_df, interactions, k=10)

    # Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIR / f"{category}_BPR_fallback_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)
    log.info("Saved scores → %s", scores_path)

    eval_path = RESULTS_DIR / f"{category}_BPR_fallback_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    log.info("Saved eval → %s", eval_path)

    return scores_df, eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Train RecBole recommender and extract relevance scores (pipeline step 1)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category to train on (default: all three)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BPR",
        help="RecBole model name: BPR, NeuMF, SASRec, LightGCN, … (default: BPR)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to RecBole YAML config file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidate items per user (default: 100)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force numpy fallback (skip RecBole)",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else ALL_CATEGORIES

    # Detect RecBole availability
    recbole_available = False
    if not args.fallback:
        try:
            import recbole  # noqa: F401
            recbole_available = True
            log.info("RecBole detected — using full pipeline")
        except ImportError:
            log.warning(
                "RecBole not installed — falling back to numpy BPR. "
                "Install with: pip install recbole torch"
            )

    for cat in categories:
        log.info("=" * 60)
        log.info("Category: %s", cat)
        log.info("=" * 60)

        if recbole_available:
            scores_df, eval_results = run_recbole_pipeline(
                category=cat,
                model_name=args.model,
                config_file=args.config,
                top_k=args.top_k,
            )
        else:
            scores_df, eval_results = run_fallback_pipeline(
                category=cat,
                top_k=args.top_k,
            )

        log.info("Results for %s: %s", cat, eval_results)

    log.info("Candidate generation complete. Run train_deepfm.py next.")


if __name__ == "__main__":
    main()
