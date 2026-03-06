#!/usr/bin/env python
"""
Train DeepFM Engagement Predictor (Pipeline Step 2)
=====================================================
Trains a DeepFM model on interim interaction data and scores
RecBole candidate items with engagement probabilities.

Pipeline context:
    1. RecBole → top-K candidates with relevance scores
    2. **This script** → DeepFM engagement prediction
    3. Carbon re-ranking → re-ranked lists at various λ
    4. Evaluation → engagement vs carbon footprint trade-off

Inputs:
    data/interim/{train,val,test}/<category>.csv            (features + labels)
    output/results/<category>_<model>_scores.parquet         (from train_recommender.py)

Outputs:
    output/results/<category>_<model>_engagement.parquet     (engagement scores)
    output/models/deepfm_<category>_<model>.joblib           (saved model)
    output/models/deepfm_<category>_<model>_history.json     (training history)

Usage:
    python scripts/train_deepfm.py                          # all categories
    python scripts/train_deepfm.py --category electronics
    python scripts/train_deepfm.py --model LightGCN
    python scripts/train_deepfm.py --config configs/deepfm/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepFM engagement predictor (pipeline step 2)",
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
        help="RecBole model name matching scores file (default: BPR)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to DeepFM YAML config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu or cuda (default: from config)",
    )
    args = parser.parse_args()

    # Build config overrides from CLI args
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device

    categories = [args.category] if args.category else ALL_CATEGORIES

    from src.engagement.train import train_and_score

    for cat in categories:
        log.info("=" * 60)
        log.info("DeepFM: %s (RecBole model=%s)", cat, args.model)
        log.info("=" * 60)

        try:
            engagement_df, history = train_and_score(
                category=cat,
                model_name=args.model,
                config_path=args.config,
                config_overrides=overrides if overrides else None,
            )

            log.info(
                "  %s engagement scores generated (median=%.4f)",
                f"{len(engagement_df):,}",
                engagement_df["engagement_score"].median(),
            )

        except FileNotFoundError as e:
            log.error("  Skipping %s: %s", cat, e)
            continue

    log.info("DeepFM training complete. Run rerank.py next.")


if __name__ == "__main__":
    main()
