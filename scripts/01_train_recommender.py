#!/usr/bin/env python
"""
Candidate Generation — RecBole (Pipeline Step 1)
=================================================
Trains a collaborative filtering model and extracts top-K relevance
scores per user.  These scores feed into the downstream pipeline:

  1. **RecBole** (this script) → top-K candidates with relevance scores
  2. Carbon-aware re-ranking using RecBole relevance scores
  3. Carbon-aware re-ranking:  score = (1−λ)·engagement − λ·carbon
  4. Evaluation: engagement vs carbon footprint trade-off

Supported models: BPR, NeuMF, LightGCN.

Outputs:
  output/results/<category>_<model>_scores.parquet  — (user_id, parent_asin, relevance_score)
  output/results/<category>_<model>_eval.json       — NDCG@10, Recall@10, MRR

Run after:  preprocess.py  (which produces data/interim/ files)
Run before: 02_rerank.py

Usage:
  python scripts/01_train_recommender.py                       # all categories, BPR
  python scripts/01_train_recommender.py --category electronics
  python scripts/01_train_recommender.py --model NeuMF
  python scripts/01_train_recommender.py --model LightGCN
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

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
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RECBOLE_DIR = PROJECT_ROOT / "data" / "processed" / "recbole"
MODEL_DIR = PROJECT_ROOT / "output" / "models"

ALL_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]

from src.recommender import SUPPORTED_MODELS, canonical_model_name


def build_scoring_context(
    category: str,
    score_split: str,
    interim_dir: Path,
) -> tuple[list[str], dict[str, list[str]]]:
    from src.recommender.recbole_formatter import load_interim_split

    split_alias = {"val": "val", "test": "test"}
    target_split = split_alias[score_split]
    target_df = load_interim_split(category, target_split, interim_dir=interim_dir)
    target_users = target_df["user_id"].astype(str).drop_duplicates().tolist()

    seen_frames = [load_interim_split(category, "train", interim_dir=interim_dir)]
    if target_split == "test":
        seen_frames.append(load_interim_split(category, "val", interim_dir=interim_dir))
    seen_df = pd.concat(seen_frames, ignore_index=True)
    seen_items = (
        seen_df.groupby("user_id")["parent_asin"]
        .agg(lambda values: sorted({str(v) for v in values}))
        .to_dict()
    )
    seen_items = {str(user_id): item_ids for user_id, item_ids in seen_items.items()}
    return target_users, seen_items


def run_recbole_pipeline(
    category: str,
    model_name: str = "BPR",
    config_file: str | Path | None = None,
    top_k: int = 100,
    interim_dir: Path | None = None,
    recbole_dir: Path | None = None,
    results_dir: Path | None = None,
    model_dir: Path | None = None,
    score_split: str = "test",
    user_batch_size: int = 256,
    max_users: int | None = None,
    train_overrides: dict[str, object] | None = None,
    force_reformat: bool = False,
    skip_existing: bool = True,
) -> tuple:
    """Format data and train with RecBole."""
    from src.recommender.recbole_formatter import format_category_for_recbole
    from src.recommender.trainer import train_and_score

    if interim_dir is None:
        interim_dir = INTERIM_DIR
    if recbole_dir is None:
        recbole_dir = RECBOLE_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR
    if model_dir is None:
        model_dir = MODEL_DIR

    _output_dir, dataset_name = format_category_for_recbole(
        category,
        dataset_name=category,
        max_users=max_users,
        interim_dir=interim_dir,
        output_root=recbole_dir,
        benchmark_splits=True,
        force=force_reformat,
    )

    if config_file is None:
        default_cfg = CONFIG_DIR / f"{model_name.lower()}.yaml"
        if default_cfg.exists():
            config_file = default_cfg

    scores_path = results_dir / f"{dataset_name}_{model_name}_scores.parquet"
    eval_path = results_dir / f"{dataset_name}_{model_name}_eval.json"
    if skip_existing and scores_path.exists() and eval_path.exists():
        log.info("Using cached training outputs for %s/%s", category, model_name)
        with eval_path.open() as handle:
            eval_results = json.load(handle)
        return pd.read_parquet(scores_path), eval_results

    target_users, seen_items = build_scoring_context(category, score_split, interim_dir)

    log.info("Training %s on %s …", model_name, dataset_name)
    scores_df, eval_results = train_and_score(
        dataset_name=dataset_name,
        model_name=model_name,
        config_file=config_file,
        overrides=train_overrides,
        top_k=top_k,
        output_path=scores_path,
        data_path=recbole_dir,
        checkpoint_dir=model_dir / "recbole_checkpoints",
        results_dir=results_dir,
        benchmark_splits=True,
        external_user_ids=target_users,
        seen_items_by_user=seen_items,
        user_batch_size=user_batch_size,
    )
    return scores_df, eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Train RecBole recommender and extract relevance scores (pipeline step 1)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=ALL_CATEGORIES,
        default=None,
        help="Category to train on (default: all three)",
    )
    parser.add_argument(
        "--model",
        type=canonical_model_name,
        choices=SUPPORTED_MODELS,
        default="BPR",
        help="RecBole model name used in the paper (default: BPR)",
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
        "--score-split",
        choices=["val", "test"],
        default="test",
        help="Which held-out split to score users from (default: test)",
    )
    parser.add_argument(
        "--user-batch-size",
        type=int,
        default=256,
        help="Users per full-sort scoring batch (default: 256)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap on sampled train users while formatting data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional RecBole epochs override",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Optional RecBole train_batch_size override",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional RecBole eval_batch_size override",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional RecBole learning_rate override",
    )
    parser.add_argument(
        "--eval-step",
        type=int,
        default=None,
        help="Optional RecBole eval_step override",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_DIR,
        help="Directory containing interim train/val/test CSVs",
    )
    parser.add_argument(
        "--recbole-dir",
        type=Path,
        default=RECBOLE_DIR,
        help="Directory for cached RecBole-formatted datasets",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for cached score/eval outputs",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory for RecBole checkpoints",
    )
    parser.add_argument(
        "--force-reformat",
        action="store_true",
        help="Rewrite cached RecBole benchmark files",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Ignore cached score/eval outputs and retrain",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else ALL_CATEGORIES

    try:
        import recbole  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "RecBole is required for candidate generation. "
            "Install `recbole` and `torch` to run BPR, NeuMF, or LightGCN."
        ) from exc

    log.info("Running RecBole candidate generation with model=%s", args.model)
    train_overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "eval_step": args.eval_step,
        }.items()
        if value is not None
    }
    if train_overrides:
        log.info("Applying RecBole overrides: %s", train_overrides)

    for cat in categories:
        log.info("=" * 60)
        log.info("Category: %s", cat)
        log.info("=" * 60)

        scores_df, eval_results = run_recbole_pipeline(
            category=cat,
            model_name=args.model,
            config_file=args.config,
            top_k=args.top_k,
            interim_dir=args.interim_dir,
            recbole_dir=args.recbole_dir,
            results_dir=args.results_dir,
            model_dir=args.model_dir,
            score_split=args.score_split,
            user_batch_size=args.user_batch_size,
            max_users=args.max_users,
            train_overrides=train_overrides or None,
            force_reformat=args.force_reformat,
            skip_existing=not args.force_train,
        )

        log.info("Results for %s: %s", cat, eval_results)

    log.info("Candidate generation complete. Run 02_rerank.py next.")


if __name__ == "__main__":
    main()
