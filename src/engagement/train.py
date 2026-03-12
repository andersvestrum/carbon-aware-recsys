"""
DeepFM Training — Pipeline Step 2

Trains the DeepFM engagement predictor on user-item interaction data
and scores RecBole candidates.

Pipeline context:
    1. RecBole → (user_id, parent_asin, relevance_score)
    2. **This module** → train DeepFM, score candidates → engagement_score
    3. Carbon re-ranking → re-ranked lists at various λ
    4. Evaluation → engagement vs carbon footprint trade-off

Main entry point: ``train_and_score()``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.engagement.deepfm import DeepFMWrapper

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
CONFIG_DIR = PROJECT_ROOT / "configs" / "deepfm"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load DeepFM config from YAML (or use defaults)."""
    if config_path is None:
        config_path = CONFIG_DIR / "default.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        log.info("Loaded DeepFM config from %s", config_path)
        return cfg

    # Fallback defaults
    return {
        "dnn_hidden_units": [256, 128, 64],
        "l2_reg_embedding": 1e-5,
        "l2_reg_dnn": 0.0,
        "dnn_dropout": 0.1,
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "rating_threshold": 4.0,
        "device": "cpu",
    }


def _load_interim(category: str, split: str) -> pd.DataFrame:
    """Load an interim CSV for a given category and split."""
    path = INTERIM_DIR / split / f"{category}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Interim data not found: {path}")
    return pd.read_csv(path)


def train_and_score(
    category: str,
    model_name: str = "BPR",
    config_path: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train DeepFM and score RecBole candidates in one step.

    Args:
        category: Dataset category (e.g. ``electronics``).
        model_name: RecBole model name (used to locate the scores file).
        config_path: Path to DeepFM YAML config.
        config_overrides: Dict overrides on top of the config.

    Returns:
        (engagement_df, training_history) where engagement_df has columns
        ``[user_id, parent_asin, relevance_score, engagement_score]``.
    """
    # ── Config ────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    if config_overrides:
        cfg.update(config_overrides)

    # ── Load training and validation data ─────────────────────────────
    train_df = _load_interim(category, "train")
    val_df = _load_interim(category, "val")
    log.info(
        "Loaded interim data: train=%s, val=%s rows",
        f"{len(train_df):,}", f"{len(val_df):,}",
    )

    # ── Build and train DeepFM ────────────────────────────────────────
    wrapper = DeepFMWrapper(
        dnn_hidden_units=tuple(cfg.get("dnn_hidden_units", [256, 128, 64])),
        l2_reg_embedding=cfg.get("l2_reg_embedding", 1e-5),
        l2_reg_dnn=cfg.get("l2_reg_dnn", 0.0),
        dnn_dropout=cfg.get("dnn_dropout", 0.1),
        epochs=cfg.get("epochs", 10),
        batch_size=cfg.get("batch_size", 256),
        learning_rate=cfg.get("learning_rate", 1e-3),
        rating_threshold=cfg.get("rating_threshold", 4.0),
        device=cfg.get("device", "cpu"),
    )

    history = wrapper.fit(train_df, val_df)

    # ── Save model ────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"deepfm_{category}_{model_name}.joblib"
    joblib.dump(
        {
            "model": wrapper.model_,
            "label_encoders": wrapper.label_encoders_,
            "scaler": wrapper.scaler_,
            "config": cfg,
        },
        model_path,
    )
    log.info("Saved DeepFM model → %s", model_path)

    # Save training history
    history_path = MODELS_DIR / f"deepfm_{category}_{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Saved training history → %s", history_path)

    # ── Load RecBole candidate scores ─────────────────────────────────
    scores_path = RESULTS_DIR / f"{category}_{model_name}_scores.parquet"
    if not scores_path.exists():
        scores_path = RESULTS_DIR / f"{category}_{model_name}_fallback_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"RecBole scores not found. Run 01_train_recommender.py first."
        )

    candidates_df = pd.read_parquet(scores_path)
    log.info(
        "Loaded %s RecBole candidates from %s",
        f"{len(candidates_df):,}", scores_path.name,
    )

    # ── Score candidates with DeepFM ──────────────────────────────────
    # Use all interim data for item features (union of splits)
    all_interactions = pd.concat(
        [train_df, val_df, _load_interim(category, "test")],
        ignore_index=True,
    )

    engagement_df = wrapper.score_candidates(candidates_df, all_interactions)
    log.info(
        "Engagement scores: min=%.4f, median=%.4f, max=%.4f",
        engagement_df["engagement_score"].min(),
        engagement_df["engagement_score"].median(),
        engagement_df["engagement_score"].max(),
    )

    # ── Save engagement scores ────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{category}_{model_name}_engagement.parquet"
    engagement_df.to_parquet(output_path, index=False)
    log.info("Saved engagement scores → %s", output_path)

    return engagement_df, history
