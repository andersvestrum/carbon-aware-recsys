"""
RecBole Trainer — Candidate Generation (Pipeline Step 1)

Trains collaborative filtering models using RecBole and produces
relevance scores for downstream engagement prediction and re-ranking.

Supported models: BPR, NeuMF, SASRec, LightGCN (any RecBole model).

Pipeline context:
  1. **This module** → top-K candidate items per user with relevance scores
  2. DeepFM engagement prediction on candidates
  3. Carbon-aware re-ranking:  score = (1−λ)·engagement − λ·carbon
  4. Evaluation: engagement vs carbon footprint trade-off

Handles:
  - Config creation (merging defaults + YAML overrides)
  - Dataset creation from pre-formatted .inter files
  - Model training with early stopping
  - Evaluation on held-out test set (NDCG@10, Recall@10, MRR)
  - Top-K relevance score extraction (model-agnostic via full_sort_predict)

NumPy 2.0 compatibility shims are applied at import time so that
RecBole (which relies on deprecated np type aliases) works correctly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ─── NumPy 2.0 compat shims (must run before importing RecBole) ─────────────
_NP_COMPAT = {
    "bool8": np.bool_,
    "float_": np.float64,
    "float": np.float64,
    "int_": np.int64,
    "int": np.int64,
    "complex_": np.complex128,
    "complex": np.complex128,
    "unicode_": np.str_,
    "unicode": np.str_,
}
for _alias, _real in _NP_COMPAT.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

log = logging.getLogger(__name__)

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECBOLE_DIR = PROJECT_ROOT / "data" / "processed" / "recbole"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
CONFIG_DIR = PROJECT_ROOT / "configs" / "recbole"

# ─── Default BPR config ─────────────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    # Model
    "model": "BPR",
    "embedding_size": 64,

    # Training
    "epochs": 50,
    "train_batch_size": 2048,
    "learning_rate": 1e-3,
    "neg_sampling": {"uniform": 1},

    # Evaluation — leave-one-out
    "eval_args": {
        "split": {"LS": "valid_and_test"},
        "group_by": "user",
        "order": "RO",
        "mode": "full",
    },
    "metrics": ["NDCG", "Recall", "MRR"],
    "topk": [10],
    "valid_metric": "NDCG@10",

    # Early stopping
    "stopping_step": 10,

    # Misc
    "show_progress": True,
}

# Number of candidate items to score per user for downstream re-ranking
TOP_K_CANDIDATES = 100


def _ensure_recbole():
    """Import RecBole or raise a helpful error."""
    try:
        import recbole  # noqa: F401
    except ImportError:
        raise ImportError(
            "RecBole is not installed.  Run:\n"
            "    pip install recbole\n"
            "RecBole requires Python ≥ 3.7 and PyTorch."
        )


def build_config(
    dataset_name: str,
    model_name: str = "BPR",
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
):
    """Build a RecBole ``Config`` object.

    Priority (highest → lowest):
        1. *overrides* dict
        2. YAML *config_file*
        3. DEFAULT_CONFIG

    Args:
        dataset_name: RecBole dataset name (must match a folder under
            ``data/processed/recbole/<dataset_name>/``).
        model_name: RecBole model class name (e.g. ``"BPR"``, ``"LightGCN"``).
        config_file: Optional YAML config to merge.
        overrides: Optional dict of config key/value overrides.

    Returns:
        A RecBole ``Config`` instance.
    """
    _ensure_recbole()
    from recbole.config import Config

    # Start from defaults and apply overrides
    params = {**DEFAULT_CONFIG}
    params["model"] = model_name
    params["data_path"] = str(RECBOLE_DIR)
    params["dataset"] = dataset_name
    params["checkpoint_dir"] = str(MODEL_DIR / "recbole_checkpoints")

    if overrides:
        params.update(overrides)

    config_file_list = []
    if config_file is not None:
        config_file_list.append(str(config_file))

    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_dict=params,
        config_file_list=config_file_list if config_file_list else None,
    )
    return config


def train(
    dataset_name: str,
    model_name: str = "BPR",
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
):
    """Train a RecBole model end-to-end.

    Args:
        dataset_name: Name of dataset under ``data/processed/recbole/``.
        model_name: RecBole model class name.
        config_file: Optional YAML with extra config.
        overrides: Optional dict overrides.

    Returns:
        (model, dataset, config, test_results)
    """
    _ensure_recbole()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer

    config = build_config(dataset_name, model_name, config_file, overrides)

    log.info("Creating RecBole dataset '%s' …", dataset_name)
    dataset = create_dataset(config)

    log.info("Preparing train / val / test data …")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    log.info("Initialising model: %s", model_name)
    model_class = get_model(model_name)
    model = model_class(config, train_data.dataset).to(config["device"])

    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)

    log.info("Training %s …", model_name)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        verbose=True,
    )
    log.info("Best validation score (NDCG@10): %.4f", best_valid_score)

    # Evaluate on test set
    # PyTorch ≥ 2.6 changed torch.load defaults, so fall back gracefully
    try:
        test_result = trainer.evaluate(test_data, load_best_model=True)
    except Exception as exc:
        log.warning(
            "Could not reload best checkpoint (%r); "
            "evaluating in-memory model instead.",
            exc,
        )
        test_result = trainer.evaluate(test_data, load_best_model=False)

    log.info("Test results: %s", test_result)

    # Persist evaluation results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{dataset_name}_{model_name}_eval.json"
    with open(results_path, "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in test_result.items()},
            f,
            indent=2,
        )
    log.info("Saved eval results → %s", results_path)

    return model, dataset, config, test_result


# ─── Score extraction (model-agnostic) ────────────────────────────────────────

def extract_relevance_scores(
    model,
    dataset,
    config,
    top_k: int = TOP_K_CANDIDATES,
) -> pd.DataFrame:
    """Extract relevance scores for the top-K items per user.

    Works with **any** RecBole general/sequential recommender by using
    ``model.full_sort_predict()``, which computes scores for all items
    in a single forward pass regardless of model architecture.

    The output feeds directly into the carbon-aware re-ranker:
        ``re_rank_score = relevance_score − λ · carbon_footprint``

    Args:
        model: Trained RecBole model (any general or sequential recommender).
        dataset: RecBole ``Dataset`` object (for ID mapping).
        config: RecBole ``Config`` (for device info).
        top_k: Number of candidate items per user.

    Returns:
        DataFrame with columns ``[user_id, parent_asin, relevance_score]``.
    """
    import torch
    from recbole.data.interaction import Interaction

    log.info("Extracting relevance scores (top-%d per user) …", top_k)

    model.eval()
    rows: list[dict] = []

    user_ids = dataset.field2id_token["user_id"]   # idx → original str
    item_ids = dataset.field2id_token["item_id"]   # idx → original str
    n_users = dataset.user_num
    n_items = dataset.item_num
    device = config["device"]

    with torch.no_grad():
        for uid in range(1, n_users):  # 0 is [PAD] in RecBole
            # Build a dummy interaction with just the user id
            interaction = Interaction({"user_id": torch.tensor([uid])})
            interaction = interaction.to(device)

            # full_sort_predict returns scores for ALL items  (shape: n_items,)
            scores = model.full_sort_predict(interaction).cpu()

            # Mask out padding item (index 0)
            scores[0] = -float("inf")

            topk_scores, topk_items = torch.topk(scores, min(top_k, n_items - 1))

            u_str = user_ids[uid] if uid < len(user_ids) else None
            if u_str is None:
                continue

            for item_idx, score in zip(
                topk_items.numpy(), topk_scores.numpy()
            ):
                i_str = (
                    item_ids[int(item_idx)]
                    if int(item_idx) < len(item_ids)
                    else None
                )
                if i_str is None:
                    continue
                rows.append({
                    "user_id": u_str,
                    "parent_asin": i_str,
                    "relevance_score": float(score),
                })

    scores_df = pd.DataFrame(rows)
    log.info("Extracted %s (user, item, score) tuples", f"{len(scores_df):,}")
    return scores_df


# ─── Convenience: train + score in one call ───────────────────────────────────

def train_and_score(
    dataset_name: str,
    model_name: str = "BPR",
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    top_k: int = TOP_K_CANDIDATES,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train a model and extract relevance scores in one step.

    This is the main entry point for pipeline step 1 (candidate generation).
    The output ``relevance_score`` column is consumed by the carbon-aware
    re-ranker in step 2.

    Args:
        dataset_name: RecBole dataset name.
        model_name: Model class name (BPR, NeuMF, SASRec, LightGCN, …).
        config_file: Optional YAML config.
        overrides: Optional dict overrides.
        top_k: Candidates per user.
        output_path: Where to save the scores parquet. Defaults to
            ``output/results/<dataset>_<model>_scores.parquet``.

    Returns:
        (scores_df, eval_results)  where scores_df has columns
        ``[user_id, parent_asin, relevance_score]``.
    """
    model, dataset, config, eval_results = train(
        dataset_name, model_name, config_file, overrides,
    )

    scores_df = extract_relevance_scores(model, dataset, config, top_k)

    # Save scores
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"{dataset_name}_{model_name}_scores.parquet"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_df.to_parquet(output_path, index=False)
    log.info("Saved relevance scores → %s", output_path)

    return scores_df, eval_results
