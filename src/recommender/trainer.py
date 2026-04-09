"""
RecBole Trainer — Candidate Generation (Pipeline Step 1)

Trains collaborative filtering models using RecBole and produces
relevance scores for downstream engagement prediction and re-ranking.

Supported candidate generators: BPR, NeuMF, LightGCN.

Pipeline context:
  1. **This module** → top-K candidate items per user with relevance scores
  2. Carbon-aware re-ranking using RecBole relevance scores
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
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import SUPPORTED_MODELS, canonical_model_name

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
BENCHMARK_SPLIT_SUFFIXES = ("train", "valid", "test")

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
    "eval_step": 5,

    # Evaluation — leave-one-out
    "eval_args": {
        "split": {"TS": [0.8, 0.1, 0.1]},
        "group_by": "user",
        "order": "TO",
        "mode": "full",
    },
    "metrics": ["NDCG", "Recall", "MRR"],
    "topk": [10],
    "valid_metric": "NDCG@10",
    "eval_batch_size": 4096,

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
            "RecBole is not installed. Run:\n"
            "    pip install recbole torch\n"
            f"This pipeline supports {', '.join(SUPPORTED_MODELS)}."
        )


def _has_benchmark_splits(dataset_name: str, data_path: str | Path) -> bool:
    dataset_dir = Path(data_path) / dataset_name
    return all(
        (dataset_dir / f"{dataset_name}.{suffix}.inter").exists()
        for suffix in BENCHMARK_SPLIT_SUFFIXES
    )


def build_config(
    dataset_name: str,
    model_name: str = "BPR",
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    data_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    benchmark_splits: bool | None = None,
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
    model_name = canonical_model_name(model_name)
    _ensure_recbole()
    from recbole.config import Config

    # Start from defaults and apply overrides
    params = {**DEFAULT_CONFIG}
    params["model"] = model_name
    params["data_path"] = str(data_path or RECBOLE_DIR)
    params["dataset"] = dataset_name
    params["checkpoint_dir"] = str(checkpoint_dir or (MODEL_DIR / "recbole_checkpoints"))
    params["TIME_FIELD"] = "timestamp"

    if benchmark_splits is None:
        benchmark_splits = _has_benchmark_splits(dataset_name, params["data_path"])
    if benchmark_splits:
        params["benchmark_filename"] = list(BENCHMARK_SPLIT_SUFFIXES)

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
    data_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    benchmark_splits: bool | None = None,
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
    model_name = canonical_model_name(model_name)
    _ensure_recbole()
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer

    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    config = build_config(
        dataset_name,
        model_name,
        config_file,
        overrides,
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        benchmark_splits=benchmark_splits,
    )

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

    # Load the best checkpoint explicitly so both evaluation and downstream
    # score extraction use the same model state under PyTorch ≥ 2.6.
    best_checkpoint_loaded = False
    checkpoint_file = getattr(trainer, "saved_model_file", None)
    if checkpoint_file:
        try:
            import torch

            try:
                checkpoint = torch.load(
                    checkpoint_file,
                    map_location=config["device"],
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    checkpoint_file,
                    map_location=config["device"],
                )
            trainer.model.load_state_dict(checkpoint["state_dict"])
            trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
            best_checkpoint_loaded = True
            log.info("Reloaded best checkpoint → %s", checkpoint_file)
        except Exception as exc:
            log.warning(
                "Could not reload best checkpoint (%r); "
                "evaluating and scoring with in-memory model instead.",
                exc,
            )

    # Evaluate on test set using whichever model state is currently loaded.
    test_result = trainer.evaluate(test_data, load_best_model=False)
    if not best_checkpoint_loaded:
        log.warning("Best checkpoint reload was skipped; test metrics use in-memory model state.")

    log.info("Test results: %s", test_result)

    # Persist evaluation results
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{dataset_name}_{model_name}_eval.json"
    with open(results_path, "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in test_result.items()},
            f,
            indent=2,
        )
    log.info("Saved eval results → %s", results_path)

    return model, train_data.dataset, config, test_result


# ─── Score extraction (model-agnostic) ────────────────────────────────────────

def _map_external_user_ids(
    dataset,
    external_user_ids: Sequence[str] | None,
) -> np.ndarray:
    if external_user_ids is None:
        return np.arange(1, dataset.user_num, dtype=np.int64)

    mapped_ids: list[int] = []
    for user_id in external_user_ids:
        try:
            mapped_ids.append(int(dataset.token2id("user_id", str(user_id))))
        except ValueError:
            continue
    return np.asarray(mapped_ids, dtype=np.int64)


def _map_seen_items(
    dataset,
    seen_items_by_user: dict[str, Sequence[str]] | None,
) -> dict[int, np.ndarray]:
    if not seen_items_by_user:
        return {}

    mapped: dict[int, np.ndarray] = {}
    for user_id, item_ids in seen_items_by_user.items():
        try:
            internal_user_id = int(dataset.token2id("user_id", str(user_id)))
        except ValueError:
            continue

        internal_items: list[int] = []
        for item_id in item_ids:
            try:
                internal_items.append(int(dataset.token2id("item_id", str(item_id))))
            except ValueError:
                continue
        if internal_items:
            mapped[internal_user_id] = np.unique(np.asarray(internal_items, dtype=np.int64))
    return mapped


def _mask_seen_items(scores, batch_user_ids, seen_item_ids_by_user):
    if not seen_item_ids_by_user:
        return
    for row_idx, internal_user_id in enumerate(batch_user_ids):
        seen = seen_item_ids_by_user.get(int(internal_user_id))
        if seen is not None and len(seen):
            scores[row_idx, seen] = -float("inf")


def _score_via_full_sort(
    model,
    dataset,
    config,
    top_k,
    target_user_ids: np.ndarray,
    seen_item_ids_by_user: dict[int, np.ndarray] | None = None,
    user_batch_size: int = 256,
):
    """Score users via full_sort_predict (general recommenders)."""
    import torch
    from recbole.data.interaction import Interaction
    from tqdm import tqdm

    user_ids = dataset.field2id_token["user_id"]
    item_ids = dataset.field2id_token["item_id"]
    n_items = dataset.item_num
    device = config["device"]
    rows: list[dict] = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(target_user_ids), user_batch_size),
            desc="Scoring users",
            unit="batch",
        ):
            batch_user_ids = target_user_ids[start:start + user_batch_size]
            interaction = Interaction(
                {"user_id": torch.tensor(batch_user_ids, dtype=torch.int64)}
            )
            interaction = interaction.to(device)

            scores = model.full_sort_predict(interaction).view(-1, n_items).cpu()
            scores[:, 0] = -float("inf")
            _mask_seen_items(scores, batch_user_ids, seen_item_ids_by_user)

            topk_scores, topk_items = torch.topk(
                scores,
                min(top_k, n_items - 1),
                dim=1,
            )

            for row_idx, uid in enumerate(batch_user_ids):
                u_str = user_ids[int(uid)] if int(uid) < len(user_ids) else None
                if u_str is None:
                    continue
                for item_idx, score in zip(
                    topk_items[row_idx].numpy(),
                    topk_scores[row_idx].numpy(),
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
    return rows


def _score_via_pairwise(
    model,
    dataset,
    config,
    top_k,
    target_user_ids: np.ndarray,
    seen_item_ids_by_user: dict[int, np.ndarray] | None = None,
):
    """Score users via predict() on (user, item) pairs — universal fallback."""
    import torch
    from recbole.data.interaction import Interaction
    from tqdm import tqdm

    user_ids = dataset.field2id_token["user_id"]
    item_ids = dataset.field2id_token["item_id"]
    n_items = dataset.item_num
    device = config["device"]
    rows: list[dict] = []

    all_item_ids = torch.arange(1, n_items, device=device)

    with torch.no_grad():
        for uid in tqdm(target_user_ids, desc="Scoring users (pairwise)", unit="user"):
            user_tensor = torch.full_like(all_item_ids, uid)
            interaction = Interaction({
                "user_id": user_tensor,
                "item_id": all_item_ids,
            })
            interaction = interaction.to(device)

            scores = model.predict(interaction).cpu()
            seen = seen_item_ids_by_user.get(int(uid)) if seen_item_ids_by_user else None
            if seen is not None and len(seen):
                scores[torch.as_tensor(seen - 1, dtype=torch.int64)] = -float("inf")
            topk_scores, topk_idx = torch.topk(scores, min(top_k, len(scores)))

            u_str = user_ids[uid] if uid < len(user_ids) else None
            if u_str is None:
                continue

            for idx, score in zip(topk_idx.numpy(), topk_scores.numpy()):
                item_internal = all_item_ids[idx].item()
                i_str = (
                    item_ids[item_internal]
                    if item_internal < len(item_ids)
                    else None
                )
                if i_str is None:
                    continue
                rows.append({
                    "user_id": u_str,
                    "parent_asin": i_str,
                    "relevance_score": float(score),
                })
    return rows


def extract_relevance_scores(
    model,
    dataset,
    config,
    top_k: int = TOP_K_CANDIDATES,
    external_user_ids: Sequence[str] | None = None,
    seen_items_by_user: dict[str, Sequence[str]] | None = None,
    user_batch_size: int = 256,
) -> pd.DataFrame:
    """Extract relevance scores for the top-K items per user.

    Tries ``full_sort_predict`` first (fast, works for most general
    recommenders).  Falls back to pairwise ``predict()`` if the model
    does not implement full-sort scoring.

    Args:
        model: Trained RecBole model from the supported candidate generators.
        dataset: RecBole ``Dataset`` object (for ID mapping).
        config: RecBole ``Config`` (for device info).
        top_k: Number of candidate items per user.
        external_user_ids: Optional external user IDs to score. Defaults to all users.
        seen_items_by_user: Optional map of external user ID → seen item IDs to mask.
        user_batch_size: Number of users to score per full-sort batch.

    Returns:
        DataFrame with columns ``[user_id, parent_asin, relevance_score]``.
    """
    import torch
    from recbole.data.interaction import Interaction

    target_user_ids = _map_external_user_ids(dataset, external_user_ids)
    seen_item_ids_by_user = _map_seen_items(dataset, seen_items_by_user)
    if len(target_user_ids) == 0:
        raise ValueError("No evaluation users were mapped into the RecBole dataset.")

    log.info("Extracting relevance scores (top-%d per user) …", top_k)
    log.info("Scoring %s users with user_batch_size=%d", f"{len(target_user_ids):,}", user_batch_size)
    model.eval()

    # Probe whether full_sort_predict works for this model
    try:
        probe = Interaction(
            {"user_id": torch.tensor([int(target_user_ids[0])], dtype=torch.int64)}
        ).to(config["device"])
        model.full_sort_predict(probe)
        log.info("Using full_sort_predict (fast path)")
        rows = _score_via_full_sort(
            model,
            dataset,
            config,
            top_k,
            target_user_ids,
            seen_item_ids_by_user=seen_item_ids_by_user,
            user_batch_size=user_batch_size,
        )
    except (NotImplementedError, TypeError, RuntimeError) as exc:
        log.info("full_sort_predict unavailable (%s), falling back to pairwise predict", exc)
        rows = _score_via_pairwise(
            model,
            dataset,
            config,
            top_k,
            target_user_ids,
            seen_item_ids_by_user=seen_item_ids_by_user,
        )

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
    data_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    benchmark_splits: bool | None = None,
    external_user_ids: Sequence[str] | None = None,
    seen_items_by_user: dict[str, Sequence[str]] | None = None,
    user_batch_size: int = 256,
) -> tuple[pd.DataFrame, dict]:
    """Train a model and extract relevance scores in one step.

    This is the main entry point for pipeline step 1 (candidate generation).
    The output ``relevance_score`` column is consumed by the carbon-aware
    re-ranker in step 2.

    Args:
        dataset_name: RecBole dataset name.
        model_name: Model class name (BPR, NeuMF, or LightGCN).
        config_file: Optional YAML config.
        overrides: Optional dict overrides.
        top_k: Candidates per user.
        output_path: Where to save the scores parquet. Defaults to
            ``output/results/<dataset>_<model>_scores.parquet``.
        data_path: Root directory containing RecBole datasets.
        checkpoint_dir: Directory for RecBole checkpoints.
        results_dir: Directory for eval JSON and default score parquet output.
        benchmark_splits: Use benchmark split files when available.
        external_user_ids: Optional evaluation users to score.
        seen_items_by_user: Optional user → seen item mapping to mask from ranking.
        user_batch_size: Number of users per full-sort batch.

    Returns:
        (scores_df, eval_results)  where scores_df has columns
        ``[user_id, parent_asin, relevance_score]``.
    """
    model_name = canonical_model_name(model_name)
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR

    model, dataset, config, eval_results = train(
        dataset_name,
        model_name,
        config_file,
        overrides,
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        benchmark_splits=benchmark_splits,
    )

    scores_df = extract_relevance_scores(
        model,
        dataset,
        config,
        top_k,
        external_user_ids=external_user_ids,
        seen_items_by_user=seen_items_by_user,
        user_batch_size=user_batch_size,
    )

    # Save scores
    if output_path is None:
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{dataset_name}_{model_name}_scores.parquet"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_df.to_parquet(output_path, index=False)
    log.info("Saved relevance scores → %s", output_path)

    return scores_df, eval_results
