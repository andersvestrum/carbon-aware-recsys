"""
Fallback BPR Trainer (NumPy only)

Minimal Bayesian Personalised Ranking implementation for environments
where RecBole or PyTorch are not available.

Implements Rendle et al. (2009) BPR-MF via SGD.  Use the RecBole trainer
(``src.recommender.trainer``) for production experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"

# ─── Defaults ────────────────────────────────────────────────────────────────
EMBED_DIM = 64
LEARNING_RATE = 0.01
REG = 0.001
N_EPOCHS = 20
BATCH_SIZE = 1024
TOP_K_CANDIDATES = 100


def train_bpr_numpy(
    interactions: pd.DataFrame,
    embed_dim: int = EMBED_DIM,
    lr: float = LEARNING_RATE,
    reg: float = REG,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    top_k: int = TOP_K_CANDIDATES,
    seed: int = 42,
) -> pd.DataFrame:
    """Train BPR-MF with pure NumPy and return top-K scores.

    The DataFrame must contain ``user_id`` and ``parent_asin`` columns
    (string IDs).  Internal integer indices are built automatically.

    Args:
        interactions: Interaction DataFrame.
        embed_dim: Latent factor dimension.
        lr: SGD learning rate.
        reg: L2 regularisation coefficient.
        n_epochs: Number of training epochs.
        batch_size: Users per mini-batch.
        top_k: Number of candidate items scored per user.
        seed: Random seed.

    Returns:
        DataFrame with columns ``[user_id, parent_asin, relevance_score]``.
    """
    log.warning("Using fallback NumPy BPR — install RecBole for full functionality")

    # Build ID mappings
    user_ids = interactions["user_id"].unique()
    item_ids = interactions["parent_asin"].unique()
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    item2idx = {iid: i for i, iid in enumerate(item_ids)}

    interactions = interactions.copy()
    interactions["user_idx"] = interactions["user_id"].map(user2idx)
    interactions["item_idx"] = interactions["parent_asin"].map(item2idx)

    n_users = len(user_ids)
    n_items = len(item_ids)

    log.info("Fallback BPR: %d users, %d items, dim=%d", n_users, n_items, embed_dim)

    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (n_users, embed_dim))
    V = rng.normal(0, 0.01, (n_items, embed_dim))

    # user → positive item set
    user_pos = interactions.groupby("user_idx")["item_idx"].apply(list).to_dict()
    all_items = np.arange(n_items)
    user_list = list(user_pos.keys())

    for epoch in range(n_epochs):
        rng.shuffle(user_list)
        total_loss = 0.0
        n_updates = 0

        for batch_start in range(0, len(user_list), batch_size):
            batch_users = user_list[batch_start : batch_start + batch_size]

            for u in batch_users:
                pos_items = user_pos[u]
                i = pos_items[rng.integers(len(pos_items))]

                neg_mask = np.ones(n_items, dtype=bool)
                neg_mask[pos_items] = False
                neg_pool = all_items[neg_mask]
                j = neg_pool[rng.integers(len(neg_pool))]

                x_uij = U[u] @ V[i] - U[u] @ V[j]
                sigmoid = 1.0 / (1.0 + np.exp(-x_uij))
                grad = 1.0 - sigmoid

                U[u] += lr * (grad * (V[i] - V[j]) - reg * U[u])
                V[i] += lr * (grad * U[u] - reg * V[i])
                V[j] += lr * (-grad * U[u] - reg * V[j])

                total_loss -= np.log(sigmoid + 1e-10)
                n_updates += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(n_updates, 1)
            log.info("  Epoch %d/%d  BPR loss: %.4f", epoch + 1, n_epochs, avg_loss)

    # Score top-K per user
    log.info("Scoring top-%d candidates per user …", top_k)
    rows: list[dict] = []
    for u in user_list:
        s = U[u] @ V.T
        top_idx = np.argpartition(s, -top_k)[-top_k:]
        for item_idx in top_idx:
            rows.append({
                "user_id": user_ids[u],
                "parent_asin": item_ids[item_idx],
                "relevance_score": float(s[item_idx]),
            })

    scores_df = pd.DataFrame(rows)
    log.info("Generated %s candidate (user, item) scores", f"{len(scores_df):,}")
    return scores_df


def evaluate_bpr(
    scores_df: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate BPR scores with NDCG@k, Recall@k, MRR (leave-one-out).

    Test item = last interaction per user (time-ordered if timestamps
    are available, otherwise positional last).

    Args:
        scores_df: DataFrame with ``[user_id, parent_asin, relevance_score]``.
        interactions: Full interaction DataFrame.
        k: Cut-off for top-k metrics.

    Returns:
        Dict with metric values.
    """
    log.info("Evaluating BPR at k=%d …", k)

    if "timestamp" in interactions.columns:
        test_items = (
            interactions.sort_values("timestamp")
            .groupby("user_id")["parent_asin"]
            .last()
            .to_dict()
        )
    else:
        test_items = (
            interactions.groupby("user_id")["parent_asin"]
            .last()
            .to_dict()
        )

    ndcg_list, recall_list, mrr_list = [], [], []

    for user_id, test_item in test_items.items():
        user_scores = scores_df[scores_df["user_id"] == user_id]
        if user_scores.empty:
            continue

        top_k_items = user_scores.nlargest(k, "relevance_score")["parent_asin"].values
        hit_positions = np.where(top_k_items == test_item)[0]

        if len(hit_positions) > 0:
            pos = hit_positions[0] + 1
            ndcg_list.append(1.0 / np.log2(pos + 1))
            recall_list.append(1.0)
            mrr_list.append(1.0 / pos)
        else:
            ndcg_list.append(0.0)
            recall_list.append(0.0)
            mrr_list.append(0.0)

    results = {
        f"NDCG@{k}": float(np.mean(ndcg_list)),
        f"Recall@{k}": float(np.mean(recall_list)),
        "MRR": float(np.mean(mrr_list)),
        "n_users_evaluated": len(ndcg_list),
    }

    log.info("  NDCG@%d:   %.4f", k, results[f"NDCG@{k}"])
    log.info("  Recall@%d: %.4f", k, results[f"Recall@{k}"])
    log.info("  MRR:        %.4f", results["MRR"])

    return results
