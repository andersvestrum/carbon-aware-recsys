"""
Carbon-Aware Re-ranker — Pipeline Step 3

Re-ranks candidate items using DeepFM engagement predictions and
carbon footprints:

    s(u, i; λ) = (1 − λ) · ẽ(u, i) − λ · c̃(i)

Where:
    ẽ      = per-user normalised engagement score from DeepFM  (∈ [0, 1])
    c̃      = globally normalised carbon footprint              (∈ [0, 1])
    λ      = trade-off parameter  (0 = pure engagement, 1 = pure carbon)

Pipeline context:
    1. RecBole → (user_id, parent_asin, relevance_score)
    2. DeepFM → (user_id, parent_asin, engagement_score)
    3. **This module** → re-ranked lists at one or more λ values
    4. Evaluation → engagement vs carbon footprint trade-off
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"


# ─── Normalisation helpers ───────────────────────────────────────────────────

def normalise_engagement_per_user(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise engagement scores to [0, 1] **per user**.

    Raw engagement scores vary in scale across users; per-user
    normalisation ensures λ has a consistent interpretation.

    Args:
        scores_df: Must contain ``user_id`` and ``engagement_score``.

    Returns:
        Copy with added ``engagement_norm`` column.
    """
    df = scores_df.copy()

    def _norm(x: pd.Series) -> pd.Series:
        mn, mx = x.min(), x.max()
        if mx == mn:
            return pd.Series(0.5, index=x.index)
        return (x - mn) / (mx - mn)

    df["engagement_norm"] = (
        df.groupby("user_id")["engagement_score"].transform(_norm)
    )
    return df


def normalise_carbon_global(carbon_series: pd.Series) -> pd.Series:
    """Min-max normalise carbon footprints to [0, 1] globally.

    Carbon footprint is an item-level property (not user-specific),
    so normalisation is across all items.
    """
    mn, mx = carbon_series.min(), carbon_series.max()
    if mx == mn:
        return pd.Series(0.5, index=carbon_series.index)
    return (carbon_series - mn) / (mx - mn)


# ─── Core re-ranker ─────────────────────────────────────────────────────────

@dataclass
class CarbonReranker:
    """Carbon-aware re-ranker.

    Usage::

        reranker = CarbonReranker(top_k=10)
        result = reranker.rerank(scores_df, carbon_df, lam=0.3)

    Attributes:
        top_k: Number of items to return per user after re-ranking.
    """

    top_k: int = 10

    def rerank(
        self,
        scores_df: pd.DataFrame,
        carbon_df: pd.DataFrame,
        lam: float,
    ) -> pd.DataFrame:
        """Re-rank candidates for a single λ value.

        Args:
            scores_df: Engagement predictions from DeepFM.  Must have
                columns ``[user_id, parent_asin, engagement_score]``.
            carbon_df: Item carbon footprints.  Must have columns
                ``[parent_asin, pcf]``.
            lam: Trade-off parameter in [0, 1].

        Returns:
            DataFrame with columns:
                ``[user_id, parent_asin, engagement_norm, carbon_norm,
                  combined_score, rank]``
            sorted by ``rank`` within each user.
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"λ must be in [0, 1], got {lam}")

        # 1. Normalise engagement per user
        df = normalise_engagement_per_user(scores_df)

        # 2. Join carbon footprints and normalise globally
        carbon_slim = carbon_df[["parent_asin", "pcf"]].drop_duplicates(
            subset="parent_asin"
        )
        df = df.merge(carbon_slim, on="parent_asin", how="left")
        df["pcf"] = df["pcf"].fillna(carbon_slim["pcf"].median())
        df["carbon_norm"] = normalise_carbon_global(df["pcf"])

        # 3. Combined score:  s(u, i; λ) = (1−λ)·engagement − λ·carbon
        df["combined_score"] = (
            (1.0 - lam) * df["engagement_norm"]
            - lam * df["carbon_norm"]
        )

        # 4. Take top-k per user
        ranked = (
            df.sort_values("combined_score", ascending=False)
            .groupby("user_id")
            .head(self.top_k)
            .copy()
        )
        ranked["rank"] = (
            ranked.groupby("user_id")["combined_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        keep_cols = [
            "user_id", "parent_asin",
            "engagement_norm", "carbon_norm", "pcf",
            "combined_score", "rank",
        ]
        return ranked[keep_cols].sort_values(["user_id", "rank"])

    def sweep(
        self,
        scores_df: pd.DataFrame,
        carbon_df: pd.DataFrame,
        lambda_values: list[float] | np.ndarray,
    ) -> dict[float, pd.DataFrame]:
        """Re-rank at multiple λ values.

        Args:
            scores_df: Engagement predictions from DeepFM.
            carbon_df: Item carbon footprints.
            lambda_values: Sequence of λ values to sweep.

        Returns:
            Dict mapping λ → re-ranked DataFrame.
        """
        results: dict[float, pd.DataFrame] = {}
        for lam in lambda_values:
            results[lam] = self.rerank(scores_df, carbon_df, lam)
        return results


# ─── Metrics for a single λ ──────────────────────────────────────────────────

def compute_reranking_metrics(
    ranked_df: pd.DataFrame,
    test_interactions: pd.DataFrame,
    lam: float,
    k: int = 10,
) -> dict[str, Any]:
    """Compute ranking quality and carbon metrics for one λ.

    Args:
        ranked_df: Output of ``CarbonReranker.rerank()`` (has ``rank``).
        test_interactions: DataFrame with ``[user_id, parent_asin]``
            representing held-out test items (one per user).
        lam: The λ value used.
        k: Cut-off for top-k metrics.

    Returns:
        Dict with keys: ``lambda``, ``avg_carbon_kg``, ``NDCG@k``,
        ``Recall@k``, ``MRR``, ``n_users``.
    """
    # Build test set lookup: user → test item
    test_lookup = test_interactions.set_index("user_id")["parent_asin"].to_dict()

    avg_carbon = ranked_df["pcf"].mean()

    ndcg_vals, recall_vals, mrr_vals = [], [], []

    for user_id, test_item in test_lookup.items():
        user_recs = ranked_df[ranked_df["user_id"] == user_id]
        if user_recs.empty:
            continue

        rec_items = user_recs.sort_values("rank")["parent_asin"].values[:k]
        hit = np.where(rec_items == test_item)[0]

        if len(hit) > 0:
            pos = hit[0] + 1
            ndcg_vals.append(1.0 / np.log2(pos + 1))
            recall_vals.append(1.0)
            mrr_vals.append(1.0 / pos)
        else:
            ndcg_vals.append(0.0)
            recall_vals.append(0.0)
            mrr_vals.append(0.0)

    return {
        "lambda": float(lam),
        "avg_carbon_kg": float(avg_carbon),
        f"NDCG@{k}": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        f"Recall@{k}": float(np.mean(recall_vals)) if recall_vals else 0.0,
        "MRR": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
        "n_users": int(ranked_df["user_id"].nunique()),
    }


def build_test_set(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build a leave-one-out test set (last interaction per user).

    If ``timestamp`` is present, uses the most recent interaction;
    otherwise uses positional last.

    Args:
        interactions: Full interaction DataFrame with ``user_id``,
            ``parent_asin``, and optionally ``timestamp``.

    Returns:
        DataFrame with one row per user: ``[user_id, parent_asin]``.
    """
    if "timestamp" in interactions.columns:
        test = (
            interactions.sort_values("timestamp")
            .groupby("user_id")
            .last()
            .reset_index()[["user_id", "parent_asin"]]
        )
    else:
        test = (
            interactions.groupby("user_id")
            .last()
            .reset_index()[["user_id", "parent_asin"]]
        )
    return test

