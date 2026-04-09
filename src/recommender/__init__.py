"""
RecBole-based recommender module — Candidate Generation (Pipeline Step 1)

Trains collaborative filtering models (BPR, NeuMF, LightGCN)
and produces top-K relevance scores per user for downstream processing.

Pipeline:
    1. **This module** → top-K candidates with relevance scores
    2. Carbon-aware re-ranking using RecBole relevance scores
    3. Carbon-aware re-ranking: score = (1−λ)·engagement − λ·carbon
    4. Evaluation → engagement vs carbon footprint trade-off

Public API:
    from src.recommender.trainer import train_and_score
    from src.recommender.recbole_formatter import format_category_for_recbole
"""

# ─── scipy >=1.11 compat shim for RecBole's LightGCN ────────────────────────
# RecBole 1.1.1 calls ``scipy.sparse.dok_matrix._update`` which was removed
# when ``dok_matrix`` stopped inheriting from ``dict`` in scipy 1.11. We
# reinstate a minimal implementation so LightGCN's graph builder runs on
# modern scipy. Must execute before any RecBole import.
try:
    from scipy.sparse import dok_matrix as _dok_matrix  # noqa: E402

    if not hasattr(_dok_matrix, "_update"):
        def _dok_update(self, data):  # pragma: no cover - compat shim
            for key, value in data.items():
                self[key] = value

        _dok_matrix._update = _dok_update
except Exception:  # pragma: no cover - defensive
    pass


SUPPORTED_MODELS = ("BPR", "NeuMF", "NeuMF_pretrained", "LightGCN")


def canonical_model_name(model_name: str) -> str:
    """Return the canonical paper-supported RecBole model name."""
    aliases = {name.lower(): name for name in SUPPORTED_MODELS}
    canonical = aliases.get(model_name.lower())
    if canonical is None:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model '{model_name}'. Expected one of: {supported}."
        )
    return canonical


__all__ = ["SUPPORTED_MODELS", "canonical_model_name"]
