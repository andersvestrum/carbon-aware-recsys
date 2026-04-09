"""
RecBole-based recommender module — Candidate Generation (Pipeline Step 1)

Trains collaborative filtering models (BPR, NeuMF, SASRec, LightGCN)
and produces top-K relevance scores per user for downstream processing.

Pipeline:
    1. **This module** → top-K candidates with relevance scores
    2. DeepFM → engagement prediction on candidates
    3. Carbon-aware re-ranking: score = (1−λ)·engagement − λ·carbon
    4. Evaluation → engagement vs carbon footprint trade-off

Public API:
    from src.recommender.trainer import train_and_score
    from src.recommender.recbole_formatter import format_category_for_recbole
    from src.recommender.bpr_fallback import train_bpr_numpy
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
            # scipy <1.11 behaviour: bulk-insert a dict of {(i,j): value}.
            for key, value in data.items():
                self[key] = value

        _dok_matrix._update = _dok_update
except Exception:  # pragma: no cover - defensive
    pass
