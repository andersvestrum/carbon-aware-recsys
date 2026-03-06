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
