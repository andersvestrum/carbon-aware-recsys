"""
Carbon-aware re-ranking module — Pipeline Step 3

Re-ranks candidate items using engagement predictions and carbon footprints:
    score = (1 − λ) · engagement_norm − λ · carbon_norm

Pipeline context:
    1. RecBole → top-K candidates with relevance scores
    2. DeepFM → engagement prediction on candidates
    3. **This module** → carbon-aware re-ranked lists
    4. Evaluation → engagement vs carbon footprint trade-off

Public API:
    from src.reranking.carbon_reranker import CarbonReranker
"""
