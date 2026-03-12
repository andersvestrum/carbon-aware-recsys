"""
DeepFM engagement prediction module — Pipeline Step 2.

Takes RecBole candidate scores (step 1) and interim interaction features,
trains a DeepFM model to predict engagement probability, and outputs
``engagement_score`` per (user, item) pair for the carbon-aware
re-ranker (step 3).

Pipeline context:
    1. RecBole → (user_id, parent_asin, relevance_score)
    2. **This module** → (user_id, parent_asin, engagement_score)
    3. Carbon re-ranking → re-ranked lists at various λ
    4. Evaluation → engagement vs carbon footprint trade-off
"""
