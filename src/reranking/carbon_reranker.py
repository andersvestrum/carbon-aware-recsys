"""
Carbon-Aware Re-ranker

Re-ranks candidate items from the base recommender using:
    score = relevance - λ * carbon_footprint

Where λ controls the trade-off between relevance and carbon awareness.
"""
