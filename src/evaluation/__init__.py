"""
Evaluation and metrics module — Pipeline Step 4.

Analyses the engagement vs carbon footprint trade-off across λ values,
computes Pareto-optimal operating points, and generates visualisations.

Pipeline context:
    1. RecBole → relevance scores
    2. Carbon-aware re-ranking using RecBole relevance scores
    3. Carbon re-ranking → re-ranked lists + per-λ metrics
    4. **This module** → Pareto frontier, summary tables, plots
"""
