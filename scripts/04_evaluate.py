#!/usr/bin/env python
"""
Evaluation & Pareto Frontier (Pipeline Step 4)
================================================
Analyses the engagement vs AvgPCF@10 trade-off, extracts
Pareto-optimal operating points, and generates visualisations.

Pipeline context:
    1. RecBole → top-K candidates with relevance scores
    2. DeepFM → engagement prediction on candidates
    3. Carbon re-ranking → re-ranked lists + metrics per λ
    4. **This script** → Pareto frontier, summary tables, plots

Inputs:
    output/results/<category>_<model>_reranking_metrics.json  (from 03_rerank.py)

Outputs:
    output/results/<category>_<model>_evaluation_summary.csv  — full summary table
    output/results/<category>_<model>_pareto.json             — Pareto-optimal points
    output/figures/<category>_<model>_tradeoff.png            — trade-off scatter
    output/figures/<category>_<model>_lambda_sensitivity.png  — dual-axis λ plot
    output/figures/all_categories_<model>_tradeoff.png        — cross-category comparison

Usage:
    python scripts/04_evaluate.py                            # all categories
    python scripts/04_evaluate.py --category electronics
    python scripts/04_evaluate.py --model LightGCN
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation & Pareto frontier analysis (pipeline step 4)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category to evaluate (default: all three)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BPR",
        help="Model name matching the metrics file (default: BPR)",
    )
    args = parser.parse_args()

    from src.evaluation.metrics import (
        evaluate_category,
        plot_multi_category,
    )

    categories = [args.category] if args.category else ALL_CATEGORIES

    # Collect metrics across categories for multi-category plot
    all_cat_metrics: dict[str, list[dict]] = {}

    for cat in categories:
        log.info("=" * 60)
        log.info("Evaluating: %s (model=%s)", cat, args.model)
        log.info("=" * 60)

        try:
            result = evaluate_category(cat, args.model)
        except FileNotFoundError as e:
            log.error("  Skipping %s: %s", cat, e)
            continue

        # Print summary
        summary = result["summary_table"]
        pareto = result["pareto_points"]
        baseline = result["baseline"]
        best = result["best_tradeoff"]

        log.info("")
        log.info("  ── Summary ──")
        log.info("  Baseline (λ=0): NDCG@10=%.4f  AvgPCF@10=%.2f kg",
                 baseline.get("NDCG@10", 0), baseline.get("avg_carbon_kg", 0))
        log.info("  Pareto-optimal points: %d", len(pareto))

        for pt in pareto:
            log.info(
                "    λ=%.3f  NDCG@10=%.4f  AvgPCF@10=%.2f kg  (−%.1f%%)",
                pt["lambda"],
                pt.get("NDCG@10", 0),
                pt.get("avg_carbon_kg", 0),
                100 * (baseline.get("avg_carbon_kg", 1) - pt.get("avg_carbon_kg", 0))
                / max(baseline.get("avg_carbon_kg", 1), 1e-9),
            )

        if best:
            log.info("")
            log.info(
                "  ★ Recommended operating point: λ=%.3f",
                best["lambda"],
            )
            log.info(
                "    NDCG@10=%.4f  AvgPCF@10=%.2f kg",
                best.get("NDCG@10", 0),
                best.get("avg_carbon_kg", 0),
            )

        # Load per_lambda for multi-category plot
        metrics_path = RESULTS_DIR / f"{cat}_{args.model}_reranking_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            all_cat_metrics[cat] = data["per_lambda"]

    # Multi-category plot if we have more than one
    if len(all_cat_metrics) > 1:
        log.info("=" * 60)
        log.info("Generating cross-category comparison plot")
        log.info("=" * 60)
        plot_multi_category(all_cat_metrics, args.model)

    log.info("")
    log.info("Evaluation complete. Check output/figures/ and output/results/.")


if __name__ == "__main__":
    main()
