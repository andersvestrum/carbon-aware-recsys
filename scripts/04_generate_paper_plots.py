#!/usr/bin/env python
"""
Generate the paper figures referenced in docs/main.tex.
"""

from __future__ import annotations

import argparse
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

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "output" / "results"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
DEFAULT_CARBON_RESULTS_DIR = PROJECT_ROOT / "output" / "pcf"
DEFAULT_CARBON_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "carbon"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper plots for docs/main.tex")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing reranking metrics JSON files",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where paper figures are written",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for paper summary CSV/manifest outputs",
    )
    parser.add_argument(
        "--carbon-metrics-path",
        type=Path,
        default=DEFAULT_CARBON_RESULTS_DIR / "pcf_evaluation_metrics.csv",
        help="CSV path for PCF evaluation metrics",
    )
    parser.add_argument(
        "--carbon-eval-predictions-path",
        type=Path,
        default=DEFAULT_CARBON_RESULTS_DIR / "pcf_evaluation_predictions.csv",
        help="CSV path for row-level PCF evaluation predictions",
    )
    parser.add_argument(
        "--amazon-predictions-path",
        type=Path,
        default=DEFAULT_CARBON_CACHE_DIR / "amazon_pcf_predictions.csv",
        help="CSV path for Amazon PCF predictions",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["electronics", "home_and_kitchen", "sports_and_outdoors"],
        help="Categories to include",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["BPR", "LightGCN", "NeuMF"],
        help="Models to include",
    )
    args = parser.parse_args()

    from src.evaluation.paper_plots import generate_all_paper_plots

    generated = generate_all_paper_plots(
        results_dir=args.results_dir,
        figure_dir=args.figure_dir,
        carbon_metrics_path=args.carbon_metrics_path,
        carbon_eval_predictions_path=args.carbon_eval_predictions_path,
        amazon_predictions_path=args.amazon_predictions_path,
        categories=args.categories,
        models=args.models,
        summary_output_dir=args.summary_output_dir,
    )

    log.info("Generated %d figure outputs", len(generated["figures"]))
    log.info("Manifest → %s", generated["manifest"])


if __name__ == "__main__":
    main()
