#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python"

echo "=== Step 1: Build interim splits (filter to few-shot predicted products) ==="
$PYTHON scripts/update_interim_pcf.py

echo ""
echo "=== Step 2: Train, rerank, evaluate ==="
for cat in electronics home_and_kitchen sports_and_outdoors; do
  for model in BPR NeuMF LightGCN; do
    echo "--- $cat / $model ---"
    $PYTHON scripts/01_train_recommender.py --category $cat --model $model
    $PYTHON scripts/02_rerank.py           --category $cat --model $model
    $PYTHON scripts/03_evaluate.py         --category $cat --model $model
  done
done

echo ""
echo "=== Step 3: Generate paper plots ==="
$PYTHON scripts/04_generate_paper_plots.py \
  --results-dir output/results \
  --figure-dir output/figures \
  --summary-output-dir output/results \
  --carbon-metrics-path output/pcf/pcf_evaluation_metrics.csv \
  --carbon-eval-predictions-path output/pcf/pcf_evaluation_predictions.csv \
  --amazon-predictions-path data/processed/carbon/amazon_pcf_predictions.csv

echo ""
echo "=== Done. Results in output/results/, figures in output/figures/ ==="
