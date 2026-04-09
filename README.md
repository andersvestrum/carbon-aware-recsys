# Carbon-Aware Recommender System

This repository implements the methodology described in [docs/main.tex](docs/main.tex):

1. Estimate Product Carbon Footprint (PCF) for Amazon products with a retrieval-first pipeline over the Carbon Catalogue, with optional zero-shot and few-shot LLM baselines.
2. Train three RecBole candidate generators: `BPR`, `NeuMF`, and `LightGCN`.
3. Re-rank each user's candidate set with `score = (1 - lambda) * engagement_norm - lambda * carbon_norm`.
4. Sweep the 16-value `lambda` grid from the paper and evaluate the engagement-carbon Pareto trade-off across `electronics`, `home_and_kitchen`, and `sports_and_outdoors`.

The maintained recommendation pipeline is intentionally limited to the models used in the paper. Exploratory alternative recommenders are not part of the current workflow.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Current Pipeline

## Colab GPU Workflow

For shared Google Drive runs across multiple Colab GPU sessions, use [run/colab_full_experiment.ipynb](run/colab_full_experiment.ipynb) for Session 1 and [run/colab_worker_session.ipynb](run/colab_worker_session.ipynb) for exactly one extra worker session.

Recommended pattern:

1. Open `run/colab_full_experiment.ipynb` and leave `MODE = "auto"` for the primary session.
2. If you want one more GPU worker, open `run/colab_worker_session.ipynb` in a fresh Colab session and run all cells there.
3. Use the final status cell in either notebook to inspect shared run state in `run/`.

The notebook is intentionally thin. Most Colab-specific logic now lives in `scripts/06_colab_session.py`, which handles repo sync, dependency verification, GPU batch-size defaults, shared-run failure context, and dispatch into `scripts/05_run_full_experiment.py`.

The worker mode uses atomic job claiming in `scripts/05_run_full_experiment.py`, so multiple Colab sessions can share the same `run/` directory without duplicating `(category, model)` jobs.

### 1. Predict product carbon footprints

Run the retrieval-only PCF pipeline:

```bash
./.venv/bin/python scripts/predict_carbon.py --device cpu --num-threads 8 --skip-llm
```

Run the 100-example LLM evaluation slice used in the paper draft:

```bash
OPENAI_API_KEY=... ./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --evaluation-limit 100 \
  --amazon-limit 0
```

Outputs:

- `data/processed/carbon/amazon_pcf_predictions.csv`
- `output/results/carbon/pcf_evaluation_predictions.csv`
- `output/results/carbon/pcf_evaluation_metrics.csv`
- `output/results/carbon/pcf_run_metadata.json`

### 2. Train candidate generators

Train one of the paper baselines:

```bash
./.venv/bin/python scripts/01_train_recommender.py --category electronics --model BPR
./.venv/bin/python scripts/01_train_recommender.py --category electronics --model NeuMF
./.venv/bin/python scripts/01_train_recommender.py --category electronics --model LightGCN
```

Outputs:

- `output/results/<category>_<model>_scores.parquet`
- `output/results/<category>_<model>_eval.json`

### 3. Run carbon-aware re-ranking

```bash
./.venv/bin/python scripts/02_rerank.py --category electronics --model BPR
```

The default sweep is defined in [configs/reranking/default.yaml](configs/reranking/default.yaml) and matches the paper: 16 `lambda` values, top-10 re-ranked lists, and a top-100 candidate pool from RecBole.

Outputs:

- `output/results/<category>_<model>_reranked_<lambda>.parquet`
- `output/results/<category>_<model>_reranking_metrics.json`

### 4. Evaluate Pareto trade-offs

```bash
./.venv/bin/python scripts/03_evaluate.py --category electronics --model BPR
```

Outputs:

- `output/results/<category>_<model>_evaluation_summary.csv`
- `output/results/<category>_<model>_pareto.json`
- `output/figures/<category>_<model>_tradeoff.png`
- `output/figures/<category>_<model>_lambda_sensitivity.png`
- `output/figures/all_categories_<model>_tradeoff.png`

## Key Files

- [docs/main.tex](docs/main.tex)
- [scripts/predict_carbon.py](scripts/predict_carbon.py)
- [scripts/01_train_recommender.py](scripts/01_train_recommender.py)
- [scripts/02_rerank.py](scripts/02_rerank.py)
- [scripts/03_evaluate.py](scripts/03_evaluate.py)
- [scripts/05_run_full_experiment.py](scripts/05_run_full_experiment.py)
- [scripts/06_colab_session.py](scripts/06_colab_session.py)
- [src/carbon/retrieval.py](src/carbon/retrieval.py)
- [src/recommender/trainer.py](src/recommender/trainer.py)
- [src/reranking/carbon_reranker.py](src/reranking/carbon_reranker.py)
- [src/evaluation/metrics.py](src/evaluation/metrics.py)
