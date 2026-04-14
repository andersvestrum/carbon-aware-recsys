# Carbon-Aware Recommender System

This repository implements the methodology described in [docs/main.tex](docs/main.tex):

1. Estimate Product Carbon Footprint (PCF) for Amazon products with a retrieval-first pipeline over the Carbon Catalogue, with optional zero-shot and few-shot LLM baselines.
2. Train three RecBole candidate generators: `BPR`, `NeuMF`, and `LightGCN`.
3. Re-rank each user's candidate set with `score = (1 - lambda) * engagement_norm - lambda * carbon_norm`.
4. Sweep the default `lambda` grid, including a denser tail from `0.90` to `1.00`, and evaluate the engagement-carbon Pareto trade-off across `electronics`, `home_and_kitchen`, and `sports_and_outdoors`.

The maintained recommendation pipeline is intentionally limited to the models used in the paper. Exploratory alternative recommenders are not part of the current workflow.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Workflow

This repository keeps the methodology as explicit stepwise scripts.

### Tracked vs ignored outputs

`.gitignore` excludes large or regenerable paths: `data/raw/`, `data/interim/`, `data/processed/`, `output/results/`, and `output/models/`. **`output/pcf/`** and **`output/figures/`** are not ignored, so PCF evaluation CSVs and figure PNGs can live in the repo and travel with a clone. The Amazon prediction cache at `data/processed/carbon/amazon_pcf_predictions.csv` is still ignored, so a paper-faithful rerun must regenerate or restore that file before preprocessing.

For a full end-to-end checklist (including optional paths for `predict_carbon.py` and PCF post-processing), see [docs/FINAL_RUN.md](docs/FINAL_RUN.md).

## Colab GPU Workflow

For the fastest Colab setup, use:

- [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb)
- [notebooks/colab_watch.ipynb](notebooks/colab_watch.ipynb)
- [scripts/05_colab_runner.py](scripts/05_colab_runner.py)
- [scripts/06_colab_watch.py](scripts/06_colab_watch.py)

Recommended pattern:

1. Keep the repo checkout on Google Drive.
2. Use a separate Drive workspace root for shared state, results, figures, logs, and checkpoints.
3. Run one Colab GPU session with `MODE = "primary"`.
4. Run one additional Colab GPU session with `MODE = "worker"`.
5. Let the runner copy the active job to `/content` scratch before training.

The Colab runner keeps Google Drive as the persistent source of truth and uses local Colab disk for category-local interim data, RecBole caches, temporary results, and plots. That avoids the old checked-in `run/` workspace while still supporting shared 2-worker execution.

### 1. Predict product carbon footprints

Run the retrieval-only PCF pipeline:

```bash
./.venv/bin/python scripts/predict_carbon.py --device cpu --num-threads 8 --skip-llm
```

Run the paper-faithful local-Qwen pipeline for the full Carbon Catalogue
holdout and the full Amazon metadata set:

```bash
./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --llm-backend transformers \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --transformers-device-map auto \
  --transformers-torch-dtype float16 \
  --evaluation-limit -1 \
  --llm-amazon-limit -1 \
  --eval-output output/pcf/pcf_evaluation_predictions.csv \
  --metrics-output output/pcf/pcf_evaluation_metrics.csv \
  --run-metadata-output output/pcf/pcf_run_metadata.json \
  --amazon-output data/processed/carbon/amazon_pcf_predictions.csv
```

This full-Amazon Qwen pass is intended for a GPU machine (for example
Colab T4). The same command on CPU will be impractically slow.

To reduce wall-clock time, the recommended paper-faithful path is to keep the
Carbon Catalogue evaluation as one run and shard the Amazon few-shot pass by
category (`electronics`, `home_and_kitchen`, `sports_and_outdoors`) across
multiple GPU workers. The repo now supports this directly via
`--amazon-categories`, `--amazon-llm-methods few_shot_llm`, and
[scripts/11_merge_amazon_pcf_shards.py](scripts/11_merge_amazon_pcf_shards.py).
See [docs/FINAL_RUN.md](docs/FINAL_RUN.md) for the exact commands.

Outputs (defaults; override with `--eval-output` / `--metrics-output` / `--amazon-output`):

- `data/processed/carbon/amazon_pcf_predictions.csv`
- `output/results/carbon/pcf_evaluation_predictions.csv`
- `output/results/carbon/pcf_evaluation_metrics.csv`
- `output/results/carbon/pcf_run_metadata.json`

The [final run guide](docs/FINAL_RUN.md) recommends writing evaluation CSVs under **`output/pcf/`** when you want them versioned alongside PCF figures (see *Tracked vs ignored outputs* above). `output/results/` remains gitignored.

### 2. Train candidate generators

Train one of the paper baselines for a category:

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

The default sweep is defined in [configs/reranking/default.yaml](configs/reranking/default.yaml) and matches the current paper/reporting setup: a denser `lambda` tail from `0.90` to `1.00`, top-10 re-ranked lists, and a top-100 candidate pool from RecBole.

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

### 5. Generate paper plots

After running the category/model evaluations you want to include, generate the paper figures referenced from `docs/main.tex`:

```bash
./.venv/bin/python scripts/04_generate_paper_plots.py \
  --results-dir output/results \
  --figure-dir output/figures \
  --summary-output-dir output/results \
  --carbon-metrics-path output/pcf/pcf_evaluation_metrics.csv \
  --carbon-eval-predictions-path output/pcf/pcf_evaluation_predictions.csv \
  --amazon-predictions-path data/processed/carbon/amazon_pcf_predictions.csv
```

Outputs:

- `output/figures/*.png` paper figures
- `output/results/paper_plot_manifest.json`
- `output/results/paper_metrics_summary.csv`

## Key Files

- [docs/main.tex](docs/main.tex)
- [scripts/predict_carbon.py](scripts/predict_carbon.py)
- [scripts/01_train_recommender.py](scripts/01_train_recommender.py)
- [scripts/02_rerank.py](scripts/02_rerank.py)
- [scripts/03_evaluate.py](scripts/03_evaluate.py)
- [scripts/04_generate_paper_plots.py](scripts/04_generate_paper_plots.py)
- [scripts/05_colab_runner.py](scripts/05_colab_runner.py)
- [scripts/06_colab_watch.py](scripts/06_colab_watch.py)
- [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb)
- [notebooks/colab_watch.ipynb](notebooks/colab_watch.ipynb)
- [src/carbon/retrieval.py](src/carbon/retrieval.py)
- [src/recommender/trainer.py](src/recommender/trainer.py)
- [src/reranking/carbon_reranker.py](src/reranking/carbon_reranker.py)
- [src/evaluation/metrics.py](src/evaluation/metrics.py)
