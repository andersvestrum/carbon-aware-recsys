# Final run (paper + recommender pipeline)

Short checklist to reproduce the full workflow after the repository is set up (`python3 -m venv .venv` and `pip install -r requirements.txt`). Paths are relative to the repo root.

**Versioned outputs:** `.gitignore` does **not** include `output/pcf/` or `output/figures/`, so committed CSVs and PNGs stay with the repository. `output/results/` is ignored (recommender runs). If `output/pcf/`, the PCF figures under `output/figures/`, **and** `data/processed/carbon/amazon_pcf_predictions.csv` already match your paper, skip §§1–2.

## 1. PCF estimation (skip if outputs are already final)

Run when you need fresh Carbon Catalogue evaluation rows and Amazon product
PCF scores generated with the paper-faithful local Qwen setup:

```bash
./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --llm-backend transformers \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --transformers-device-map auto \
  --transformers-torch-dtype float16 \
  --evaluation-limit -1 \
  --eval-llm-methods zero_shot_llm few_shot_llm \
  --amazon-llm-methods few_shot_llm \
  --llm-amazon-limit -1 \
  --eval-output output/pcf/pcf_evaluation_predictions.csv \
  --metrics-output output/pcf/pcf_evaluation_metrics.csv \
  --run-metadata-output output/pcf/pcf_run_metadata.json \
  --amazon-output data/processed/carbon/amazon_pcf_predictions.csv
```

Notes:

- `--llm-backend transformers` runs the zero-shot and few-shot baselines with a
  local HuggingFace model instead of the OpenAI client.
- `--llm-model Qwen/Qwen2.5-3B-Instruct` matches the reported run in
  `docs/main.tex`.
- `--eval-llm-methods zero_shot_llm few_shot_llm` preserves the full
  evaluation table for the paper.
- `--amazon-llm-methods few_shot_llm` skips zero-shot on Amazon products,
  because the downstream recommender uses few-shot with neighbour fallback
  rather than the zero-shot column.
- `--llm-amazon-limit -1` means “score the full Amazon metadata set with the
  LLM baselines”, so the downstream `pcf` column is driven by local few-shot
  Qwen predictions rather than silently falling back to neighbour-average for
  every row.
- `--evaluation-limit -1` evaluates the **full** Carbon Catalogue holdout. If
  you omit `--evaluation-limit` and enable the LLM, the script defaults to
  **100** rows to control cost/runtime.
- `--amazon-limit 0` still skips Amazon scoring entirely (evaluation-only).
- On Colab or another GPU box, keeping `--device cpu` is fine: that flag only
  controls the sentence-embedding side; the Qwen pipeline uses
  `--transformers-device-map auto`.
- In practice, the full-Amazon Qwen pass is a GPU job. Running the same
  command on CPU is technically possible but will be prohibitively slow.
- Default outputs without the `--*-output` flags go under `output/results/carbon/`; the commands above match `scripts/08_rerun_pcf_eval.py` and `docs/main.tex` conventions that use `output/pcf/` for evaluation CSVs.

### Parallel Amazon Qwen run (recommended)

The expensive part is the Amazon few-shot pass. The clean way to parallelise it
is to keep the Carbon Catalogue evaluation as one run, then shard the Amazon
metadata by category across multiple GPU workers.

**Step A: run the full evaluation once (no Amazon scoring)**

```bash
./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --llm-backend transformers \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --transformers-device-map auto \
  --transformers-torch-dtype float16 \
  --evaluation-limit -1 \
  --eval-llm-methods zero_shot_llm few_shot_llm \
  --amazon-limit 0 \
  --amazon-output output/pcf/shards/no_amazon_predictions.csv \
  --eval-output output/pcf/pcf_evaluation_predictions.csv \
  --metrics-output output/pcf/pcf_evaluation_metrics.csv \
  --run-metadata-output output/pcf/pcf_run_metadata.json
```

**Step B: run one Amazon shard per category, ideally on separate GPUs**

Electronics:

```bash
./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --evaluation-limit 0 \
  --amazon-categories electronics \
  --llm-backend transformers \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --transformers-device-map auto \
  --transformers-torch-dtype float16 \
  --amazon-llm-methods few_shot_llm \
  --llm-amazon-limit -1 \
  --llm-cache data/processed/carbon/shards/electronics_llm_cache.jsonl \
  --amazon-output data/processed/carbon/shards/electronics_pcf_predictions.csv \
  --eval-output output/pcf/shards/electronics_eval_predictions.csv \
  --metrics-output output/pcf/shards/electronics_eval_metrics.csv \
  --run-metadata-output output/pcf/shards/electronics_run_metadata.json
```

Home and Kitchen:

```bash
./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --evaluation-limit 0 \
  --amazon-categories home_and_kitchen \
  --llm-backend transformers \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --transformers-device-map auto \
  --transformers-torch-dtype float16 \
  --amazon-llm-methods few_shot_llm \
  --llm-amazon-limit -1 \
  --llm-cache data/processed/carbon/shards/home_and_kitchen_llm_cache.jsonl \
  --amazon-output data/processed/carbon/shards/home_and_kitchen_pcf_predictions.csv \
  --eval-output output/pcf/shards/home_and_kitchen_eval_predictions.csv \
  --metrics-output output/pcf/shards/home_and_kitchen_eval_metrics.csv \
  --run-metadata-output output/pcf/shards/home_and_kitchen_run_metadata.json
```

Sports and Outdoors:

```bash
./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --evaluation-limit 0 \
  --amazon-categories sports_and_outdoors \
  --llm-backend transformers \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --transformers-device-map auto \
  --transformers-torch-dtype float16 \
  --amazon-llm-methods few_shot_llm \
  --llm-amazon-limit -1 \
  --llm-cache data/processed/carbon/shards/sports_and_outdoors_llm_cache.jsonl \
  --amazon-output data/processed/carbon/shards/sports_and_outdoors_pcf_predictions.csv \
  --eval-output output/pcf/shards/sports_and_outdoors_eval_predictions.csv \
  --metrics-output output/pcf/shards/sports_and_outdoors_eval_metrics.csv \
  --run-metadata-output output/pcf/shards/sports_and_outdoors_run_metadata.json
```

**Step C: merge the shard outputs into the final Amazon PCF file**

```bash
./.venv/bin/python scripts/11_merge_amazon_pcf_shards.py \
  --inputs \
    data/processed/carbon/shards/electronics_pcf_predictions.csv \
    data/processed/carbon/shards/home_and_kitchen_pcf_predictions.csv \
    data/processed/carbon/shards/sports_and_outdoors_pcf_predictions.csv \
  --output data/processed/carbon/amazon_pcf_predictions.csv
```

Use distinct `--llm-cache` and `--amazon-output` paths per worker. The JSONL
cache is append-only and should not be shared between concurrent writers.
`--evaluation-limit 0` turns the shard jobs into Amazon-only runs; the tiny
evaluation CSVs under `output/pcf/shards/` are just throwaway bookkeeping so
each worker writes to separate files.

## 2. PCF metrics and figures (from row-level predictions)

Skip this block if the repo already contains fresh `output/pcf/pcf_evaluation_metrics_by_subset.csv` and the PNGs under `output/figures/pcf_insights/` and `output/figures/pcf_subset_eval/`.

After `pcf_evaluation_predictions.csv` exists:

```bash
./.venv/bin/python scripts/08_rerun_pcf_eval.py \
  --predictions-path output/pcf/pcf_evaluation_predictions.csv

./.venv/bin/python scripts/07_plot_pcf_insights.py \
  --predictions-path output/pcf/pcf_evaluation_predictions.csv \
  --paper-style

./.venv/bin/python scripts/09_plot_pcf_subset_comparison.py
./.venv/bin/python scripts/10_plot_pcf_downstream_figures.py
```

Use `--panel` on `09_plot_pcf_subset_comparison.py` if you want the combined accuracy panel figure.

**Where this lands (all under `output/`):**

| Step | Main inputs | Main outputs |
|------|-------------|--------------|
| `08` | Row-level `pcf_evaluation_predictions.csv` | `output/pcf/pcf_evaluation_metrics_by_subset.csv` |
| `07` | Same predictions CSV (pass `--predictions-path` if not under `output/results/carbon/`) | PNGs under `output/figures/pcf_insights/` |
| `09` | Metrics from `08` (default `output/pcf/pcf_evaluation_metrics_by_subset.csv`) | PNGs under `output/figures/pcf_subset_eval/` |
| `10` | Amazon + evaluation CSVs (see script defaults) | PNGs under `output/figures/pcf_insights/` by default |

For Overleaf, copy or symlink only what `docs/main.tex` references into `docs/figures/` (see §5). Figures under `output/figures/` can remain the canonical copies in git.

## 3. Merge interactions + PCF for training

Ensures `data/interim/` splits include the `pcf` column (uses `data/processed/carbon/amazon_pcf_predictions.csv` when present; otherwise builds retrieval estimates):

```bash
./.venv/bin/python -m src.data.preprocess
```

## 4. Recommender experiments (paper order)

For each category (`electronics`, `home_and_kitchen`, `sports_and_outdoors`) and model (`BPR`, `NeuMF`, `LightGCN`):

```bash
./.venv/bin/python scripts/01_train_recommender.py --category <cat> --model <model>
./.venv/bin/python scripts/02_rerank.py --category <cat> --model <model>
./.venv/bin/python scripts/03_evaluate.py --category <cat> --model <model>
```

Then aggregate figures for the paper:

```bash
./.venv/bin/python scripts/04_generate_paper_plots.py \
  --results-dir output/results \
  --figure-dir output/figures \
  --summary-output-dir output/results \
  --carbon-metrics-path output/pcf/pcf_evaluation_metrics.csv \
  --carbon-eval-predictions-path output/pcf/pcf_evaluation_predictions.csv \
  --amazon-predictions-path data/processed/carbon/amazon_pcf_predictions.csv
```

## 5. LaTeX

Copy or symlink generated PNGs into `docs/figures/` as referenced by `docs/main.tex`, then compile as usual.

For GPU Colab jobs (optional), see the main [README](../README.md) Colab section and `scripts/05_colab_runner.py` / `scripts/06_colab_watch.py`.
