# Final run (paper + recommender pipeline)

Short checklist to reproduce the full workflow after the repository is set up (`python3 -m venv .venv` and `pip install -r requirements.txt`). Paths are relative to the repo root.

**Versioned outputs:** `.gitignore` does **not** include `output/pcf/` or `output/figures/`, so committed CSVs and PNGs stay with the repository. `output/results/` is ignored (recommender runs).

**Dataset scope:** The recommender pipeline runs on the subset of Amazon interaction products that have few-shot LLM PCF predictions from the Colab runs. Products without a prediction are excluded. This gives ~8k/7k/3k unique products per category with full PCF coverage and no fallback zeros.

---

## 1. PCF estimation — Colab sampled-shard runs (already done)

PCF predictions were produced by running `notebooks/colab_pcf_llm_few_shot_{category}_{shard}.ipynb` in parallel on Google Colab (8 shards per category, `xk50_seed42_ns12_vllm_iter2`). Results were downloaded as zip files and extracted to `{category}_sampled_shards/` at the repo root.

**If re-running from scratch**, extract new shard results and run the merge steps below.

### Merge shard results per category

```bash
for cat in electronics home_and_kitchen sports_and_outdoors; do
  ./.venv/bin/python scripts/12_merge_sampled_shards.py \
    --input-dir ${cat}_sampled_shards \
    --glob “**/predictions_xk50_seed42_ns12_sh*_vllm_iter2_sampled_shard.csv” \
    --output data/processed/carbon/shards/${cat}_pcf_predictions.csv
done
```

### Merge categories into the final Amazon PCF file

```bash
./.venv/bin/python scripts/11_merge_amazon_pcf_shards.py \
  --inputs \
    data/processed/carbon/shards/electronics_pcf_predictions.csv \
    data/processed/carbon/shards/home_and_kitchen_pcf_predictions.csv \
    data/processed/carbon/shards/sports_and_outdoors_pcf_predictions.csv \
  --output data/processed/carbon/amazon_pcf_predictions.csv
```

Output: `data/processed/carbon/amazon_pcf_predictions.csv` — ~100k products with `few_shot_llm_pcf` and selected `pcf`.

## 2. PCF metrics and figures (from row-level predictions)

Skip if `output/pcf/pcf_evaluation_metrics_by_subset.csv` and PNGs under `output/figures/` are already final.

```bash
./.venv/bin/python scripts/08_rerun_pcf_eval.py \
  --predictions-path output/pcf/pcf_evaluation_predictions.csv

./.venv/bin/python scripts/07_plot_pcf_insights.py \
  --predictions-path output/pcf/pcf_evaluation_predictions.csv \
  --paper-style

./.venv/bin/python scripts/09_plot_pcf_subset_comparison.py
./.venv/bin/python scripts/10_plot_pcf_downstream_figures.py
```

## 3. Run everything

```bash
./run_final.sh
```

This runs in order:
1. `scripts/update_interim_pcf.py` — filters raw interactions to few-shot predicted products, writes `data/interim/`
2. `scripts/01_train_recommender.py` → `02_rerank.py` → `03_evaluate.py` for all 3 categories × 3 models
3. `scripts/04_generate_paper_plots.py` — aggregates results into figures

**Dataset sizes after step 1:**

| Category | Train | Val | Test | Unique products |
|---|---|---|---|---|
| Electronics | 280k rows | 27k | 21k | 7,989 |
| Home & Kitchen | 185k rows | 18k | 15k | 6,778 |
| Sports & Outdoors | 49k rows | 6k | 5k | 3,052 |

Note: The Colab runs sampled ~33k products per category from the full Amazon catalog. Only the subset that users have interacted with appears here (~8k/7k/3k), as collaborative filtering requires observed interactions.

**Outputs:** `output/results/` (metrics), `output/figures/` (plots)

## 4. LaTeX

Copy or symlink generated PNGs into `docs/figures/` as referenced by `docs/main.tex`, then compile as usual.
