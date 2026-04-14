# Final run (paper + recommender pipeline)

Short checklist to reproduce the full workflow after the repository is set up (`python3 -m venv .venv` and `pip install -r requirements.txt`). Paths are relative to the repo root.

**Versioned outputs:** `.gitignore` does **not** include `output/pcf/` or `output/figures/`, so committed CSVs and PNGs stay with the repository. `output/results/` is ignored (recommender runs). If `output/pcf/` and the PCF figures under `output/figures/` already match your paper, skip §§1–2.

## 1. PCF estimation (skip if outputs are already final)

Run when you need fresh Carbon Catalogue evaluation rows and/or Amazon product PCF scores:

```bash
OPENAI_API_KEY=... ./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --evaluation-limit -1 \
  --eval-output output/pcf/pcf_evaluation_predictions.csv \
  --metrics-output output/pcf/pcf_evaluation_metrics.csv \
  --amazon-output data/processed/carbon/amazon_pcf_predictions.csv
```

Notes:

- `--evaluation-limit -1` evaluates the **full** Carbon Catalogue holdout. If you omit `--evaluation-limit` and enable the LLM, the script defaults to **100** rows to limit API cost.
- `--amazon-limit 0` skips Amazon scoring (evaluation-only).
- Default outputs without the `--*-output` flags go under `output/results/carbon/`; the commands above match `scripts/08_rerun_pcf_eval.py` and `docs/main.tex` conventions that use `output/pcf/` for evaluation CSVs.

## 2. PCF metrics and figures (from row-level predictions)

Skip this block if the repo already contains fresh `output/pcf/pcf_evaluation_metrics_by_subset.csv` and the PNGs under `output/figures/pcf_insights/` and `output/figures/pcf_subset_eval/`.

After `pcf_evaluation_predictions.csv` exists:

```bash
./.venv/bin/python scripts/08_rerun_pcf_eval.py \
  --predictions-path output/pcf/pcf_evaluation_predictions.csv

./.venv/bin/python scripts/07_plot_pcf_insights.py \
  --predictions-path output/pcf/pcf_evaluation_predictions.csv

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
./.venv/bin/python scripts/04_generate_paper_plots.py
```

## 5. LaTeX

Copy or symlink generated PNGs into `docs/figures/` as referenced by `docs/main.tex`, then compile as usual.

For GPU Colab jobs (optional), see the main [README](../README.md) Colab section and `scripts/05_colab_runner.py` / `scripts/06_colab_watch.py`.
