# Carbon-Aware Recommender System

This repository estimates Product Carbon Footprint (PCF) for Amazon products, then uses those estimates in a carbon-aware recommendation pipeline. The new PCF module is retrieval-first: it maps Amazon items to similar Carbon Catalogue products, then optionally uses those retrieved examples as few-shot context for an LLM.

## PCF Estimation Approach

1. Load labeled products from the Carbon Catalogue and unlabeled Amazon product metadata.
2. Build sentence embeddings for product text with `sentence-transformers/all-MiniLM-L6-v2`.
3. Retrieve the top-5 Carbon Catalogue neighbors for each query product with cosine similarity.
4. Estimate PCF with three methods:
   - `neighbor_average`: mean PCF of the top-5 retrieved neighbors
   - `zero_shot_llm`: LLM prediction from the product title alone
   - `few_shot_llm`: LLM prediction conditioned on the retrieved neighbors and their PCFs
5. Evaluate on held-out Carbon Catalogue rows with RMSE, MAE, and Spearman correlation.
6. Write predicted PCFs for Amazon products for downstream re-ranking.

## Key Files

- Runner: [scripts/predict_carbon.py](/Users/noahsyrdal/carbon-aware-recsys/scripts/predict_carbon.py)
- Retrieval module: [src/carbon/retrieval.py](/Users/noahsyrdal/carbon-aware-recsys/src/carbon/retrieval.py)
- Amazon predictions: [data/processed/carbon/amazon_pcf_predictions.csv](/Users/noahsyrdal/carbon-aware-recsys/data/processed/carbon/amazon_pcf_predictions.csv)
- Evaluation predictions: [output/results/carbon/pcf_evaluation_predictions.csv](/Users/noahsyrdal/carbon-aware-recsys/output/results/carbon/pcf_evaluation_predictions.csv)
- Evaluation metrics: [output/results/carbon/pcf_evaluation_metrics.csv](/Users/noahsyrdal/carbon-aware-recsys/output/results/carbon/pcf_evaluation_metrics.csv)
- LLM cache: [data/processed/carbon/llm_prediction_cache.jsonl](/Users/noahsyrdal/carbon-aware-recsys/data/processed/carbon/llm_prediction_cache.jsonl)

## Run

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Run the full retrieval pipeline without LLM calls:

```bash
./.venv/bin/python scripts/predict_carbon.py --device cpu --num-threads 8 --skip-llm
```

Run the LLM evaluation slice:

```bash
OPENAI_API_KEY=... ./.venv/bin/python scripts/predict_carbon.py \
  --device cpu \
  --num-threads 8 \
  --evaluation-limit 100 \
  --amazon-limit 0
```

For reproducible retrieval runs, keep the default seed, stay on CPU, and use `--llm-cache-only` when replaying saved LLM outputs. Live API generations are not fully reproducible provider-side.

## Latest Result

**How to regenerate (Table 1):** From the repo root, with `OPENAI_API_KEY` set for LLM baselines:

```bash
uv run python scripts/predict_carbon.py --evaluation-limit 100 --amazon-limit 0
```

Outputs: `output/results/carbon/pcf_evaluation_metrics.csv` and `pcf_evaluation_predictions.csv`. Use `--amazon-limit 0` to run only the 100-item evaluation and skip full Amazon scoring.

Current checked LLM comparison on a 100-example Carbon Catalogue evaluation slice with `gpt-4.1-mini`:

| method | n | RMSE | MAE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| neighbor_average | 100 | 3,964 | 1,326 | 0.771 |
| zero_shot_llm | 100 | 8,878 | 3,334 | 0.421 |
| few_shot_llm | 100 | 8,328 | 1,696 | 0.853 |

Interpretation:

- `few_shot_llm` achieves the best rank correlation but higher MAE than the neighbor baseline, confirming that retrieved in-context examples primarily help on ranking quality rather than absolute error.
- `zero_shot_llm` remains the weakest baseline (lowest Spearman, highest MAE) even after adding scale/format instructions and clamping, but now operates in a plausible numeric range instead of producing catastrophic outliers.
- The result is directional, not final: the LLM comparison uses 100 evaluation examples to control API cost. For a stronger research claim, rerun on the full Carbon Catalogue or across repeated fixed-seed samples.
