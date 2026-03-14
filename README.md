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

Current checked LLM comparison on a 100-example Carbon Catalogue evaluation slice with `gpt-4.1-mini`:

| method | n | RMSE | MAE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| neighbor_average | 100 | 3964.05 | 1325.51 | 0.7709 |
| zero_shot_llm | 100 | 499999999.81 | 50002277.35 | 0.4902 |
| few_shot_llm | 100 | 3479.51 | 870.61 | 0.8528 |

Interpretation:

- `few_shot_llm` is the strongest method in this run. It improves both absolute error and rank correlation over the neighbor-average baseline.
- `zero_shot_llm` is unstable as a numeric regressor in the current prompt format because it can produce extreme scale errors.
- The result is directional, not final: the LLM comparison above uses 100 evaluation examples to control API cost. For a stronger research claim, rerun on the full Carbon Catalogue or across repeated fixed-seed samples.
