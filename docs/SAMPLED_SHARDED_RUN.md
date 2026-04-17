# Sampled + Sharded Few-Shot Run (CF-Eligible v2)

This runbook explains how to execute sampled-shard few-shot prediction in parallel using the CF-eligible v2 shard notebooks and then merge outputs safely:

- `notebooks/colab_pcf_llm_few_shot_<category>_00.ipynb`
- `notebooks/colab_pcf_llm_few_shot_<category>_01.ipynb`
- `notebooks/colab_pcf_llm_few_shot_<category>_02.ipynb`
- `scripts/12_merge_sampled_shards.py`

## Current backend status

The notebook now supports both backends through one config flag:

- `LLM_BACKEND = 'transformers'` (baseline)
- `LLM_BACKEND = 'vllm'` (recommended for throughput on larger runs)

Notes:

- vLLM is installed in the setup cell as an optional dependency.
- If vLLM install fails in a runtime, use `LLM_BACKEND = 'transformers'` for that run.
- Keep one backend per run. Do not mix backends across shard tabs for the same run.
- The backend is included in `RUN_TAG`, so outputs and cache files remain backend-isolated.

## 1) Fixed v2 setup (do not override shard controls)

In each v2 notebook, shard controls are intentionally fixed:

- `TARGET_SAMPLE_SIZE = 24_000`
- `NUM_SHARDS = 3`
- notebook-specific `SHARD_ID` from filename (`*_00 -> 0`, `*_01 -> 1`, `*_02 -> 2`)
- `RUN_LABEL = 'v2_cf_eligible'`
- `RUN_MODE = 'sampled_shard'` (set to `test50` only for smoke tests)

The notebooks also enforce a CF-eligibility gate:

- build interaction `parent_asin` universe from train/val/test for category
- filter metadata to interaction-supported ASINs only
- fail fast if fewer than 24,000 eligible ASINs are available
- assert shard outputs remain unique and interaction-supported

## 2) Test path (50 rows per shard runner)

Purpose: verify timing, logging, and output schema before larger runs.

Open 3 Colab tabs/sessions and run one fixed shard notebook in each:

- tab 1: `notebooks/colab_pcf_llm_few_shot_<category>_00.ipynb`
- tab 2: `notebooks/colab_pcf_llm_few_shot_<category>_01.ipynb`
- tab 3: `notebooks/colab_pcf_llm_few_shot_<category>_02.ipynb`

For each runner/tab:

1. Set `RUN_MODE = 'test50'`
2. Keep `SAMPLE_SEED` and `LLM_BACKEND` identical across tabs
3. Run the notebook **top-to-bottom** (all cells in order)

In `test50` mode, the notebook takes the first 50 rows of each shard's workload. This means test outputs are shard-specific and not intended to represent a merged global 50-row benchmark unless you merge and inspect across shards.

Each runner writes to a unique folder:

`/content/drive/MyDrive/carbon-aware-recsys-colab/pcf/<category>_sampled_shards_v2_cf_eligible/<run_tag>/`

with shard-specific files:

- `predictions_<run_tag>_test50.csv`
- `predictions_<run_tag>_test50.jsonl`
- `run_metadata_<run_tag>_test50.json`
- `run_<run_tag>.log`
- `llm_cache_<run_tag>.jsonl`

## 3) Parallel sampled-shard production run (24k/category)

For each parallel Colab tab/session:

1. Open one of the three shard notebooks (`*_00`, `*_01`, `*_02`) for the same category
2. Keep `SAMPLE_SEED` and `LLM_BACKEND` the same across tabs
3. Set `RUN_MODE = 'sampled_shard'`
4. Run the notebook **from the first cell to the last cell**

Important:

- Do not start from the middle; the workload, logger, model client, and output paths
  are created by earlier cells.
- Parallelism comes from running the three shard-specific notebooks in parallel.
- Reusing a tab for another shard is supported by opening the corresponding shard notebook file and rerunning from the first cell.

Each runner outputs:

- `predictions_<run_tag>_sampled_shard.csv`
- `predictions_<run_tag>_sampled_shard.jsonl`
- `run_metadata_<run_tag>_sampled_shard.json`

## 4) Merge shard CSV outputs

From repo root, merge per category:

```bash
./.venv/bin/python scripts/12_merge_sampled_shards.py \
  --input-dir "/path/to/<category>_sampled_shards_v2_cf_eligible" \
  --glob "**/predictions_*_sampled_shard.csv" \
  --output "/path/to/<category>_sampled_shards_v2_cf_eligible/predictions_<category>_xk24_ns3_v2_cf_eligible_merged.csv"
```

Alternative with explicit files:

```bash
./.venv/bin/python scripts/12_merge_sampled_shards.py \
  --inputs \
    "/path/to/shard0.csv" \
    "/path/to/shard1.csv" \
    "/path/to/shard2.csv" \
  --output "/path/to/predictions_merged_sampled.csv"
```

By default, merge fails if duplicate `parent_asin` appears across shards.
Use `--allow-duplicates` only if duplicates are expected intentionally.

## 5) Required validation checks before downstream use

After each category merge:

```bash
./.venv/bin/python - <<'PY'
import pandas as pd

path = "/path/to/merged.csv"
df = pd.read_csv(path)

print("rows", len(df))
print("unique_parent_asin", df["parent_asin"].nunique(dropna=True))
print("duplicates", int(df["parent_asin"].duplicated().sum()))
assert len(df) == 24000, "Merged rows must be exactly 24,000"
assert df["parent_asin"].nunique(dropna=True) == 24000, "parent_asin must be unique"
print("ok")
PY
```

In addition, confirm each shard metadata JSON contains:

- `sample_size_required = 24000`
- `num_shards_required = 3`
- `output_parent_asin_in_interactions = true`
