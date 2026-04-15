# Sampled + Sharded Electronics Few-Shot Run

This runbook explains how to execute sampled-shard few-shot prediction in parallel with:

- `notebooks/colab_pcf_llm_few_shot_electronics.ipynb`
- `scripts/12_merge_sampled_shards.py`

## 1) Configure one canonical run setup

In the notebook config cell, set:

- `SAMPLE_SIZE_K` (Xk target sample size, i.e. `Xk * 1000` rows)
- `SAMPLE_SEED = 42`
- `NUM_SHARDS` (number of parallel runners)
- `LLM_BACKEND` (`'transformers'` or `'vllm'`; keep one backend per run)
- `LLM_REASONING_STYLE = 'terse'`
- `MAX_NEW_TOKENS = 128` (or your preferred cap)

Keep these values identical across all shard runners.

## 2) Test path (50 rows per shard runner)

Purpose: verify timing, logging, and output schema before larger runs.

You should open **multiple Colab tabs/sessions** and run the **same notebook**
in each tab:

- `notebooks/colab_pcf_llm_few_shot_electronics.ipynb`

In each tab, edit only the config cell:

- `## 3) Run Configuration and Logging`

For each runner/tab:

1. Use a unique `SHARD_ID` in `0..NUM_SHARDS-1`
2. Set `RUN_MODE = 'test50'`
3. Keep `SAMPLE_SIZE_K`, `SAMPLE_SEED`, `NUM_SHARDS`, and `LLM_BACKEND` identical across all tabs
4. Run the notebook **top-to-bottom** (all cells in order)

Example for 4 parallel tabs:

- Tab 1: `SHARD_ID = 0`
- Tab 2: `SHARD_ID = 1`
- Tab 3: `SHARD_ID = 2`
- Tab 4: `SHARD_ID = 3`

Each runner writes to a unique folder:

`/content/drive/MyDrive/carbon-aware-recsys-colab/pcf/electronics_sampled_shards/<run_tag>/`

with shard-specific files:

- `predictions_<run_tag>_test50.csv`
- `predictions_<run_tag>_test50.jsonl`
- `run_metadata_<run_tag>_test50.json`
- `run_<run_tag>.log`
- `llm_cache_<run_tag>.jsonl`

## 3) Parallel sampled-shard production run

For each parallel Colab tab/session:

1. Set the same `SAMPLE_SIZE_K`, `SAMPLE_SEED`, `NUM_SHARDS`, and `LLM_BACKEND`
2. Set unique `SHARD_ID`
3. Set `RUN_MODE = 'sampled_shard'`
4. Run the notebook **from the first cell to the last cell**

Important:

- Do not start from the middle; the workload, logger, model client, and output paths
  are created by earlier cells.
- All tabs use the same notebook; parallelism comes from different `SHARD_ID`s.

Each runner outputs:

- `predictions_<run_tag>_sampled_shard.csv`
- `predictions_<run_tag>_sampled_shard.jsonl`
- `run_metadata_<run_tag>_sampled_shard.json`

## 4) Merge shard CSV outputs

From repo root:

```bash
./.venv/bin/python scripts/12_merge_sampled_shards.py \
  --input-dir "/path/to/run_root" \
  --glob "predictions_*_sampled_shard.csv" \
  --output "/path/to/run_root/predictions_merged_sampled.csv"
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
