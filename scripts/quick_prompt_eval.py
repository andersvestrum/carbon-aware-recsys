#!/usr/bin/env python
"""Quick eval of the new LCA/CoT prompt on the carbon catalogue.

Leave-one-out over a small random sample; reports RMSE / MAE / Spearman
for nearest-neighbor, zero-shot LLM, and few-shot LLM methods.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.carbon.retrieval import (
    PCFRetrievalEstimator,
    RetrievalConfig,
    OpenAILLMClient,
)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model", default="gpt-4o-mini")
    args = p.parse_args()

    cfg = RetrievalConfig(
        top_k=5,
        deterministic=True,
        random_seed=args.seed,
        num_threads=1,
    )
    est = PCFRetrievalEstimator(cfg)
    est.fit_carbon_catalogue()

    client = OpenAILLMClient(model=args.llm_model)
    if not client.is_available:
        raise SystemExit("OPENAI_API_KEY not set")

    # Use a fresh cache so we actually hit the live LLM with the new prompt
    cache_path = ROOT / "output" / "quick_eval_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    preds, metrics = est.evaluate_on_carbon_catalogue(
        limit=args.limit,
        random_state=args.seed,
        llm_client=client,
        llm_model_name=args.llm_model,
        llm_cache_path=cache_path,
    )
    print("\n=== METRICS (n=", args.limit, ") ===", sep="")
    print(metrics.to_string(index=False))
    print("\nSample predictions:")
    cols = [
        "pcf",
        "neighbor_average_pcf",
        "zero_shot_llm_pcf",
        "few_shot_llm_pcf",
    ]
    cols = [c for c in cols if c in preds.columns]
    print(preds[cols].head(10).to_string())


if __name__ == "__main__":
    main()
