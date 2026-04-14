"""
Run retrieval-based Product Carbon Footprint estimation.

Outputs:
    data/processed/carbon/amazon_pcf_predictions.csv
    output/results/carbon/pcf_evaluation_predictions.csv
    output/results/carbon/pcf_evaluation_metrics.csv

Notes:
    - The nearest-neighbor baseline runs across the full Amazon metadata set.
    - LLM baselines are optional and can run either via OpenAI or a local
      HuggingFace Transformers model.
    - If LLM evaluation is enabled without an explicit evaluation limit, the
      script defaults to 100 Carbon Catalogue examples to control cost.
      Pass --evaluation-limit -1 to evaluate the full catalogue.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.amazon_loader import load_meta
from src.carbon.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_LLM_REASONING_STYLE,
    LLM_METHODS,
    LLM_REASONING_STYLES,
    DEFAULT_TOP_K,
    NumericLLMClient,
    PROCESSED_CARBON_DIR,
    RESULTS_DIR,
    OpenAILLMClient,
    PCFRetrievalEstimator,
    RetrievalConfig,
    TransformersLLMClient,
    set_global_determinism,
)

log = logging.getLogger(__name__)

ALL_AMAZON_CATEGORIES = ("electronics", "home_and_kitchen", "sports_and_outdoors")


@dataclass(frozen=True)
class OutputPaths:
    amazon: Path
    evaluation: Path
    metrics: Path
    run_metadata: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval-based Product Carbon Footprint estimation",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformer model used for semantic retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of Carbon Catalogue neighbors to retrieve.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for sentence-transformers.",
    )
    parser.add_argument(
        "--evaluation-limit",
        type=int,
        default=None,
        help="Carbon Catalogue eval size; omit for default (100 if LLM on, else full). "
        "Use -1 for the full catalogue.",
    )
    parser.add_argument(
        "--amazon-limit",
        type=int,
        default=None,
        help="Optional number of Amazon metadata rows to score.",
    )
    parser.add_argument(
        "--amazon-categories",
        nargs="+",
        choices=ALL_AMAZON_CATEGORIES,
        default=None,
        help="Optional subset of Amazon metadata categories to score. "
        "Useful for parallel sharded runs.",
    )
    parser.add_argument(
        "--llm-amazon-limit",
        type=int,
        default=None,
        help="Optional number of Amazon rows to score with the LLM baselines. "
        "Use -1 for all rows. Defaults to 0 for OpenAI and all rows for local Transformers.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("openai", "transformers"),
        default="openai",
        help="Backend used for the zero-shot and few-shot LLM baselines.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name for the chosen LLM backend. "
        "Defaults to gpt-4.1-mini for OpenAI and Qwen/Qwen2.5-3B-Instruct for local Transformers.",
    )
    parser.add_argument(
        "--transformers-device-map",
        default="auto",
        help="device_map passed to HuggingFace pipeline when --llm-backend transformers.",
    )
    parser.add_argument(
        "--transformers-torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype for the local Transformers backend.",
    )
    parser.add_argument(
        "--transformers-max-new-tokens",
        type=int,
        default=384,
        help="max_new_tokens for the local Transformers backend.",
    )
    parser.add_argument(
        "--llm-reasoning-style",
        choices=LLM_REASONING_STYLES,
        default=DEFAULT_LLM_REASONING_STYLE,
        help="Prompt verbosity style for LLM methods (default: detailed).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip zero-shot and few-shot LLM baselines.",
    )
    parser.add_argument(
        "--eval-llm-methods",
        nargs="+",
        choices=LLM_METHODS,
        default=None,
        help="LLM methods to run for Carbon Catalogue evaluation. Defaults to both.",
    )
    parser.add_argument(
        "--amazon-llm-methods",
        nargs="+",
        choices=LLM_METHODS,
        default=None,
        help="LLM methods to run for Amazon scoring. Defaults to both. "
        "Use `few_shot_llm` for the downstream recommender pipeline.",
    )
    parser.add_argument(
        "--no-amazon-metadata",
        action="store_true",
        help="Use title-only Amazon embeddings instead of title + metadata.",
    )
    parser.add_argument(
        "--no-carbon-context",
        action="store_true",
        help="Use title-only Carbon Catalogue embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic retrieval and evaluation.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Execution threads for deterministic embedding runs.",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic retrieval settings.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=1,
        help="Number of concurrent LLM prediction workers (default: 1). "
        "Increase for API-based or vLLM-backed models.",
    )
    parser.add_argument(
        "--llm-cache-only",
        action="store_true",
        help="Use only cached LLM responses; do not make live API calls.",
    )
    parser.add_argument(
        "--amazon-output",
        default=str(PROCESSED_CARBON_DIR / "amazon_pcf_predictions.csv"),
        help="CSV path for Amazon PCF predictions.",
    )
    parser.add_argument(
        "--eval-output",
        default=str(RESULTS_DIR / "pcf_evaluation_predictions.csv"),
        help="CSV path for row-level evaluation predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(RESULTS_DIR / "pcf_evaluation_metrics.csv"),
        help="CSV path for evaluation summary metrics.",
    )
    parser.add_argument(
        "--run-metadata-output",
        default=str(RESULTS_DIR / "pcf_run_metadata.json"),
        help="JSON path for the run metadata summary.",
    )
    parser.add_argument(
        "--llm-cache",
        default=str(PROCESSED_CARBON_DIR / "llm_prediction_cache.jsonl"),
        help="JSONL cache for LLM prompt completions.",
    )
    return parser.parse_args()


def _resolve_llm_model(args: argparse.Namespace) -> str:
    if args.llm_model:
        return args.llm_model
    if args.llm_backend == "transformers":
        return DEFAULT_LOCAL_LLM_MODEL
    return DEFAULT_LLM_MODEL


def _resolve_llm_methods(methods: list[str] | None) -> tuple[str, ...]:
    return tuple(methods) if methods else LLM_METHODS


def _build_llm_client(args: argparse.Namespace) -> NumericLLMClient | None:
    if args.skip_llm:
        log.info("Skipping LLM baselines by request.")
        return None

    llm_model = _resolve_llm_model(args)
    if args.llm_backend == "transformers":
        client = TransformersLLMClient(
            model=llm_model,
            torch_dtype=args.transformers_torch_dtype,
            device_map=args.transformers_device_map,
            max_new_tokens=args.transformers_max_new_tokens,
            reasoning_style=args.llm_reasoning_style,
        )
        log.info("Using local Transformers LLM backend with model %s", llm_model)
        return client

    client = OpenAILLMClient(
        model=llm_model,
        reasoning_style=args.llm_reasoning_style,
    )
    if client.is_available:
        return client

    if args.llm_cache_only:
        log.info(
            "OPENAI_API_KEY not found. Replaying cached LLM responses only for model %s.",
            llm_model,
        )
        return None

    log.info("OPENAI_API_KEY not found. LLM baselines will be skipped.")
    return None


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %s rows to %s", f"{len(df):,}", path)


def _save_amazon_predictions(
    df: pd.DataFrame,
    path: Path,
    *,
    amazon_limit: int | None,
) -> None:
    if amazon_limit == 0 and df.empty and path.exists():
        log.info(
            "Amazon scoring limit is 0. Preserving existing predictions at %s.",
            path,
        )
        return

    _save_csv(df, path)


def _resolve_evaluation_limit(
    requested_limit: int | None,
    llm_client: NumericLLMClient | None,
) -> int | None:
    if requested_limit is not None and requested_limit < 0:
        return None
    if requested_limit is not None or llm_client is None:
        return requested_limit

    log.info(
        "LLM baselines enabled without --evaluation-limit. "
        "Defaulting to 100 evaluation examples to control cost."
    )
    return 100


def _resolve_llm_amazon_limit(
    requested_limit: int | None,
    *,
    llm_client: NumericLLMClient | None,
    llm_backend: str,
) -> int | None:
    if requested_limit is not None and requested_limit < 0:
        return None
    if requested_limit is not None:
        return requested_limit
    if llm_client is None:
        return 0
    if llm_backend == "transformers":
        log.info(
            "Local Transformers LLM enabled without --llm-amazon-limit. "
            "Defaulting to all Amazon rows."
        )
        return None
    log.info(
        "OpenAI LLM enabled without --llm-amazon-limit. "
        "Defaulting to 0 Amazon rows to control cost."
    )
    return 0


def _load_amazon_metadata(limit: int | None) -> pd.DataFrame:
    return _load_selected_amazon_metadata(limit=limit, categories=None)


def _load_selected_amazon_metadata(
    *,
    limit: int | None,
    categories: list[str] | None,
) -> pd.DataFrame:
    if limit == 0:
        log.info("Amazon scoring disabled with --amazon-limit 0.")
        return pd.DataFrame()

    if categories:
        frames = []
        log.info("Loading Amazon metadata for categories: %s", ", ".join(categories))
        for category in categories:
            frame = load_meta(category)
            copy = frame.copy()
            copy["source_category"] = category
            frames.append(copy)
        amazon_meta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        amazon_meta = PCFRetrievalEstimator.load_all_amazon_metadata()

    if limit is not None and limit < len(amazon_meta):
        amazon_meta = amazon_meta.head(limit).copy()
        log.info("Scoring a limited Amazon slice: %s products", f"{len(amazon_meta):,}")
    else:
        log.info("Scoring all Amazon metadata rows: %s products", f"{len(amazon_meta):,}")
    return amazon_meta


def _build_estimator(
    args: argparse.Namespace,
    *,
    deterministic: bool,
) -> PCFRetrievalEstimator:
    config = RetrievalConfig(
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_amazon_metadata=not args.no_amazon_metadata,
        use_carbon_context=not args.no_carbon_context,
        device=args.device,
        random_seed=args.seed,
        deterministic=deterministic,
        num_threads=args.num_threads,
        llm_workers=args.llm_workers,
    )
    return PCFRetrievalEstimator(config)


def _resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    return OutputPaths(
        amazon=Path(args.amazon_output),
        evaluation=Path(args.eval_output),
        metrics=Path(args.metrics_output),
        run_metadata=Path(args.run_metadata_output),
    )


def _build_run_metadata(
    args: argparse.Namespace,
    *,
    deterministic: bool,
    llm_client: NumericLLMClient | None,
    llm_model: str | None,
    llm_amazon_limit: int | None,
    eval_llm_methods: Sequence[str],
    amazon_llm_methods: Sequence[str],
    amazon_predictions: pd.DataFrame,
    eval_predictions: pd.DataFrame,
    amazon_output: Path,
    eval_output: Path,
    metrics_output: Path,
) -> dict[str, object]:
    return {
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "seed": args.seed,
        "deterministic": deterministic,
        "num_threads": args.num_threads,
        "amazon_rows": int(len(amazon_predictions)),
        "evaluation_rows": int(len(eval_predictions)),
        "llm_enabled": llm_client is not None,
        "llm_backend": args.llm_backend if llm_client is not None else None,
        "llm_model": llm_model if llm_client is not None else None,
        "amazon_categories": args.amazon_categories,
        "eval_llm_methods": list(eval_llm_methods),
        "amazon_llm_methods": list(amazon_llm_methods),
        "llm_amazon_limit": llm_amazon_limit,
        "llm_reasoning_style": args.llm_reasoning_style,
        "transformers_max_new_tokens": args.transformers_max_new_tokens,
        "llm_cache_only": args.llm_cache_only,
        "amazon_output": str(amazon_output),
        "evaluation_output": str(eval_output),
        "metrics_output": str(metrics_output),
    }


def _save_run_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    log.info("Saved run metadata to %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    llm_model = _resolve_llm_model(args)
    llm_client = _build_llm_client(args)
    eval_llm_methods = _resolve_llm_methods(args.eval_llm_methods)
    amazon_llm_methods = _resolve_llm_methods(args.amazon_llm_methods)
    deterministic = not args.non_deterministic
    set_global_determinism(
        args.seed,
        deterministic=deterministic,
        num_threads=args.num_threads,
    )
    evaluation_limit = _resolve_evaluation_limit(args.evaluation_limit, llm_client)
    llm_amazon_limit = _resolve_llm_amazon_limit(
        args.llm_amazon_limit,
        llm_client=llm_client,
        llm_backend=args.llm_backend,
    )
    estimator = _build_estimator(args, deterministic=deterministic)

    estimator.fit_carbon_catalogue()

    llm_cache_path = Path(args.llm_cache)
    eval_predictions, eval_metrics = estimator.evaluate_on_carbon_catalogue(
        limit=evaluation_limit,
        random_state=args.seed,
        llm_client=llm_client,
        llm_model_name=llm_model,
        llm_cache_path=llm_cache_path,
        llm_cache_only=args.llm_cache_only,
        llm_methods=eval_llm_methods,
        llm_reasoning_style=args.llm_reasoning_style,
    )

    amazon_meta = _load_selected_amazon_metadata(
        limit=args.amazon_limit,
        categories=args.amazon_categories,
    )

    llm_amazon_limit = None if args.llm_amazon_limit < 0 else args.llm_amazon_limit
    amazon_predictions = estimator.predict_amazon_products(
        amazon_meta,
        llm_client=llm_client,
        llm_model_name=llm_model,
        llm_cache_path=llm_cache_path,
        llm_limit=llm_amazon_limit,
        llm_cache_only=args.llm_cache_only,
        llm_methods=amazon_llm_methods,
        llm_reasoning_style=args.llm_reasoning_style,
    )

    output_paths = _resolve_output_paths(args)

    _save_amazon_predictions(
        amazon_predictions,
        output_paths.amazon,
        amazon_limit=args.amazon_limit,
    )
    _save_csv(eval_predictions, output_paths.evaluation)
    _save_csv(eval_metrics, output_paths.metrics)

    run_metadata = _build_run_metadata(
        args,
        deterministic=deterministic,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_amazon_limit=llm_amazon_limit,
        eval_llm_methods=eval_llm_methods,
        amazon_llm_methods=amazon_llm_methods,
        amazon_predictions=amazon_predictions,
        eval_predictions=eval_predictions,
        amazon_output=output_paths.amazon,
        eval_output=output_paths.evaluation,
        metrics_output=output_paths.metrics,
    )
    _save_run_metadata(output_paths.run_metadata, run_metadata)


if __name__ == "__main__":
    main()
