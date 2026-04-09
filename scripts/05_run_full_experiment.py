#!/usr/bin/env python
"""
Orchestrate a larger full-data experiment run with caching and optional Colab-friendly parallelism.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]
DEFAULT_MODELS = ["BPR", "NeuMF", "LightGCN"]


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_dir": run_dir,
        "cache_dir": run_dir / "cache",
        "carbon_cache_dir": run_dir / "cache" / "carbon",
        "recbole_dir": run_dir / "cache" / "recbole",
        "model_dir": run_dir / "cache" / "checkpoints",
        "results_dir": run_dir / "results",
        "figures_dir": run_dir / "figures",
        "logs_dir": run_dir / "logs",
    }


def _ensure_run_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def _dataset_summary(interim_dir: Path, categories: list[str], top_k_candidates: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in categories:
        for split in ["train", "val", "test"]:
            path = interim_dir / split / f"{category}.csv"
            df = pd.read_csv(path, usecols=["user_id", "parent_asin", "timestamp"])
            rows.append(
                {
                    "category": category,
                    "split": split,
                    "rows": int(len(df)),
                    "users": int(df["user_id"].nunique()),
                    "items": int(df["parent_asin"].nunique()),
                    "timestamp_min": int(df["timestamp"].min()),
                    "timestamp_max": int(df["timestamp"].max()),
                    "candidate_rows_at_top_k": int(df["user_id"].nunique() * top_k_candidates)
                    if split == "test"
                    else None,
                }
            )
    return pd.DataFrame(rows)


def _default_parallelism(requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)

    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0

    if gpu_count > 1:
        return gpu_count
    if gpu_count == 1:
        return 1
    return 1


def _prepare_recbole_cache(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    from src.recommender.recbole_formatter import format_category_for_recbole

    stats_rows: list[dict[str, object]] = []
    for category in args.categories:
        output_dir, dataset_name = format_category_for_recbole(
            category,
            dataset_name=category,
            max_users=args.max_users,
            interim_dir=args.interim_dir,
            output_root=paths["recbole_dir"],
            benchmark_splits=True,
            force=args.force,
        )
        stats_path = output_dir / "dataset_stats.json"
        if not stats_path.exists():
            continue
        with stats_path.open(encoding="utf-8") as handle:
            stats = json.load(handle)
        stats_rows.append(
            {
                "category": category,
                "dataset_name": dataset_name,
                "max_users": stats.get("max_users"),
                "combined_rows": stats["combined"]["rows"],
                "combined_users": stats["combined"]["users"],
                "combined_items": stats["combined"]["items"],
                "train_rows": stats["per_split"]["train"]["rows"],
                "valid_rows": stats["per_split"]["val"]["rows"],
                "test_rows": stats["per_split"]["test"]["rows"],
                "train_users": stats["per_split"]["train"]["users"],
                "valid_users": stats["per_split"]["val"]["users"],
                "test_users": stats["per_split"]["test"]["users"],
                "timestamp_min": stats["combined"]["timestamp_min"],
                "timestamp_max": stats["combined"]["timestamp_max"],
            }
        )

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(paths["results_dir"] / "recbole_dataset_stats.csv", index=False)


def _prepare_carbon_predictions(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    amazon_output = paths["carbon_cache_dir"] / "amazon_pcf_predictions.csv"
    eval_output = paths["results_dir"] / "carbon" / "pcf_evaluation_predictions.csv"
    metrics_output = paths["results_dir"] / "carbon" / "pcf_evaluation_metrics.csv"
    metadata_output = paths["results_dir"] / "carbon" / "pcf_run_metadata.json"
    llm_cache = paths["carbon_cache_dir"] / "llm_prediction_cache.jsonl"
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    if not args.force_carbon and amazon_output.exists() and metrics_output.exists():
        log.info("Using cached carbon outputs in %s", paths["carbon_cache_dir"])
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "predict_carbon.py"),
        "--amazon-output",
        str(amazon_output),
        "--eval-output",
        str(eval_output),
        "--metrics-output",
        str(metrics_output),
        "--run-metadata-output",
        str(metadata_output),
        "--llm-cache",
        str(llm_cache),
        "--evaluation-limit",
        str(args.carbon_evaluation_limit),
        "--amazon-limit",
        str(args.amazon_limit),
        "--num-threads",
        str(args.carbon_num_threads),
    ]
    if args.skip_llm:
        cmd.append("--skip-llm")
    if args.llm_cache_only:
        cmd.append("--llm-cache-only")

    log.info("Preparing carbon predictions …")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _worker_mode(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    paths = _run_paths(run_dir)
    _ensure_run_dirs(paths)

    common_train_args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "01_train_recommender.py"),
        "--category",
        args.worker_category,
        "--model",
        args.worker_model,
        "--top-k",
        str(args.top_k_candidates),
        "--score-split",
        "test",
        "--user-batch-size",
        str(args.user_batch_size),
        "--interim-dir",
        str(args.interim_dir),
        "--recbole-dir",
        str(paths["recbole_dir"]),
        "--results-dir",
        str(paths["results_dir"]),
        "--model-dir",
        str(paths["model_dir"]),
    ]
    if args.force:
        common_train_args.extend(["--force-reformat", "--force-train"])
    if args.max_users is not None:
        common_train_args.extend(["--max-users", str(args.max_users)])
    if args.epochs is not None:
        common_train_args.extend(["--epochs", str(args.epochs)])
    if args.train_batch_size is not None:
        common_train_args.extend(["--train-batch-size", str(args.train_batch_size)])
    if args.eval_batch_size is not None:
        common_train_args.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.learning_rate is not None:
        common_train_args.extend(["--learning-rate", str(args.learning_rate)])
    if args.eval_step is not None:
        common_train_args.extend(["--eval-step", str(args.eval_step)])

    common_rerank_args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "02_rerank.py"),
        "--category",
        args.worker_category,
        "--model",
        args.worker_model,
        "--top-k",
        str(args.top_k_rerank),
        "--results-dir",
        str(paths["results_dir"]),
        "--interim-dir",
        str(args.interim_dir),
    ]
    common_eval_args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "03_evaluate.py"),
        "--category",
        args.worker_category,
        "--model",
        args.worker_model,
        "--results-dir",
        str(paths["results_dir"]),
        "--figures-dir",
        str(paths["figures_dir"]),
    ]

    for cmd in (common_train_args, common_rerank_args, common_eval_args):
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _launch_workers(args: argparse.Namespace, paths: dict[str, Path], max_parallel_jobs: int) -> None:
    jobs = list(itertools.product(args.categories, args.models))
    active: list[tuple[subprocess.Popen[str], tuple[str, str], Path, object, int | None]] = []

    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0
    gpu_slots = list(range(gpu_count)) if gpu_count > 1 else []
    free_slots = gpu_slots.copy()

    while jobs or active:
        while jobs and len(active) < max_parallel_jobs:
            category, model = jobs.pop(0)
            log_path = paths["logs_dir"] / f"{category}_{model}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()

            slot = None
            if free_slots:
                slot = free_slots.pop(0)
                env["CUDA_VISIBLE_DEVICES"] = str(slot)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--run-dir",
                str(paths["run_dir"]),
                "--interim-dir",
                str(args.interim_dir),
                "--worker-category",
                category,
                "--worker-model",
                model,
                "--top-k-candidates",
                str(args.top_k_candidates),
                "--top-k-rerank",
                str(args.top_k_rerank),
                "--user-batch-size",
                str(args.user_batch_size),
            ]
            if args.force:
                cmd.append("--force")
            if args.max_users is not None:
                cmd.extend(["--max-users", str(args.max_users)])
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.train_batch_size is not None:
                cmd.extend(["--train-batch-size", str(args.train_batch_size)])
            if args.eval_batch_size is not None:
                cmd.extend(["--eval-batch-size", str(args.eval_batch_size)])
            if args.learning_rate is not None:
                cmd.extend(["--learning-rate", str(args.learning_rate)])
            if args.eval_step is not None:
                cmd.extend(["--eval-step", str(args.eval_step)])

            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((process, (category, model), log_path, log_handle, slot))
            log.info("Launched %s/%s → %s", category, model, log_path)

        time.sleep(2)
        still_active: list[tuple[subprocess.Popen[str], tuple[str, str], Path, object, int | None]] = []
        for process, job, log_path, log_handle, slot in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, job, log_path, log_handle, slot))
                continue

            category, model = job
            log_handle.close()
            if return_code != 0:
                raise RuntimeError(
                    f"Worker failed for {category}/{model}. See log: {log_path}"
                )
            log.info("Completed %s/%s", category, model)
            if slot is not None:
                free_slots.append(slot)
                free_slots.sort()
        active = still_active


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the larger full-data experiment pipeline")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-category", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "run",
        help="Run workspace containing caches, results, figures, logs, and the notebook",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim",
        help="Directory containing interim train/val/test CSVs",
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--top-k-candidates", type=int, default=100)
    parser.add_argument("--top-k-rerank", type=int, default=10)
    parser.add_argument("--user-batch-size", type=int, default=256)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--eval-step",
        type=int,
        default=5,
        help="Validate every N epochs during RecBole training (default: 5)",
    )
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=None,
        help="Maximum concurrent model/category workers; defaults to available hardware",
    )
    parser.add_argument("--prepare-carbon", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore cached train/rerank/eval outputs")
    parser.add_argument("--force-carbon", action="store_true", help="Ignore cached carbon outputs")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-cache-only", action="store_true")
    parser.add_argument("--carbon-evaluation-limit", type=int, default=100)
    parser.add_argument("--amazon-limit", type=int, default=None)
    parser.add_argument("--carbon-num-threads", type=int, default=1)
    args = parser.parse_args()

    if args.worker:
        _worker_mode(args)
        return

    paths = _run_paths(args.run_dir.resolve())
    _ensure_run_dirs(paths)

    summary_df = _dataset_summary(args.interim_dir, args.categories, args.top_k_candidates)
    summary_path = paths["results_dir"] / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _prepare_recbole_cache(args, paths)

    run_manifest = {
        "run_dir": str(paths["run_dir"]),
        "categories": args.categories,
        "models": args.models,
        "top_k_candidates": args.top_k_candidates,
        "top_k_rerank": args.top_k_rerank,
        "user_batch_size": args.user_batch_size,
        "max_users": args.max_users,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "eval_step": args.eval_step,
        "prepare_carbon": args.prepare_carbon,
        "force": args.force,
    }
    manifest_path = paths["results_dir"] / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    if args.prepare_carbon:
        _prepare_carbon_predictions(args, paths)

    max_parallel_jobs = _default_parallelism(args.max_parallel_jobs)
    log.info("Using max_parallel_jobs=%d", max_parallel_jobs)
    _launch_workers(args, paths, max_parallel_jobs=max_parallel_jobs)

    plot_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "04_generate_paper_plots.py"),
        "--results-dir",
        str(paths["results_dir"]),
        "--figure-dir",
        str(paths["figures_dir"]),
        "--summary-output-dir",
        str(paths["results_dir"]),
        "--carbon-metrics-path",
        str(paths["results_dir"] / "carbon" / "pcf_evaluation_metrics.csv"),
        "--carbon-eval-predictions-path",
        str(paths["results_dir"] / "carbon" / "pcf_evaluation_predictions.csv"),
        "--amazon-predictions-path",
        str(paths["carbon_cache_dir"] / "amazon_pcf_predictions.csv"),
        "--categories",
        *args.categories,
        "--models",
        *args.models,
    ]
    subprocess.run(plot_cmd, cwd=PROJECT_ROOT, check=True)
    log.info("Full experiment run complete → %s", paths["run_dir"])


if __name__ == "__main__":
    main()
