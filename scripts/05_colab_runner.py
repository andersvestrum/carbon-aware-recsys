#!/usr/bin/env python
"""
Google Colab runner for the carbon-aware recommendation pipeline.

This script keeps Google Drive as the persistent source of truth while using
local Colab scratch space under ``/content`` for the heavy I/O parts of each
job. It is intentionally thin around the existing stepwise pipeline:

    1. scripts/01_train_recommender.py
    2. scripts/02_rerank.py
    3. scripts/03_evaluate.py
    4. scripts/04_generate_paper_plots.py

Subcommands:
    prepare   Bootstrap Drive-side caches and create the shared job manifest.
    worker    Claim and execute jobs from the shared Drive-backed queue.
    status    Inspect queue state and recent failures.
    finalize  Wait for workers to finish, then generate paper plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import SUPPORTED_MODELS, canonical_model_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]
DEFAULT_MODELS = ["BPR", "NeuMF", "LightGCN"]
JOB_POLL_SECONDS = 30
HEARTBEAT_INTERVAL_SECONDS = 60
STALE_TIMEOUT_SECONDS = 15 * 60
PREPROCESS_CATEGORY_MAP = {
    "electronics": "electronics",
    "home_and_kitchen": "hak",
    "sports_and_outdoors": "sao",
}


@dataclass(frozen=True)
class Layout:
    repo_root: Path
    drive_root: Path
    scratch_root: Path
    worker_name: str
    drive_interim_dir: Path
    drive_carbon_dir: Path
    drive_carbon_results_dir: Path
    drive_results_dir: Path
    drive_figures_dir: Path
    drive_logs_dir: Path
    drive_checkpoints_dir: Path
    jobs_dir: Path
    manifest_path: Path
    worker_root: Path
    scratch_repo_dir: Path
    scratch_data_dir: Path
    scratch_interim_dir: Path
    scratch_processed_dir: Path
    scratch_recbole_dir: Path
    scratch_carbon_dir: Path
    scratch_results_dir: Path
    scratch_figures_dir: Path
    scratch_tmp_dir: Path


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def build_layout(args: argparse.Namespace, *, worker_name: str | None = None) -> Layout:
    resolved_worker_name = worker_name or args.worker_name or "primary"
    drive_root = _resolve(args.drive_root)
    repo_root = _resolve(args.repo_root)
    scratch_root = _resolve(args.scratch_root)
    worker_root = scratch_root / resolved_worker_name
    return Layout(
        repo_root=repo_root,
        drive_root=drive_root,
        scratch_root=scratch_root,
        worker_name=resolved_worker_name,
        drive_interim_dir=drive_root / "data" / "interim",
        drive_carbon_dir=drive_root / "data" / "processed" / "carbon",
        drive_carbon_results_dir=drive_root / "results" / "carbon",
        drive_results_dir=drive_root / "results",
        drive_figures_dir=drive_root / "figures",
        drive_logs_dir=drive_root / "logs",
        drive_checkpoints_dir=drive_root / "checkpoints",
        jobs_dir=drive_root / "state" / "jobs",
        manifest_path=drive_root / "state" / "jobs" / "manifest.json",
        worker_root=worker_root,
        scratch_repo_dir=worker_root / "repo",
        scratch_data_dir=worker_root / "data",
        scratch_interim_dir=worker_root / "data" / "interim",
        scratch_processed_dir=worker_root / "data" / "processed",
        scratch_recbole_dir=worker_root / "data" / "processed" / "recbole",
        scratch_carbon_dir=worker_root / "data" / "processed" / "carbon",
        scratch_results_dir=worker_root / "tmp_results",
        scratch_figures_dir=worker_root / "tmp_figures",
        scratch_tmp_dir=worker_root / "tmp",
    )


def _job_slug(category: str, model: str) -> str:
    return f"{category}__{model}"


def _job_paths(layout: Layout, category: str, model: str) -> dict[str, Path]:
    slug = _job_slug(category, model)
    return {
        "running": layout.jobs_dir / f"{slug}.running.json",
        "done": layout.jobs_dir / f"{slug}.done.json",
        "failed": layout.jobs_dir / f"{slug}.failed.json",
        "log": layout.drive_logs_dir / f"{category}_{model}.log",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return True


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.name}.tmp-{uuid.uuid4().hex}")
    shutil.copy2(src, tmp_path)
    os.replace(tmp_path, dst)


def _copy_if_missing(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    _copy_atomic(src, dst)
    return True


def _repo_ignore(_src: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".ipynb_checkpoints",
            "data",
            "output",
            "log_tensorboard",
        }:
            ignored.add(name)
    return ignored


def _sync_repo_to_scratch(layout: Layout) -> None:
    if layout.scratch_repo_dir.exists():
        shutil.rmtree(layout.scratch_repo_dir)
    layout.scratch_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        layout.repo_root,
        layout.scratch_repo_dir,
        ignore=_repo_ignore,
    )
    log.info("Synced repo → %s", layout.scratch_repo_dir)


def _reset_job_scratch(layout: Layout) -> None:
    for path in [layout.scratch_results_dir, layout.scratch_figures_dir, layout.scratch_tmp_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _ensure_drive_dirs(layout: Layout) -> None:
    for path in [
        layout.drive_root,
        layout.drive_interim_dir,
        layout.drive_carbon_dir,
        layout.drive_carbon_results_dir,
        layout.drive_results_dir,
        layout.drive_figures_dir,
        layout.drive_logs_dir,
        layout.drive_checkpoints_dir,
        layout.jobs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _run_preprocess_for_interim(layout: Layout, categories: list[str]) -> None:
    preprocess_categories = [PREPROCESS_CATEGORY_MAP[category] for category in categories]
    cmd = [
        sys.executable,
        "-m",
        "src.data.preprocess",
        "--category",
        *preprocess_categories,
    ]
    log.info("Generating missing interim data in repo checkout …")
    subprocess.run(cmd, cwd=layout.repo_root, check=True)


def _ensure_interim_on_drive(layout: Layout, categories: list[str]) -> None:
    def _copy_missing_to_drive() -> list[Path]:
        missing_paths: list[Path] = []
        for split in ["train", "val", "test"]:
            for category in categories:
                drive_path = layout.drive_interim_dir / split / f"{category}.csv"
                if drive_path.exists():
                    continue
                repo_path = layout.repo_root / "data" / "interim" / split / f"{category}.csv"
                if _copy_if_missing(repo_path, drive_path):
                    log.info("Bootstrapped interim split → %s", drive_path)
                    continue
                missing_paths.append(drive_path)
        return missing_paths

    missing = _copy_missing_to_drive()
    if missing:
        try:
            _run_preprocess_for_interim(layout, categories)
        except subprocess.CalledProcessError as exc:
            log.warning("Automatic interim preprocessing failed: %r", exc)
        missing = _copy_missing_to_drive()

    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required interim data on Drive. "
            "The runner also tried to build `repo_root/data/interim` automatically, "
            "but the required files are still missing. Populate the following files "
            "on Drive or make sure preprocessing can run successfully:\n"
            f"{joined}"
        )


def _ensure_carbon_outputs(layout: Layout) -> None:
    amazon_predictions = layout.drive_carbon_dir / "amazon_pcf_predictions.csv"
    repo_amazon_predictions = layout.repo_root / "data" / "processed" / "carbon" / "amazon_pcf_predictions.csv"
    legacy_amazon_predictions = layout.repo_root / "data" / "processed" / "carbon" / "amazon_carbon_estimates.csv"
    eval_predictions = layout.drive_carbon_results_dir / "pcf_evaluation_predictions.csv"
    repo_eval_predictions = layout.repo_root / "output" / "results" / "carbon" / "pcf_evaluation_predictions.csv"
    metrics_output = layout.drive_carbon_results_dir / "pcf_evaluation_metrics.csv"
    repo_metrics_output = layout.repo_root / "output" / "results" / "carbon" / "pcf_evaluation_metrics.csv"
    metadata_output = layout.drive_carbon_results_dir / "pcf_run_metadata.json"
    repo_metadata_output = layout.repo_root / "output" / "results" / "carbon" / "pcf_run_metadata.json"
    llm_cache = layout.drive_carbon_dir / "llm_prediction_cache.jsonl"
    repo_llm_cache = layout.repo_root / "data" / "processed" / "carbon" / "llm_prediction_cache.jsonl"

    _copy_if_missing(repo_amazon_predictions, amazon_predictions)
    if not amazon_predictions.exists():
        _copy_if_missing(legacy_amazon_predictions, amazon_predictions)
    _copy_if_missing(repo_eval_predictions, eval_predictions)
    _copy_if_missing(repo_metrics_output, metrics_output)
    _copy_if_missing(repo_metadata_output, metadata_output)
    _copy_if_missing(repo_llm_cache, llm_cache)

    missing_any = [
        path
        for path in [amazon_predictions, eval_predictions, metrics_output, metadata_output]
        if not path.exists()
    ]
    if not missing_any:
        return

    raw_amazon_dir = layout.repo_root / "data" / "raw" / "amazon"
    raw_carbon_catalogue = layout.repo_root / "data" / "raw" / "carbon_catalogue" / "carbon_catalogue.csv"
    raw_amazon_available = raw_amazon_dir.exists() and any(
        path.is_file() and not path.name.startswith(".")
        for path in raw_amazon_dir.rglob("*")
    )
    if not raw_carbon_catalogue.exists() or not raw_amazon_available:
        log.warning(
            "Carbon outputs are incomplete on Drive and raw inputs are unavailable under %s. "
            "Continuing without regenerating missing carbon evaluation files.",
            layout.repo_root / "data" / "raw",
        )
        return

    cmd = [
        sys.executable,
        str(layout.repo_root / "scripts" / "predict_carbon.py"),
        "--device",
        "cpu",
        "--num-threads",
        "8",
        "--skip-llm",
        "--amazon-output",
        str(amazon_predictions),
        "--eval-output",
        str(eval_predictions),
        "--metrics-output",
        str(metrics_output),
        "--run-metadata-output",
        str(metadata_output),
        "--llm-cache",
        str(llm_cache),
    ]
    log.info("Preparing missing carbon outputs …")
    try:
        subprocess.run(cmd, cwd=layout.repo_root, check=True)
    except subprocess.CalledProcessError as exc:
        log.warning(
            "Could not regenerate missing carbon outputs (%r). "
            "Continuing with whatever carbon files are already available on Drive.",
            exc,
        )


def _write_manifest(args: argparse.Namespace, layout: Layout) -> dict[str, Any]:
    jobs = [
        {
            "category": category,
            "model": model,
            "slug": _job_slug(category, model),
        }
        for category in args.categories
        for model in args.models
    ]
    manifest = {
        "created_at": time.time(),
        "repo_root": str(layout.repo_root),
        "drive_root": str(layout.drive_root),
        "categories": args.categories,
        "models": args.models,
        "jobs": jobs,
        "top_k_candidates": args.top_k_candidates,
        "top_k_rerank": args.top_k_rerank,
        "user_batch_size": args.user_batch_size,
        "max_users": args.max_users,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "eval_step": args.eval_step,
    }
    _write_json(layout.manifest_path, manifest)
    return manifest


def _copy_category_inputs_to_scratch(layout: Layout, category: str) -> None:
    for split in ["train", "val", "test"]:
        src = layout.drive_interim_dir / split / f"{category}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Missing Drive interim file: {src}")
        dst = layout.scratch_interim_dir / split / f"{category}.csv"
        _copy_atomic(src, dst)

    if layout.scratch_carbon_dir.exists():
        shutil.rmtree(layout.scratch_carbon_dir)
    layout.scratch_carbon_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(layout.drive_carbon_dir.glob("*")):
        if src.is_file():
            _copy_atomic(src, layout.scratch_carbon_dir / src.name)


def _job_output_files(layout: Layout, category: str, model: str) -> tuple[list[Path], list[Path]]:
    result_patterns = [
        f"{category}_{model}_scores.parquet",
        f"{category}_{model}_eval.json",
        f"{category}_{model}_reranking_metrics.json",
        f"{category}_{model}_evaluation_summary.csv",
        f"{category}_{model}_pareto.json",
        f"{category}_{model}_reranked_*.parquet",
    ]
    figure_patterns = [
        f"{category}_{model}_tradeoff.png",
        f"{category}_{model}_lambda_sensitivity.png",
    ]

    result_files: list[Path] = []
    for pattern in result_patterns:
        result_files.extend(sorted(layout.scratch_results_dir.glob(pattern)))

    figure_files: list[Path] = []
    for pattern in figure_patterns:
        figure_files.extend(sorted(layout.scratch_figures_dir.glob(pattern)))

    return result_files, figure_files


def _sync_job_outputs_to_drive(layout: Layout, category: str, model: str) -> None:
    result_files, figure_files = _job_output_files(layout, category, model)
    for src in result_files:
        _copy_atomic(src, layout.drive_results_dir / src.name)
    for src in figure_files:
        _copy_atomic(src, layout.drive_figures_dir / src.name)


def _heartbeat_thread(
    layout: Layout,
    category: str,
    model: str,
    stop_event: threading.Event,
    *,
    heartbeat_interval: int,
    started_at: float,
) -> threading.Thread:
    state_path = _job_paths(layout, category, model)["running"]

    def _beat() -> None:
        while not stop_event.wait(heartbeat_interval):
            payload = {
                "category": category,
                "model": model,
                "worker_name": layout.worker_name,
                "started_at": started_at,
                "heartbeat_at": time.time(),
            }
            _write_json(state_path, payload)

    thread = threading.Thread(target=_beat, name=f"heartbeat-{category}-{model}", daemon=True)
    thread.start()
    return thread


def _running_state_is_stale(state_path: Path, *, stale_timeout: int) -> bool:
    try:
        payload = _read_json(state_path)
    except Exception:
        return True

    last_seen = float(payload.get("heartbeat_at") or payload.get("started_at") or 0.0)
    return (time.time() - last_seen) > stale_timeout


def _claim_next_job(
    layout: Layout,
    manifest: dict[str, Any],
    *,
    stale_timeout: int,
) -> tuple[str, str] | None:
    for job in manifest["jobs"]:
        category = job["category"]
        model = job["model"]
        paths = _job_paths(layout, category, model)
        if paths["done"].exists() or paths["failed"].exists():
            continue

        if paths["running"].exists():
            if not _running_state_is_stale(paths["running"], stale_timeout=stale_timeout):
                continue
            try:
                stale_payload = _read_json(paths["running"])
            except Exception:
                stale_payload = {}
            log.warning(
                "Reclaiming stale job %s/%s from worker=%s",
                category,
                model,
                stale_payload.get("worker_name", "<unknown>"),
            )
            paths["running"].unlink(missing_ok=True)

        started_at = time.time()
        payload = {
            "category": category,
            "model": model,
            "worker_name": layout.worker_name,
            "started_at": started_at,
            "heartbeat_at": started_at,
        }
        if _atomic_create_json(paths["running"], payload):
            return category, model
    return None


def _mark_job_done(layout: Layout, category: str, model: str) -> None:
    paths = _job_paths(layout, category, model)
    paths["running"].unlink(missing_ok=True)
    _write_json(
        paths["done"],
        {
            "category": category,
            "model": model,
            "worker_name": layout.worker_name,
            "completed_at": time.time(),
        },
    )


def _mark_job_failed(layout: Layout, category: str, model: str, message: str) -> None:
    paths = _job_paths(layout, category, model)
    paths["running"].unlink(missing_ok=True)
    _write_json(
        paths["failed"],
        {
            "category": category,
            "model": model,
            "worker_name": layout.worker_name,
            "failed_at": time.time(),
            "message": message,
        },
    )


def _queue_counts(layout: Layout, manifest: dict[str, Any]) -> dict[str, int]:
    counts = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    for job in manifest["jobs"]:
        paths = _job_paths(layout, job["category"], job["model"])
        if paths["done"].exists():
            counts["done"] += 1
        elif paths["failed"].exists():
            counts["failed"] += 1
        elif paths["running"].exists():
            counts["running"] += 1
        else:
            counts["pending"] += 1
    return counts


def _latest_failed_state(layout: Layout) -> dict[str, Any] | None:
    failed_files = sorted(layout.jobs_dir.glob("*.failed.json"), key=lambda p: p.stat().st_mtime)
    if not failed_files:
        return None
    return _read_json(failed_files[-1])


def _latest_log_tail(layout: Layout, *, max_lines: int = 60) -> tuple[Path | None, list[str]]:
    log_files = sorted(layout.drive_logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        return None, []
    latest = log_files[-1]
    lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
    return latest, lines[-max_lines:]


def _base_env(layout: Layout) -> dict[str, str]:
    layout.scratch_tmp_dir.mkdir(parents=True, exist_ok=True)
    (layout.scratch_tmp_dir / "mpl").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = str(layout.scratch_tmp_dir / "mpl")
    env["TMPDIR"] = str(layout.scratch_tmp_dir)
    return env


def _run_logged_command(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(cmd)} ===\n")
        handle.flush()
        subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _train_cmd(args: argparse.Namespace, layout: Layout, category: str, model: str) -> list[str]:
    cmd = [
        sys.executable,
        str(layout.scratch_repo_dir / "scripts" / "01_train_recommender.py"),
        "--category",
        category,
        "--model",
        model,
        "--top-k",
        str(args.top_k_candidates),
        "--score-split",
        "test",
        "--user-batch-size",
        str(args.user_batch_size),
        "--interim-dir",
        str(layout.scratch_interim_dir),
        "--recbole-dir",
        str(layout.scratch_recbole_dir),
        "--results-dir",
        str(layout.scratch_results_dir),
        "--model-dir",
        str(layout.drive_checkpoints_dir),
    ]
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
    return cmd


def _rerank_cmd(args: argparse.Namespace, layout: Layout, category: str, model: str) -> list[str]:
    return [
        sys.executable,
        str(layout.scratch_repo_dir / "scripts" / "02_rerank.py"),
        "--category",
        category,
        "--model",
        model,
        "--top-k",
        str(args.top_k_rerank),
        "--results-dir",
        str(layout.scratch_results_dir),
        "--interim-dir",
        str(layout.scratch_interim_dir),
    ]


def _evaluate_cmd(layout: Layout, category: str, model: str) -> list[str]:
    return [
        sys.executable,
        str(layout.scratch_repo_dir / "scripts" / "03_evaluate.py"),
        "--category",
        category,
        "--model",
        model,
        "--results-dir",
        str(layout.scratch_results_dir),
        "--figures-dir",
        str(layout.scratch_figures_dir),
    ]


def _run_job(args: argparse.Namespace, layout: Layout, category: str, model: str) -> None:
    _reset_job_scratch(layout)
    _copy_category_inputs_to_scratch(layout, category)

    log_path = _job_paths(layout, category, model)["log"]
    env = _base_env(layout)
    train_cmd = _train_cmd(args, layout, category, model)
    rerank_cmd = _rerank_cmd(args, layout, category, model)
    evaluate_cmd = _evaluate_cmd(layout, category, model)

    _run_logged_command(train_cmd, cwd=layout.scratch_repo_dir, log_path=log_path, env=env)
    _run_logged_command(rerank_cmd, cwd=layout.scratch_repo_dir, log_path=log_path, env=env)
    _run_logged_command(evaluate_cmd, cwd=layout.scratch_repo_dir, log_path=log_path, env=env)
    _sync_job_outputs_to_drive(layout, category, model)


def _wait_for_queue(layout: Layout, manifest: dict[str, Any], *, poll_seconds: int) -> dict[str, int]:
    while True:
        counts = _queue_counts(layout, manifest)
        if counts["running"] == 0:
            return counts
        log.info("Waiting for active workers to finish: %s", counts)
        time.sleep(poll_seconds)


def cmd_prepare(args: argparse.Namespace) -> int:
    layout = build_layout(args)
    _ensure_drive_dirs(layout)
    _ensure_interim_on_drive(layout, args.categories)
    _ensure_carbon_outputs(layout)
    manifest = _write_manifest(args, layout)
    log.info("Prepared Drive workspace at %s", layout.drive_root)
    log.info("Job manifest → %s (%d jobs)", layout.manifest_path, len(manifest["jobs"]))
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    layout = build_layout(
        args,
        worker_name=args.worker_name or f"{socket.gethostname()}-{os.getpid()}",
    )
    _ensure_drive_dirs(layout)
    if not layout.manifest_path.exists():
        raise FileNotFoundError(
            f"Missing job manifest: {layout.manifest_path}. Run `prepare` first."
        )
    manifest = _read_json(layout.manifest_path)
    _sync_repo_to_scratch(layout)

    while True:
        job = _claim_next_job(layout, manifest, stale_timeout=args.stale_timeout)
        if job is None:
            counts = _queue_counts(layout, manifest)
            if counts["running"] == 0:
                log.info("No more claimable jobs. Final queue state: %s", counts)
                return 0
            log.info("No claimable jobs yet. Queue state: %s", counts)
            time.sleep(args.poll_seconds)
            continue

        category, model = job
        started_at = time.time()
        stop_event = threading.Event()
        heartbeat = _heartbeat_thread(
            layout,
            category,
            model,
            stop_event,
            heartbeat_interval=args.heartbeat_interval,
            started_at=started_at,
        )
        try:
            log.info("[%s] Running %s/%s", layout.worker_name, category, model)
            _run_job(args, layout, category, model)
        except Exception as exc:
            stop_event.set()
            heartbeat.join(timeout=5)
            _mark_job_failed(layout, category, model, repr(exc))
            log.exception("Job failed for %s/%s", category, model)
            continue
        else:
            stop_event.set()
            heartbeat.join(timeout=5)
            _mark_job_done(layout, category, model)
            log.info("[%s] Completed %s/%s", layout.worker_name, category, model)


def cmd_status(args: argparse.Namespace) -> int:
    layout = build_layout(args)
    info = {
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(layout.repo_root),
        "drive_root": str(layout.drive_root),
        "scratch_root": str(layout.scratch_root),
        "repo_branch": _git_branch(layout.repo_root),
        "git_commit": _git_commit(layout.repo_root),
        "mode": "status",
    }
    print(info)

    if not layout.manifest_path.exists():
        print({"manifest": "missing", "path": str(layout.manifest_path)})
        return 0

    manifest = _read_json(layout.manifest_path)
    counts = _queue_counts(layout, manifest)
    print({"drive_root": str(layout.drive_root), **counts})
    print("\nrun_manifest.json:")
    print(json.dumps(manifest, indent=2))

    failed = _latest_failed_state(layout)
    if failed is not None:
        print("\nLatest failed job state:\n")
        print(json.dumps(failed, indent=2))

    latest_log, lines = _latest_log_tail(layout)
    if latest_log is not None:
        print(f"\nTail of latest worker log: {latest_log}\n")
        print("\n".join(lines))
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    layout = build_layout(args, worker_name=args.worker_name or "finalize")
    _ensure_drive_dirs(layout)
    if not layout.manifest_path.exists():
        raise FileNotFoundError(
            f"Missing job manifest: {layout.manifest_path}. Run `prepare` first."
        )

    manifest = _read_json(layout.manifest_path)
    counts = _wait_for_queue(layout, manifest, poll_seconds=args.poll_seconds)
    if counts["failed"] > 0 or counts["pending"] > 0:
        raise SystemExit(
            "Cannot finalize because the queue is not fully successful: "
            f"{counts}"
        )

    _sync_repo_to_scratch(layout)
    env = _base_env(layout)
    cmd = [
        sys.executable,
        str(layout.scratch_repo_dir / "scripts" / "04_generate_paper_plots.py"),
        "--results-dir",
        str(layout.drive_results_dir),
        "--figure-dir",
        str(layout.drive_figures_dir),
        "--summary-output-dir",
        str(layout.drive_results_dir),
        "--carbon-metrics-path",
        str(layout.drive_carbon_results_dir / "pcf_evaluation_metrics.csv"),
        "--carbon-eval-predictions-path",
        str(layout.drive_carbon_results_dir / "pcf_evaluation_predictions.csv"),
        "--amazon-predictions-path",
        str(layout.drive_carbon_dir / "amazon_pcf_predictions.csv"),
        "--categories",
        *args.categories,
        "--models",
        *args.models,
    ]
    log_path = layout.drive_logs_dir / "finalize.log"
    _run_logged_command(cmd, cwd=layout.scratch_repo_dir, log_path=log_path, env=env)
    log.info("Finalize complete. Figures → %s", layout.drive_figures_dir)
    return 0


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _git_branch(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--drive-root", type=Path, required=True, help="Drive-backed workspace root.")
    parent.add_argument("--scratch-root", type=Path, required=True, help="Local scratch root, e.g. /content/carecs.")
    parent.add_argument("--repo-root", type=Path, required=True, help="Repo checkout to copy code from.")
    parent.add_argument(
        "--categories",
        nargs="+",
        choices=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
        help="Categories to include.",
    )
    parent.add_argument(
        "--models",
        nargs="+",
        type=canonical_model_name,
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODELS,
        help="Models to include.",
    )
    parent.add_argument("--top-k-candidates", type=int, default=100)
    parent.add_argument("--top-k-rerank", type=int, default=10)
    parent.add_argument("--user-batch-size", type=int, default=1024)
    parent.add_argument("--max-users", type=int, default=None)
    parent.add_argument("--epochs", type=int, default=50)
    parent.add_argument("--train-batch-size", type=int, default=8192)
    parent.add_argument("--eval-batch-size", type=int, default=16384)
    parent.add_argument("--learning-rate", type=float, default=1e-3)
    parent.add_argument("--eval-step", type=int, default=10)
    parent.add_argument("--worker-name", type=str, default=None)
    parent.add_argument("--heartbeat-interval", type=int, default=HEARTBEAT_INTERVAL_SECONDS)
    parent.add_argument("--stale-timeout", type=int, default=STALE_TIMEOUT_SECONDS)
    parent.add_argument("--poll-seconds", type=int, default=JOB_POLL_SECONDS)

    parser = argparse.ArgumentParser(description="Drive-backed Colab runner for the recommendation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", parents=[parent], help="Bootstrap Drive workspace and job manifest")
    subparsers.add_parser("worker", parents=[parent], help="Claim and run jobs from the shared queue")
    subparsers.add_parser("status", parents=[parent], help="Inspect shared queue state")
    subparsers.add_parser("finalize", parents=[parent], help="Generate paper plots after workers finish")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "worker":
        return cmd_worker(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "finalize":
        return cmd_finalize(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
