#!/usr/bin/env python
"""
Minimal Colab-friendly wrapper around the full experiment runner.

This script keeps notebook cells thin by handling:
  - optional repo sync
  - optional dependency install + verification
  - automatic batch-size selection from GPU memory
  - mode dispatch (`auto`, `prepare`, `worker`, `finalize`, `status`)
  - concise failure context from shared run logs
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]
DEFAULT_MODELS = ["BPR", "NeuMF", "LightGCN"]
DEFAULT_REPO_URL = "https://github.com/andersvestrum/carbon-aware-recsys.git"


def _git_short_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sync_repo(project_root: Path, repo_url: str, repo_branch: str) -> None:
    project_root = project_root.resolve()
    if (project_root / ".git").exists():
        subprocess.run(
            ["git", "-C", str(project_root), "fetch", "origin", repo_branch],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(project_root), "checkout", repo_branch],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(project_root), "pull", "--ff-only", "origin", repo_branch],
            check=True,
        )
        return

    project_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            repo_branch,
            "--single-branch",
            repo_url,
            str(project_root),
        ],
        check=True,
    )


def install_requirements(project_root: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=project_root,
        check=True,
    )


def verify_runtime(project_root: Path) -> None:
    verify_code = (
        "import recbole, torch, pandas, pyarrow; "
        "from recbole.utils import get_model; "
        "import kmeans_pytorch; "
        "get_model('BPR'); "
        "print({"
        "'recbole': recbole.__version__, "
        "'torch': torch.__version__, "
        "'pandas': pandas.__version__, "
        "'pyarrow': pyarrow.__version__, "
        "'kmeans_pytorch': getattr(kmeans_pytorch, '__version__', 'installed'), "
        "'bpr_model_import': 'ok'"
        "})"
    )
    subprocess.run(
        [sys.executable, "-c", verify_code],
        cwd=project_root,
        check=True,
    )


def detect_batch_sizes() -> tuple[int, int, int, dict[str, object]]:
    gpu_name = "CPU"
    gpu_memory_gb = 0.0

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
    except Exception:
        pass

    if gpu_memory_gb >= 35:
        train_batch_size, eval_batch_size, user_batch_size = 8192, 16384, 1024
    elif gpu_memory_gb >= 20:
        train_batch_size, eval_batch_size, user_batch_size = 6144, 12288, 768
    elif gpu_memory_gb >= 14:
        train_batch_size, eval_batch_size, user_batch_size = 4096, 8192, 512
    else:
        train_batch_size, eval_batch_size, user_batch_size = 2048, 4096, 256

    gpu_info = {
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_memory_gb, 2),
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "user_batch_size": user_batch_size,
    }
    return train_batch_size, eval_batch_size, user_batch_size, gpu_info


def latest_failure_context(run_dir: Path) -> str:
    state_dir = run_dir / "results" / "_job_state"
    logs_dir = run_dir / "logs"
    chunks: list[str] = []

    if state_dir.exists():
        failed = sorted(state_dir.glob("*.failed.json"))
        if failed:
            chunks.append("Latest failed job state:")
            chunks.append(failed[-1].read_text(encoding="utf-8"))

    if logs_dir.exists():
        logs = sorted(logs_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
        if logs:
            latest_log = logs[-1]
            lines = latest_log.read_text(encoding="utf-8", errors="replace").splitlines()
            chunks.append(f"Tail of latest worker log: {latest_log}")
            chunks.append("\n".join(lines[-80:]))

    return "\n\n".join(chunk for chunk in chunks if chunk)


def print_status(run_dir: Path) -> None:
    state_dir = run_dir / "results" / "_job_state"
    results_dir = run_dir / "results"
    status_summary = {"done": 0, "failed": 0, "running": 0}

    if state_dir.exists():
        status_summary["done"] = len(list(state_dir.glob("*.done.json")))
        status_summary["failed"] = len(list(state_dir.glob("*.failed.json")))
        status_summary["running"] = len(list(state_dir.glob("*.running.json")))

    print({"run_dir": str(run_dir), **status_summary})

    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        print("\nrun_manifest.json:")
        print(manifest_path.read_text(encoding="utf-8"))

    context = latest_failure_context(run_dir)
    if context:
        print(f"\n{context}")


def run_step(name: str, cmd: list[str], project_root: Path, run_dir: Path) -> None:
    print(f"Running {name}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError:
        context = latest_failure_context(run_dir)
        if context:
            print(f"\n{context}")
        raise


def build_base_cmd(
    args: argparse.Namespace,
    *,
    project_root: Path,
    run_dir: Path,
    interim_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
    user_batch_size: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/05_run_full_experiment.py",
        "--run-dir",
        str(run_dir),
        "--interim-dir",
        str(interim_dir),
        "--top-k-candidates",
        str(args.top_k_candidates),
        "--top-k-rerank",
        str(args.top_k_rerank),
        "--user-batch-size",
        str(args.user_batch_size or user_batch_size),
        "--train-batch-size",
        str(args.train_batch_size or train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size or eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--epochs",
        str(args.epochs),
        "--eval-step",
        str(args.eval_step),
        "--categories",
        *args.categories,
        "--models",
        *args.models,
    ]

    if args.max_users is not None:
        cmd.extend(["--max-users", str(args.max_users)])
    if args.prepare_carbon:
        cmd.append("--prepare-carbon")
    if args.force:
        cmd.append("--force")
    if args.force_carbon:
        cmd.append("--force-carbon")
    if args.skip_llm:
        cmd.append("--skip-llm")
    if args.llm_cache_only:
        cmd.append("--llm-cache-only")
    if args.retry_failed_jobs:
        cmd.append("--retry-failed-jobs")

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin Colab session wrapper")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repo checkout used for the Colab session",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Remote repository URL used if repo sync is requested",
    )
    parser.add_argument(
        "--repo-branch",
        default=os.environ.get("REPO_BRANCH", "6-neumfs-sharp-curve"),
        help="Git branch to sync before running",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "prepare", "worker", "finalize", "status"],
        default="auto",
        help="Session mode",
    )
    parser.add_argument("--sync-repo", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--interim-dir", type=Path, default=None)
    parser.add_argument("--worker-name", default=None)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--top-k-candidates", type=int, default=100)
    parser.add_argument("--top-k-rerank", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-step", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--user-batch-size", type=int, default=None)
    parser.add_argument("--prepare-carbon", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-carbon", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-cache-only", action="store_true")
    parser.add_argument("--retry-failed-jobs", action="store_true")
    parser.add_argument("--finalize-when-done", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    if args.sync_repo:
        sync_repo(project_root, args.repo_url, args.repo_branch)

    if not (project_root / ".git").exists():
        raise FileNotFoundError(f"Expected a git checkout at {project_root}")

    run_dir = (args.run_dir or (project_root / "run")).expanduser().resolve()
    interim_dir = (args.interim_dir or (project_root / "data" / "interim")).expanduser().resolve()

    print(
        {
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "interim_dir": str(interim_dir),
            "repo_branch": args.repo_branch,
            "git_commit": _git_short_head(project_root),
            "mode": args.mode,
        }
    )

    if args.mode == "status":
        print_status(run_dir)
        return

    if args.install:
        install_requirements(project_root)
    if args.verify_runtime:
        verify_runtime(project_root)

    train_batch_size, eval_batch_size, user_batch_size, gpu_info = detect_batch_sizes()
    print(gpu_info)

    worker_name = args.worker_name or f"colab-{socket.gethostname()}-{int(time.time())}"
    base_cmd = build_base_cmd(
        args,
        project_root=project_root,
        run_dir=run_dir,
        interim_dir=interim_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        user_batch_size=user_batch_size,
    )

    if args.mode == "auto":
        run_step("prepare", base_cmd + ["--prepare-only"], project_root, run_dir)
        worker_cmd = base_cmd + [
            "--claim-jobs",
            "--skip-cache-prepare",
            "--worker-name",
            worker_name,
            "--finalize-when-done",
        ]
        run_step("worker", worker_cmd, project_root, run_dir)
        return

    if args.mode == "prepare":
        run_step("prepare", base_cmd + ["--prepare-only"], project_root, run_dir)
        return

    if args.mode == "worker":
        worker_cmd = base_cmd + [
            "--claim-jobs",
            "--skip-cache-prepare",
            "--worker-name",
            worker_name,
        ]
        if args.finalize_when_done:
            worker_cmd.append("--finalize-when-done")
        run_step("worker", worker_cmd, project_root, run_dir)
        return

    if args.mode == "finalize":
        run_step("finalize", base_cmd + ["--plots-only"], project_root, run_dir)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
