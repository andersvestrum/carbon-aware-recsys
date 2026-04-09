#!/usr/bin/env python
"""
Monitor the shared Colab runner queue and estimate remaining runtime.

This script is intentionally read-only. It inspects the Drive-backed job state
written by ``scripts/05_colab_runner.py`` plus the worker log files, then prints
an ETA-oriented snapshot or a live-updating watch view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_REFRESH_SECONDS = 60
DEFAULT_TAIL_LINES = 12
LAMBDA_SWEEP_SIZE = 16
TRAIN_WEIGHT = 0.70
RERANK_WEIGHT = 0.25
EVALUATE_WEIGHT = 0.05

EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+(training|evaluating)", re.IGNORECASE)
LAMBDA_RE = re.compile(r"[λ\\]?\s*=?\s*([01](?:\.\d+)?)")


@dataclass
class JobSnapshot:
    category: str
    model: str
    slug: str
    state: str
    worker_name: str | None
    started_at: float | None
    heartbeat_at: float | None
    finished_at: float | None
    duration_seconds: float | None
    elapsed_seconds: float | None
    progress: float
    stage: str
    eta_seconds: float | None
    log_path: Path
    log_summary: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _job_paths(drive_root: Path, category: str, model: str) -> dict[str, Path]:
    jobs_dir = drive_root / "state" / "jobs"
    slug = f"{category}__{model}"
    return {
        "slug": slug,
        "running": jobs_dir / f"{slug}.running.json",
        "done": jobs_dir / f"{slug}.done.json",
        "failed": jobs_dir / f"{slug}.failed.json",
        "log": drive_root / "logs" / f"{category}_{model}.log",
    }


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds):
        return "?"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _latest_lines(path: Path, *, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]


def _detect_stage(lines: list[str]) -> tuple[str, float, str]:
    if not lines:
        return "pending", 0.0, "No log yet"

    last_command = ""
    for line in reversed(lines):
        if line.startswith("===") and "scripts/" in line:
            last_command = line
            break

    if "scripts/03_evaluate.py" in last_command:
        for line in reversed(lines):
            if "Evaluation complete" in line:
                return "evaluate", 1.0, "Evaluation complete"
            if "Evaluating:" in line or "Summary" in line or "Saved evaluation summary" in line:
                return "evaluate", TRAIN_WEIGHT + RERANK_WEIGHT + 0.75 * EVALUATE_WEIGHT, line.strip()
        return "evaluate", TRAIN_WEIGHT + RERANK_WEIGHT + 0.50 * EVALUATE_WEIGHT, "Evaluating outputs"

    if "scripts/02_rerank.py" in last_command:
        lambda_values: list[str] = []
        for line in lines:
            if "λ=" not in line:
                continue
            match = re.search(r"λ=([0-9.]+)", line)
            if match:
                lambda_values.append(match.group(1))
        unique_count = len(dict.fromkeys(lambda_values))
        fraction = min(1.0, unique_count / LAMBDA_SWEEP_SIZE) if unique_count else 0.0
        summary = f"Rerank λ sweep {unique_count}/{LAMBDA_SWEEP_SIZE}"
        for line in reversed(lines):
            if "λ=" in line:
                summary = line.strip()
                break
        return "rerank", TRAIN_WEIGHT + RERANK_WEIGHT * fraction, summary

    if "scripts/01_train_recommender.py" in last_command:
        for line in reversed(lines):
            match = EPOCH_RE.search(line)
            if match:
                epoch_number = int(match.group(1))
                summary = line.strip()
                return "train", min(0.99, 0.05 + TRAIN_WEIGHT * (epoch_number + 1) / 50.0), summary
        for line in reversed(lines):
            if "Training " in line or "Initialising model" in line or "Preparing train / val / test data" in line:
                return "train", 0.05, line.strip()
        return "train", 0.02, "Preparing training job"

    for line in reversed(lines):
        if "Evaluation complete" in line:
            return "evaluate", 1.0, line.strip()
        if "Re-ranking complete" in line:
            return "rerank", TRAIN_WEIGHT + RERANK_WEIGHT, line.strip()
        if "Candidate generation complete" in line:
            return "train", TRAIN_WEIGHT, line.strip()

    return "running", 0.02, lines[-1].strip()


def _progress_from_log(log_path: Path, *, epochs: int, tail_lines: int) -> tuple[str, float, str]:
    lines = _latest_lines(log_path, max_lines=max(120, tail_lines))
    stage, rough_progress, summary = _detect_stage(lines)
    if stage != "train":
        return stage, rough_progress, summary

    # Refine epoch progress using the full configured epoch count when available.
    for line in reversed(lines):
        match = EPOCH_RE.search(line)
        if match:
            epoch_number = int(match.group(1))
            refined = min(0.99, 0.05 + TRAIN_WEIGHT * (epoch_number + 1) / max(1, epochs))
            return stage, refined, line.strip()
    return stage, rough_progress, summary


def _build_snapshot(
    drive_root: Path,
    job: dict[str, Any],
    *,
    now: float,
    epochs: int,
    tail_lines: int,
    mean_done_seconds: float | None,
) -> JobSnapshot:
    category = job["category"]
    model = job["model"]
    paths = _job_paths(drive_root, category, model)
    running_path = paths["running"]
    done_path = paths["done"]
    failed_path = paths["failed"]
    log_path = paths["log"]

    state = "pending"
    payload: dict[str, Any] = {}
    finished_at = None
    duration_seconds = None
    started_at = None
    heartbeat_at = None
    elapsed_seconds = None
    progress = 0.0
    stage = "pending"
    summary = "Pending"

    if done_path.exists():
        state = "done"
        payload = _read_json(done_path)
        finished_at = float(payload.get("completed_at") or done_path.stat().st_mtime)
        started_at = payload.get("started_at")
        duration_seconds = payload.get("duration_seconds")
        if duration_seconds is None and started_at is not None:
            duration_seconds = finished_at - float(started_at)
        progress = 1.0
        stage = "done"
        summary = "Completed"
    elif failed_path.exists():
        state = "failed"
        payload = _read_json(failed_path)
        finished_at = float(payload.get("failed_at") or failed_path.stat().st_mtime)
        started_at = payload.get("started_at")
        duration_seconds = payload.get("duration_seconds")
        if duration_seconds is None and started_at is not None:
            duration_seconds = finished_at - float(started_at)
        progress = 1.0
        stage = "failed"
        summary = str(payload.get("message") or "Failed").splitlines()[0]
    elif running_path.exists():
        state = "running"
        payload = _read_json(running_path)
        started_at = payload.get("started_at")
        heartbeat_at = payload.get("heartbeat_at")
        if started_at is not None:
            elapsed_seconds = max(0.0, now - float(started_at))
        stage, progress, summary = _progress_from_log(log_path, epochs=epochs, tail_lines=tail_lines)

    worker_name = payload.get("worker_name")
    eta_seconds = None
    if state == "running" and elapsed_seconds is not None:
        if progress >= 0.05:
            eta_seconds = max(0.0, elapsed_seconds * (1.0 / min(progress, 0.99) - 1.0))
        elif mean_done_seconds is not None:
            eta_seconds = max(0.0, mean_done_seconds - elapsed_seconds)
    elif state == "pending" and mean_done_seconds is not None:
        eta_seconds = mean_done_seconds

    return JobSnapshot(
        category=category,
        model=model,
        slug=paths["slug"],
        state=state,
        worker_name=worker_name,
        started_at=float(started_at) if started_at is not None else None,
        heartbeat_at=float(heartbeat_at) if heartbeat_at is not None else None,
        finished_at=finished_at,
        duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
        elapsed_seconds=elapsed_seconds,
        progress=progress,
        stage=stage,
        eta_seconds=eta_seconds,
        log_path=log_path,
        log_summary=summary,
    )


def _estimate_queue_eta(
    snapshots: list[JobSnapshot],
    *,
    mean_done_seconds: float | None,
    parallelism: int | None,
) -> float | None:
    running = [job for job in snapshots if job.state == "running"]
    pending = [job for job in snapshots if job.state == "pending"]

    running_etas = [job.eta_seconds for job in running if job.eta_seconds is not None]
    if not running and not pending:
        return 0.0
    if not running_etas and mean_done_seconds is None:
        return None

    worker_slots = parallelism or max(1, len(running_etas) or len(running) or 1)
    slot_times = sorted(running_etas)
    if len(slot_times) < worker_slots:
        slot_times.extend([0.0] * (worker_slots - len(slot_times)))
    elif len(slot_times) > worker_slots:
        slot_times = slot_times[:worker_slots]

    pending_duration = mean_done_seconds
    if pending_duration is None:
        pending_duration = statistics.mean(running_etas) if running_etas else None
    if pending_duration is None:
        return max(slot_times) if slot_times else None

    for _ in pending:
        earliest_index = min(range(len(slot_times)), key=slot_times.__getitem__)
        slot_times[earliest_index] += pending_duration
    return max(slot_times) if slot_times else 0.0


def _render_snapshot(
    *,
    drive_root: Path,
    manifest: dict[str, Any],
    snapshots: list[JobSnapshot],
    tail_lines: int,
    parallelism: int | None,
) -> str:
    now = time.time()
    done = [job for job in snapshots if job.state == "done"]
    failed = [job for job in snapshots if job.state == "failed"]
    running = [job for job in snapshots if job.state == "running"]
    pending = [job for job in snapshots if job.state == "pending"]

    durations = [job.duration_seconds for job in done if job.duration_seconds]
    mean_done_seconds = statistics.mean(durations) if durations else None
    median_done_seconds = statistics.median(durations) if durations else None
    queue_eta = _estimate_queue_eta(
        snapshots,
        mean_done_seconds=mean_done_seconds,
        parallelism=parallelism,
    )

    lines = [
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        f"drive_root: {drive_root}",
        (
            f"queue: done={len(done)} failed={len(failed)} "
            f"running={len(running)} pending={len(pending)} total={len(snapshots)}"
        ),
    ]

    if mean_done_seconds is not None:
        lines.append(
            f"completed-job duration: mean={_format_seconds(mean_done_seconds)} "
            f"median={_format_seconds(median_done_seconds)}"
        )
    else:
        lines.append("completed-job duration: not enough completed jobs yet")

    lines.append(f"estimated time left: {_format_seconds(queue_eta)}")

    if running:
        lines.append("")
        lines.append("running jobs:")
        for job in sorted(running, key=lambda item: item.slug):
            heartbeat_age = None
            if job.heartbeat_at is not None:
                heartbeat_age = max(0.0, now - job.heartbeat_at)
            lines.append(
                "  - "
                f"{job.category}/{job.model} "
                f"[worker={job.worker_name or '?'}] "
                f"stage={job.stage} "
                f"progress={job.progress * 100:.0f}% "
                f"elapsed={_format_seconds(job.elapsed_seconds)} "
                f"eta={_format_seconds(job.eta_seconds)} "
                f"heartbeat_age={_format_seconds(heartbeat_age)}"
            )
            lines.append(f"    {job.log_summary}")

    if pending:
        lines.append("")
        lines.append("pending jobs:")
        for job in pending:
            lines.append(f"  - {job.category}/{job.model}")

    if failed:
        lines.append("")
        lines.append("failed jobs:")
        for job in failed:
            lines.append(
                f"  - {job.category}/{job.model} [{job.worker_name or '?'}] {job.log_summary}"
            )

    if running:
        lines.append("")
        lines.append(f"latest log tails ({tail_lines} lines each):")
        for job in sorted(running, key=lambda item: item.slug):
            lines.append(f"  --- {job.log_path.name} ---")
            tail = _latest_lines(job.log_path, max_lines=tail_lines)
            if tail:
                lines.extend(f"  {line}" for line in tail)
            else:
                lines.append("  <no log output yet>")

    manifest_preview = {
        "categories": manifest.get("categories", []),
        "models": manifest.get("models", []),
        "epochs": manifest.get("epochs"),
        "eval_step": manifest.get("eval_step"),
    }
    lines.append("")
    lines.append("manifest:")
    lines.append(json.dumps(manifest_preview, indent=2))
    return "\n".join(lines)


def _load_snapshot(
    *,
    drive_root: Path,
    tail_lines: int,
) -> tuple[dict[str, Any], list[JobSnapshot], float | None]:
    manifest_path = drive_root / "state" / "jobs" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing job manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    epochs = int(manifest.get("epochs") or 50)
    now = time.time()

    provisional_done_durations: list[float] = []
    for job in manifest["jobs"]:
        paths = _job_paths(drive_root, job["category"], job["model"])
        if paths["done"].exists():
            payload = _read_json(paths["done"])
            duration = payload.get("duration_seconds")
            if duration:
                provisional_done_durations.append(float(duration))

    mean_done_seconds = (
        statistics.mean(provisional_done_durations) if provisional_done_durations else None
    )

    snapshots = [
        _build_snapshot(
            drive_root,
            job,
            now=now,
            epochs=epochs,
            tail_lines=tail_lines,
            mean_done_seconds=mean_done_seconds,
        )
        for job in manifest["jobs"]
    ]
    return manifest, snapshots, mean_done_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor the shared Colab runner queue")
    parser.add_argument("--drive-root", required=True, type=Path, help="Shared Drive workspace root")
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=DEFAULT_TAIL_LINES,
        help="Number of recent log lines to show per running job",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
        help="Optional worker count override for ETA simulation",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh in-place until interrupted",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=DEFAULT_REFRESH_SECONDS,
        help="Refresh cadence for --watch",
    )
    return parser.parse_args()


def _print_snapshot(args: argparse.Namespace) -> None:
    drive_root = args.drive_root.expanduser().resolve()
    manifest, snapshots, _mean_done_seconds = _load_snapshot(
        drive_root=drive_root,
        tail_lines=args.tail_lines,
    )
    output = _render_snapshot(
        drive_root=drive_root,
        manifest=manifest,
        snapshots=snapshots,
        tail_lines=args.tail_lines,
        parallelism=args.parallelism,
    )
    print(output)


def main() -> int:
    args = parse_args()
    if not args.watch:
        _print_snapshot(args)
        return 0

    try:
        while True:
            os.system("clear")
            _print_snapshot(args)
            print(f"\nRefreshing every {args.refresh_seconds}s. Press Ctrl+C to stop.")
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        print("\nStopped watch.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
