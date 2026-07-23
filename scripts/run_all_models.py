"""Thin sweeper for the existing headless experiment CLI.

This module owns only enumeration and process control.  Data loading, pipeline
construction, model execution, evaluation, and result persistence remain the
responsibility of the configured headless CLI (``core.main`` by default).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_manager import ConfigManager
from core.job_queue import MODEL_FAMILIES


STGNN_BACKENDS = ("gcn", "graphsage", "gat", "nnconv")


@dataclass(frozen=True)
class ModelJob:
    model: str
    graph_backend: str | None = None

    @property
    def label(self) -> str:
        if self.model == "stgnn":
            return f"stgnn-{self.graph_backend}"
        return self.model


@dataclass(frozen=True)
class SweepPlan:
    jobs: tuple[ModelJob, ...]
    headless_argv: tuple[str, ...]
    headless_module: str
    target_stock: str
    seed: int
    results_dir: str = "./results"
    fail_fast: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class JobOutcome:
    label: str
    returncode: int | None
    command: tuple[str, ...]


def all_model_jobs() -> tuple[ModelJob, ...]:
    """The 10 non-STGNN models plus four STGNN backend configurations."""

    jobs = [
        ModelJob(model=model)
        for model in MODEL_FAMILIES["all"]
        if model != "stgnn"
    ]
    jobs.extend(
        ModelJob(model="stgnn", graph_backend=backend)
        for backend in STGNN_BACKENDS
    )
    if len(jobs) != 14:
        raise RuntimeError(f"Expected 14 model configurations, found {len(jobs)}")
    return tuple(jobs)


ALL_MODEL_JOBS = all_model_jobs()
MODEL_LABELS = tuple(job.label for job in ALL_MODEL_JOBS)
CONTROLLED_OPTIONS = frozenset(
    {"--run_mode", "--target_stock", "--seed", "--model", "--graph_model"}
)


def _parse_sweeper_options(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=MODEL_LABELS,
        default=None,
        help="Run only these configurations; the default is all 14.",
    )
    parser.add_argument(
        "--headless-module",
        default="core.main",
        help="Python module providing the existing headless CLI (default: core.main).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first non-zero headless CLI exit code.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the 14 configuration labels and exit.",
    )
    return parser.parse_known_args(argv)


def _parse_headless_args(argv: Sequence[str]):
    """Use the headless CLI's own parser solely to validate target and seed."""

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        return ConfigManager.parseArgs()
    finally:
        sys.argv = original_argv


def _without_controlled_options(argv: Sequence[str]) -> tuple[str, ...]:
    """Remove values that the sweeper sets separately for each child process."""

    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        option = token.split("=", 1)[0]
        if option in CONTROLLED_OPTIONS:
            index += 1 if "=" in token else 2
            continue
        cleaned.append(token)
        index += 1
    return tuple(cleaned)


def make_plan(argv: Sequence[str] | None = None) -> SweepPlan | None:
    sweeper, headless_argv = _parse_sweeper_options(argv)
    if sweeper.list_models:
        return None

    headless_args = _parse_headless_args(headless_argv)
    target = str(getattr(headless_args, "target_stock", "") or "").strip().upper()
    if not target:
        raise ValueError(
            "The sweeper requires one target. Pass, for example, --target_stock AAPL."
        )

    requested = set(sweeper.only or MODEL_LABELS)
    jobs = tuple(job for job in ALL_MODEL_JOBS if job.label in requested)
    return SweepPlan(
        jobs=jobs,
        headless_argv=_without_controlled_options(headless_argv),
        headless_module=str(sweeper.headless_module).strip(),
        target_stock=target,
        seed=int(getattr(headless_args, "seed", 42)),
        results_dir=str(getattr(headless_args, "results_dir", "./results")),
        fail_fast=bool(sweeper.fail_fast),
        dry_run=bool(sweeper.dry_run),
    )


def build_command(plan: SweepPlan, job: ModelJob) -> tuple[str, ...]:
    """Build one isolated invocation of the existing headless CLI."""

    command = [
        sys.executable,
        "-m",
        plan.headless_module,
        *plan.headless_argv,
        "--run_mode",
        "headless",
        "--target_stock",
        plan.target_stock,
        "--seed",
        str(plan.seed),
        "--model",
        job.model,
    ]
    if job.model == "stgnn":
        command.extend(("--graph_model", str(job.graph_backend)))
    return tuple(command)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("run_all_models")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def _persisted_success_for_job(
    plan: SweepPlan,
    job: ModelJob,
    *,
    child_started_at: datetime,
) -> bool:
    """Confirm an exact successful row written during this child process."""

    results_root = Path(plan.results_dir)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    csv_path = results_root / "runs" / "run_results.csv"
    if not csv_path.exists():
        return False

    expected_backend = str(job.graph_backend or "").lower()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("status", "")).lower() != "success":
                    continue
                if str(row.get("model", "")).lower() != job.model:
                    continue
                if str(row.get("ticker", "")).upper() != plan.target_stock:
                    continue
                if str(row.get("seed", "")) != str(plan.seed):
                    continue
                if job.model == "stgnn" and str(row.get("graph_model", "")).lower() != expected_backend:
                    continue
                timestamp = datetime.fromisoformat(str(row.get("timestamp_start", "")))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if timestamp >= child_started_at:
                    return True
    except (OSError, ValueError, TypeError):
        return False
    return False


def run_sweep(plan: SweepPlan, logger: logging.Logger | None = None) -> list[JobOutcome]:
    """Invoke the headless CLI once per job and return only process outcomes."""

    logger = logger or setup_logging()
    child_env = os.environ.copy()
    child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    child_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    logger.info(
        "Running %d headless experiments: target=%s seed=%d runner=%s",
        len(plan.jobs),
        plan.target_stock,
        plan.seed,
        plan.headless_module,
    )

    outcomes: list[JobOutcome] = []
    for index, job in enumerate(plan.jobs, start=1):
        command = build_command(plan, job)

        if plan.dry_run:
            logger.info(
                "[%d/%d] %s (dry run; not executed)",
                index,
                len(plan.jobs),
                job.label,
            )
            print(f"  {subprocess.list2cmdline(command)}")
            returncode = None
        else:
            logger.info("[%d/%d] Running %s", index, len(plan.jobs), job.label)
            # Inherit stdout/stderr so the existing CLI remains fully in control
            # of progress output, prompts, logging, and result persistence.
            child_started_at = datetime.now(timezone.utc)
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=child_env,
                check=False,
            )
            returncode = int(completed.returncode)
            if returncode != 0 and _persisted_success_for_job(
                plan,
                job,
                child_started_at=child_started_at,
            ):
                logger.warning(
                    "[%d/%d] %s exited %d after its successful row was persisted; "
                    "reconciled as completed.",
                    index,
                    len(plan.jobs),
                    job.label,
                    returncode,
                )
                returncode = 0

        outcomes.append(
            JobOutcome(label=job.label, returncode=returncode, command=command)
        )
        if returncode == 0:
            logger.info("[%d/%d] Completed %s", index, len(plan.jobs), job.label)
        elif returncode is not None:
            logger.error(
                "[%d/%d] Failed %s (exit code %d)",
                index,
                len(plan.jobs),
                job.label,
                returncode,
            )
            if plan.fail_fast:
                break

    executed = sum(outcome.returncode is not None for outcome in outcomes)
    failures = sum(
        outcome.returncode not in (None, 0) for outcome in outcomes
    )
    if plan.dry_run:
        logger.info("Dry run complete: %d commands shown, 0 executed.", len(outcomes))
    else:
        logger.info(
            "Sweep complete: %d succeeded, %d failed, %d not run.",
            executed - failures,
            failures,
            len(plan.jobs) - executed,
        )
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    sweeper, _ = _parse_sweeper_options(raw_argv)
    if sweeper.list_models:
        for label in MODEL_LABELS:
            print(label)
        return 0

    plan = make_plan(raw_argv)
    if plan is None:
        return 0

    try:
        outcomes = run_sweep(plan)
    except KeyboardInterrupt:
        setup_logging().warning("Sweep interrupted by user.")
        return 130
    return 1 if any(outcome.returncode not in (None, 0) for outcome in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
