"""Resumable full-factorial benchmark launcher.

The launcher owns matrix expansion and process recovery. Each worker owns one
universe/ticker/k state and persists every model result through ExperimentStore
before moving to the next task.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.universe_service import UniverseService
from scripts.run_all_models import ALL_MODEL_JOBS, MODEL_LABELS, ModelJob


GRAPH_MODELS = frozenset({"gcn", "gat", "nnconv", "graphsage", "stgnn"})
GRAPH_JOBS = tuple(job for job in ALL_MODEL_JOBS if job.model in GRAPH_MODELS)
NON_GRAPH_JOBS = tuple(job for job in ALL_MODEL_JOBS if job.model not in GRAPH_MODELS)
DEFAULT_DATASETS = ("sp500", "nasdaq100")
DEFAULT_SEEDS = tuple(range(42, 46))
DEFAULT_K_VALUES = tuple(range(8))
CONTROLLED_CORE_OPTIONS = frozenset(
    {
        "--run_mode",
        "--dataset_name",
        "--top_n",
        "--target_stock",
        "--seed",
        "--model",
        "--graph_model",
        "--k",
        "--interval",
        "--prediction_window",
        "--date_start",
        "--date_end",
        "--graph_mode",
        "--graph_embed",
        "--graph_ablation",
        "--ablate_feature",
        "--decision-threshold-policy",
        "--training-log",
        "--headless-report",
        "--results_dir",
        "--deterministic",
        "--no_deterministic",
    }
)


@dataclass(frozen=True)
class BenchmarkSpec:
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    k_values: tuple[int, ...]
    interval: str
    prediction_window: str
    date_start: str
    date_end: str
    graph_mode: str
    graph_embed: str
    graph_ablation: str
    ablate_feature: str
    threshold_policy: str
    deterministic: bool
    reference_k: int
    results_dir: str


@dataclass(frozen=True)
class WorkerGroup:
    dataset: str
    ticker: str
    k: int
    jobs: tuple[ModelJob, ...]
    universe_size: int

    @property
    def label(self) -> str:
        scope = "graph" if self.jobs and self.jobs[0].model in GRAPH_MODELS else "non_graph"
        return f"{self.dataset}/{self.ticker}/{scope}/k{self.k}"


class BenchmarkRunLock:
    """Prevent two launchers from writing to the same benchmark directory."""

    def __init__(self, results_dir: str | Path) -> None:
        self.path = Path(results_dir) / ".benchmark.lock"
        self.token = f"{os.getpid()}:{uuid4().hex}"
        self.acquired = False

    @staticmethod
    def _process_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    existing = self.path.read_text(encoding="utf-8").strip()
                    existing_pid = int(existing.split(":", 1)[0])
                except (OSError, ValueError):
                    existing_pid = -1
                if self._process_is_running(existing_pid):
                    raise RuntimeError(
                        "A benchmark launcher is already running for this "
                        f"results directory (pid={existing_pid}): {self.path}"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue

            try:
                os.write(descriptor, self.token.encode("utf-8"))
            finally:
                os.close(descriptor)
            self.acquired = True
            return

        raise RuntimeError(f"Could not acquire benchmark lock: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self.path.read_text(encoding="utf-8").strip() == self.token:
                self.path.unlink(missing_ok=True)
        except OSError:
            pass
        self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("full_benchmark")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _normalise_unique(values: Iterable, transform):
    return tuple(dict.fromkeys(transform(value) for value in values))


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run every meaningful model configuration across two universes.",
    )
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--reference-k", type=int, default=3)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--prediction_window", "--prediction-window", default="1d")
    parser.add_argument("--date_start", "--date-start", default=None)
    parser.add_argument("--date_end", "--date-end", default=None)
    parser.add_argument("--graph_mode", "--graph-mode", default="knn_mst")
    parser.add_argument("--graph_embed", "--graph-embed", default="pca")
    parser.add_argument("--graph_ablation", "--graph-ablation", default="none")
    parser.add_argument("--ablate_feature", "--ablate-feature", default="none")
    parser.add_argument(
        "--decision-threshold-policy",
        choices=("fixed", "macro_f1_dense"),
        default="macro_f1_dense",
    )
    determinism = parser.add_mutually_exclusive_group()
    determinism.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Require deterministic algorithms (default).",
    )
    determinism.add_argument(
        "--no-deterministic",
        "--no_deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow non-deterministic kernels.",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument("--training-log", choices=("quiet", "summary", "epochs"), default="summary")
    parser.add_argument("--headless-report", choices=("compact", "full", "none"), default="compact")
    parser.add_argument("--results_dir", "--results-dir", default="./results/full_benchmark")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Run only the first N pending worker groups; useful for smoke tests.",
    )
    args, extra_args = parser.parse_known_args(argv)

    args.datasets = _normalise_unique(args.datasets, lambda value: str(value).strip().lower())
    args.seeds = _normalise_unique(args.seeds, int)
    args.k_values = _normalise_unique(args.k_values, int)
    if not args.seeds:
        parser.error("--seeds must contain at least one value")
    if not args.k_values or min(args.k_values) < 0:
        parser.error("--k-values must contain non-negative integers")
    if args.reference_k < 0:
        parser.error("--reference-k must be non-negative")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("--max-groups must be positive")
    conflicting = sorted(
        {
            str(token).split("=", 1)[0]
            for token in extra_args
            if str(token).split("=", 1)[0] in CONTROLLED_CORE_OPTIONS
        }
    )
    if conflicting:
        parser.error(
            "These options are controlled by the benchmark launcher: "
            + ", ".join(conflicting)
        )
    return args, tuple(extra_args)


def resolve_dates(date_start: str | None, date_end: str | None) -> tuple[str, str]:
    end = date.fromisoformat(date_end) if date_end else date.today()
    start = date.fromisoformat(date_start) if date_start else end - timedelta(days=729)
    if start >= end:
        raise ValueError("date_start must be earlier than date_end")
    return start.isoformat(), end.isoformat()


def make_spec(args) -> BenchmarkSpec:
    date_start, date_end = resolve_dates(args.date_start, args.date_end)
    return BenchmarkSpec(
        datasets=tuple(args.datasets),
        seeds=tuple(args.seeds),
        k_values=tuple(args.k_values),
        interval=str(args.interval),
        prediction_window=str(args.prediction_window),
        date_start=date_start,
        date_end=date_end,
        graph_mode=str(args.graph_mode),
        graph_embed=str(args.graph_embed),
        graph_ablation=str(args.graph_ablation),
        ablate_feature=str(args.ablate_feature),
        threshold_policy=str(args.decision_threshold_policy),
        deterministic=bool(args.deterministic),
        reference_k=int(args.reference_k),
        results_dir=str(Path(args.results_dir).resolve()),
    )


def apply_resume_manifest_dates(args) -> None:
    """Reuse the original frozen window when resuming on a later date."""

    if not args.resume:
        return
    path = Path(args.results_dir).resolve() / "benchmark_manifest.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_spec = payload.get("spec", {}) or {}
    if args.date_start is None:
        args.date_start = stored_spec.get("date_start")
    if args.date_end is None:
        args.date_end = stored_spec.get("date_end")


def resolve_universes(datasets: Iterable[str]) -> dict[str, tuple[str, ...]]:
    service = UniverseService()
    return {
        dataset: tuple(
            service.resolve_definition(
                universe_id=dataset,
                universe_provider="static_csv",
                top_n=None,
            ).tickers
        )
        for dataset in datasets
    }


def build_groups(spec: BenchmarkSpec, universes: dict[str, tuple[str, ...]]) -> tuple[WorkerGroup, ...]:
    groups: list[WorkerGroup] = []
    for dataset in spec.datasets:
        tickers = universes[dataset]
        for ticker in tickers:
            groups.append(
                WorkerGroup(
                    dataset=dataset,
                    ticker=ticker,
                    k=spec.reference_k,
                    jobs=NON_GRAPH_JOBS,
                    universe_size=len(tickers),
                )
            )
            groups.extend(
                WorkerGroup(
                    dataset=dataset,
                    ticker=ticker,
                    k=k,
                    jobs=GRAPH_JOBS,
                    universe_size=len(tickers),
                )
                for k in spec.k_values
            )
    return tuple(groups)


def csv_model_label(job: ModelJob) -> str:
    if job.model == "stgnn":
        return f"stgnn+{job.graph_backend}"
    return job.model


def task_key(
    *,
    dataset: str,
    ticker: str,
    seed: int,
    job: ModelJob,
    k: int,
    spec: BenchmarkSpec,
) -> tuple:
    return (
        str(dataset).lower(),
        str(ticker).upper(),
        int(seed),
        csv_model_label(job),
        int(k),
        spec.interval,
        spec.prediction_window,
        spec.graph_mode,
        spec.graph_embed,
        spec.graph_ablation,
        spec.ablate_feature,
        spec.threshold_policy,
        bool(spec.deterministic),
    )


def load_completed_keys(results_dir: Path) -> set[tuple]:
    csv_path = results_dir / "runs" / "run_results.csv"
    if not csv_path.exists():
        return set()

    completed: set[tuple] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).lower() != "success":
                continue
            try:
                completed.add(
                    (
                        str(row.get("universe_id", "")).lower(),
                        str(row.get("ticker", "")).upper(),
                        int(row.get("seed", "")),
                        str(row.get("model_label", "")).lower(),
                        int(row.get("k", "")),
                        str(row.get("interval", "")),
                        str(row.get("prediction_window", "")),
                        str(row.get("graph_mode", "")),
                        str(row.get("graph_embed", "")),
                        str(row.get("graph_ablation", "")),
                        str(row.get("ablate_feature", "")),
                        str(row.get("threshold_policy", "")),
                        str(row.get("deterministic", "")).strip().lower()
                        in {"1", "true", "yes"},
                    )
                )
            except (TypeError, ValueError):
                continue
    return completed


def pending_tasks(group: WorkerGroup, spec: BenchmarkSpec, completed: set[tuple]):
    return tuple(
        (seed, job)
        for seed in spec.seeds
        for job in group.jobs
        if task_key(
            dataset=group.dataset,
            ticker=group.ticker,
            seed=seed,
            job=job,
            k=group.k,
            spec=spec,
        )
        not in completed
    )


def lost_completed_keys(
    known_completed: set[tuple],
    observed_completed: set[tuple],
) -> set[tuple]:
    """Return results that vanished since the previous persistence checkpoint."""

    return known_completed.difference(observed_completed)


def write_or_validate_manifest(
    spec: BenchmarkSpec,
    universes: dict[str, tuple[str, ...]],
    *,
    resume: bool,
    extra_args: Sequence[str] = (),
) -> Path:
    results_dir = Path(spec.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "benchmark_manifest.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "universe_tickers": {key: list(value) for key, value in universes.items()},
        "model_labels": list(MODEL_LABELS),
        "k_scope": "graph_models_only",
        "extra_core_args": list(extra_args),
    }
    # Compare the same JSON-native representation that is persisted. Dataclass
    # tuple fields deserialize as lists, although they describe the same matrix.
    payload = json.loads(json.dumps(payload))

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_existing.pop("created_at", None)
        comparable_payload = dict(payload)
        comparable_payload.pop("created_at", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError(
                f"Benchmark manifest differs from the requested matrix: {path}. "
                "Use a new --results-dir for a different experiment."
            )
        if not resume:
            raise RuntimeError(
                f"Benchmark already exists at {path}. Pass --resume to continue it."
            )
        return path

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def build_worker_command(
    group: WorkerGroup,
    spec: BenchmarkSpec,
    *,
    training_log: str,
    headless_report: str,
    extra_args: Sequence[str],
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_benchmark_worker.py"),
        "--jobs",
        *(job.label for job in group.jobs),
        "--seeds",
        *(str(seed) for seed in spec.seeds),
        "--resume",
        "--run_mode",
        "headless",
        "--dataset_name",
        group.dataset,
        "--top_n",
        str(group.universe_size),
        "--target_stock",
        group.ticker,
        "--k",
        str(group.k),
        "--interval",
        spec.interval,
        "--prediction_window",
        spec.prediction_window,
        "--date_start",
        spec.date_start,
        "--date_end",
        spec.date_end,
        "--graph_mode",
        spec.graph_mode,
        "--graph_embed",
        spec.graph_embed,
        "--graph_ablation",
        spec.graph_ablation,
        "--ablate_feature",
        spec.ablate_feature,
        "--decision-threshold-policy",
        spec.threshold_policy,
        "--deterministic" if spec.deterministic else "--no_deterministic",
        "--training-log",
        training_log,
        "--headless-report",
        headless_report,
        "--results_dir",
        spec.results_dir,
        *extra_args,
    ]
    return tuple(command)


def append_worker_outcome(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = parse_args(argv)
    if args.dry_run:
        return _run_benchmark(args, extra_args)

    with BenchmarkRunLock(Path(args.results_dir).resolve()):
        return _run_benchmark(args, extra_args)


def _run_benchmark(args, extra_args: Sequence[str]) -> int:
    logger = setup_logging()
    apply_resume_manifest_dates(args)
    spec = make_spec(args)
    universes = resolve_universes(spec.datasets)
    groups = build_groups(spec, universes)
    total_tasks = sum(len(spec.seeds) * len(group.jobs) for group in groups)

    manifest_path = Path(spec.results_dir) / "benchmark_manifest.json"
    if not args.dry_run:
        manifest_path = write_or_validate_manifest(
            spec,
            universes,
            resume=args.resume,
            extra_args=extra_args,
        )
    completed = load_completed_keys(Path(spec.results_dir)) if args.resume else set()
    pending = [(group, pending_tasks(group, spec, completed)) for group in groups]
    pending = [(group, tasks) for group, tasks in pending if tasks]
    pending_task_count = sum(len(tasks) for _, tasks in pending)
    if args.max_groups is not None:
        pending = pending[: args.max_groups]

    logger.info(
        "Benchmark matrix: datasets=%s stocks=%d seeds=%d total_results=%d pending=%d groups=%d",
        ",".join(spec.datasets),
        sum(len(tickers) for tickers in universes.values()),
        len(spec.seeds),
        total_tasks,
        pending_task_count,
        len(pending),
    )
    logger.info("Manifest: %s", manifest_path)
    logger.info("Frozen data window: %s to %s (%s)", spec.date_start, spec.date_end, spec.interval)

    if args.dry_run:
        for group, tasks in pending[:5]:
            command = build_worker_command(
                group,
                spec,
                training_log=args.training_log,
                headless_report=args.headless_report,
                extra_args=extra_args,
            )
            logger.info("DRY RUN %s pending_tasks=%d", group.label, len(tasks))
            print("  " + subprocess.list2cmdline(command))
        if len(pending) > 5:
            logger.info("... %d additional worker groups omitted from preview", len(pending) - 5)
        return 0

    outcome_path = Path(spec.results_dir) / "benchmark_worker_outcomes.jsonl"
    failures = 0
    known_completed = set(completed)
    for index, (group, tasks) in enumerate(pending, start=1):
        completed_before_worker = load_completed_keys(Path(spec.results_dir))
        missing_before_worker = lost_completed_keys(
            known_completed,
            completed_before_worker,
        )
        if missing_before_worker:
            failures += 1
            logger.critical(
                "Storage integrity failure before %s: %d previously completed "
                "results disappeared; aborting benchmark",
                group.label,
                len(missing_before_worker),
            )
            break

        command = build_worker_command(
            group,
            spec,
            training_log=args.training_log,
            headless_report=args.headless_report,
            extra_args=extra_args,
        )
        logger.info(
            "[%d/%d] %s pending_tasks=%d",
            index,
            len(pending),
            group.label,
            len(tasks),
        )
        started = time.monotonic()
        completed_process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )
        raw_returncode = int(completed_process.returncode)
        effective_returncode = raw_returncode
        completed_after_worker = load_completed_keys(Path(spec.results_dir))
        missing_after_worker = lost_completed_keys(
            completed_before_worker,
            completed_after_worker,
        )
        still_pending = pending_tasks(group, spec, completed_after_worker)
        storage_integrity_failure = bool(missing_after_worker)

        if storage_integrity_failure:
            effective_returncode = 2
            logger.critical(
                "Storage integrity failure after %s: %d previously completed "
                "results disappeared; aborting benchmark",
                group.label,
                len(missing_after_worker),
            )
        elif still_pending:
            if raw_returncode == 0:
                effective_returncode = 1
                logger.error(
                    "Worker reported success but %d expected rows are missing: %s",
                    len(still_pending),
                    group.label,
                )
        elif raw_returncode != 0:
            effective_returncode = 0
            logger.warning(
                "Worker exited %d after all rows were persisted; reconciled as complete: %s",
                raw_returncode,
                group.label,
            )

        known_completed = set(completed_after_worker)
        outcome = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "group": group.label,
            "returncode": effective_returncode,
            "raw_returncode": raw_returncode,
            "elapsed_sec": time.monotonic() - started,
            "pending_tasks_at_start": len(tasks),
            "pending_tasks_after_worker": len(still_pending),
            "lost_completed_results": len(missing_after_worker),
        }
        append_worker_outcome(outcome_path, outcome)
        if effective_returncode != 0:
            failures += 1
            logger.error("Worker failed: %s exit=%d", group.label, raw_returncode)
            if args.fail_fast or storage_integrity_failure:
                break

    logger.info("Benchmark launcher finished: failed_groups=%d", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
