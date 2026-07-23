"""Grouped worker used by run_full_benchmark.py."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_manager import ConfigManager
from scripts.run_all_models import MODEL_LABELS, ModelJob
from scripts.run_full_benchmark import BenchmarkSpec, load_completed_keys, task_key


_WORKER_LIFETIME_HOLD = None


def parse_worker_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jobs", nargs="+", choices=MODEL_LABELS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    worker_args, core_args = parser.parse_known_args(argv)
    return worker_args, list(core_args)


def parse_core_args(core_argv: Sequence[str]):
    original = sys.argv[:]
    try:
        sys.argv = [original[0], *core_argv]
        return ConfigManager.parseArgs()
    finally:
        sys.argv = original


def label_to_job(label: str) -> ModelJob:
    if label.startswith("stgnn-"):
        return ModelJob("stgnn", label.split("-", 1)[1])
    return ModelJob(label)


def make_spec(args, seeds: Sequence[int]) -> BenchmarkSpec:
    return BenchmarkSpec(
        datasets=(str(args.dataset_name),),
        seeds=tuple(int(seed) for seed in seeds),
        k_values=(int(args.k),),
        interval=str(args.interval),
        prediction_window=str(args.prediction_window),
        date_start=str(args.date_start),
        date_end=str(args.date_end),
        graph_mode=str(args.graph_mode),
        graph_embed=str(args.graph_embed),
        graph_ablation=str(args.graph_ablation),
        ablate_feature=str(args.ablate_feature),
        threshold_policy=str(args.decision_threshold_policy),
        deterministic=bool(args.deterministic),
        reference_k=int(args.k),
        results_dir=str(Path(args.results_dir).resolve()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    global _WORKER_LIFETIME_HOLD

    worker_args, core_argv = parse_worker_args(argv)
    args = parse_core_args(core_argv)
    jobs = tuple(label_to_job(label) for label in worker_args.jobs)
    spec = make_spec(args, worker_args.seeds)
    completed = load_completed_keys(Path(spec.results_dir)) if worker_args.resume else set()
    pending = [
        (seed, job)
        for seed in spec.seeds
        for job in jobs
        if task_key(
            dataset=args.dataset_name,
            ticker=args.target_stock,
            seed=seed,
            job=job,
            k=args.k,
            spec=spec,
        )
        not in completed
    ]
    if not pending:
        print("WORKER | no pending tasks")
        return 0

    # Importing MainApp is intentionally deferred until after the cheap resume
    # check so completed groups do not initialize torch/model dependencies.
    from core.main import MainApp

    original_argv = sys.argv[:]
    first_seed, first_job = pending[0]
    initial_core_argv = list(core_argv)
    initial_core_argv.extend(("--model", first_job.model, "--seed", str(first_seed)))
    if first_job.model == "stgnn":
        initial_core_argv.extend(("--graph_model", str(first_job.graph_backend)))

    app = None
    failures = 0
    try:
        sys.argv = [original_argv[0], *initial_core_argv]
        app = MainApp()
        stock, window, _model, state = app.prepare_headless_state(
            stock=args.target_stock,
            gui_window=args.prediction_window,
            model_name=first_job.model,
        )

        for index, (seed, job) in enumerate(pending, start=1):
            result = None
            app.args.seed = int(seed)
            app.args.base_seed = int(seed)
            app.args.model = job.model
            if job.model == "stgnn":
                app.args.graph_model = str(job.graph_backend)
            app._set_all_seeds(seed)

            try:
                result = app.run_headless_from_state(
                    stock=stock,
                    gui_window=window,
                    model_name=job.model,
                    state=state,
                )
                report_mode = str(getattr(app.args, "headless_report", "compact")).lower()
                if report_mode == "full":
                    print(app._format_headless_report(result), flush=True)
                elif report_mode == "compact":
                    print(app._format_headless_result_line(result), flush=True)
                print(
                    f"WORKER | {index}/{len(pending)} | seed={seed} | job={job.label} | success",
                    flush=True,
                )
            except Exception:
                failures += 1
                logging.getLogger("benchmark_worker").exception(
                    "WORKER | %d/%d | seed=%s | job=%s | failed",
                    index,
                    len(pending),
                    seed,
                    job.label,
                )
            finally:
                app._release_headless_result(result)
    finally:
        sys.argv = original_argv
        if app is not None:
            app.shutdown()

    # Keep pandas/torch/graph state alive until the Windows CLI boundary calls
    # os._exit(). Releasing this frame first can enter native destructors and
    # produce 0xC0000409 after every result has already been persisted.
    _WORKER_LIFETIME_HOLD = (app, state, pending)
    return 1 if failures else 0


if __name__ == "__main__":
    exit_code = main()
    if os.name == "nt":
        logging.shutdown()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)
    raise SystemExit(exit_code)
