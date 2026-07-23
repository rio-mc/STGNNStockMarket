import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_full_benchmark import (
    GRAPH_JOBS,
    NON_GRAPH_JOBS,
    BenchmarkRunLock,
    BenchmarkSpec,
    apply_resume_manifest_dates,
    build_groups,
    build_worker_command,
    load_completed_keys,
    lost_completed_keys,
    main,
    pending_tasks,
    write_or_validate_manifest,
)


def make_spec(workspace_tmp: Path) -> BenchmarkSpec:
    return BenchmarkSpec(
        datasets=("sp500", "nasdaq100"),
        seeds=(42, 43, 44, 45),
        k_values=tuple(range(8)),
        interval="1h",
        prediction_window="1d",
        date_start="2024-07-23",
        date_end="2026-07-22",
        graph_mode="knn_mst",
        graph_embed="pca",
        graph_ablation="none",
        ablate_feature="none",
        threshold_policy="macro_f1_dense",
        deterministic=True,
        reference_k=3,
        results_dir=str(workspace_tmp),
    )


def test_matrix_varies_k_only_for_models_that_consume_the_graph(workspace_tmp):
    spec = make_spec(workspace_tmp)
    groups = build_groups(spec, {"sp500": ("AAPL",), "nasdaq100": ("MSFT",)})

    assert len(NON_GRAPH_JOBS) == 6
    assert len(GRAPH_JOBS) == 8
    assert len(groups) == 18
    assert sum(len(spec.seeds) * len(group.jobs) for group in groups) == 560

    non_graph = [group for group in groups if group.jobs == NON_GRAPH_JOBS]
    graph = [group for group in groups if group.jobs == GRAPH_JOBS]
    assert {group.k for group in non_graph} == {3}
    assert {group.k for group in graph} == set(range(8))


def test_worker_command_carries_frozen_window_and_all_seeds(workspace_tmp):
    spec = make_spec(workspace_tmp)
    group = build_groups(spec, {"sp500": ("AAPL",), "nasdaq100": ()})[0]

    command = build_worker_command(
        group,
        spec,
        training_log="summary",
        headless_report="compact",
        extra_args=("--lstm_epochs", "2"),
    )

    assert "run_benchmark_worker.py" in command[1]
    assert command[command.index("--seeds") + 1 : command.index("--resume")] == (
        "42",
        "43",
        "44",
        "45",
    )
    assert command[command.index("--date_start") + 1] == "2024-07-23"
    assert command[command.index("--date_end") + 1] == "2026-07-22"
    assert "--deterministic" in command
    assert command[-2:] == ("--lstm_epochs", "2")


def test_successful_csv_row_is_removed_from_pending_tasks(workspace_tmp):
    spec = make_spec(workspace_tmp)
    group = build_groups(spec, {"sp500": ("AAPL",), "nasdaq100": ()})[0]
    csv_path = workspace_tmp / "runs" / "run_results.csv"
    csv_path.parent.mkdir(parents=True)
    fieldnames = [
        "status",
        "universe_id",
        "ticker",
        "seed",
        "model_label",
        "k",
        "interval",
        "prediction_window",
        "graph_mode",
        "graph_embed",
        "graph_ablation",
        "ablate_feature",
        "threshold_policy",
        "deterministic",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "status": "success",
                "universe_id": "sp500",
                "ticker": "AAPL",
                "seed": 42,
                "model_label": "lstm",
                "k": 3,
                "interval": "1h",
                "prediction_window": "1d",
                "graph_mode": "knn_mst",
                "graph_embed": "pca",
                "graph_ablation": "none",
                "ablate_feature": "none",
                "threshold_policy": "macro_f1_dense",
                "deterministic": "true",
            }
        )

    pending = pending_tasks(group, spec, load_completed_keys(workspace_tmp))

    assert len(pending) == len(spec.seeds) * len(NON_GRAPH_JOBS) - 1
    assert (42, NON_GRAPH_JOBS[0]) not in pending


def test_lost_completed_keys_detects_storage_regression():
    first = ("sp500", "AAPL", 42, "lstm")
    second = ("sp500", "AAPL", 43, "lstm")

    assert lost_completed_keys({first, second}, {second}) == {first}
    assert lost_completed_keys({second}, {first, second}) == set()


def test_dry_run_does_not_create_manifest(workspace_tmp):
    exit_code = main(
        [
            "--datasets",
            "sp500",
            "--seeds",
            "42",
            "--k-values",
            "0",
            "--results-dir",
            str(workspace_tmp),
            "--dry-run",
            "--max-groups",
            "1",
        ]
    )

    assert exit_code == 0
    assert not (workspace_tmp / "benchmark_manifest.json").exists()


def test_resume_reuses_frozen_manifest_dates(workspace_tmp):
    manifest = workspace_tmp / "benchmark_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "spec": {
                    "date_start": "2024-07-23",
                    "date_end": "2026-07-22",
                }
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        resume=True,
        results_dir=str(workspace_tmp),
        date_start=None,
        date_end=None,
    )

    apply_resume_manifest_dates(args)

    assert args.date_start == "2024-07-23"
    assert args.date_end == "2026-07-22"


def test_manifest_resume_accepts_json_round_trip_of_tuple_fields(workspace_tmp):
    spec = make_spec(workspace_tmp)
    universes = {"sp500": ("AAPL",), "nasdaq100": ("MSFT",)}

    first_path = write_or_validate_manifest(
        spec,
        universes,
        resume=False,
    )
    resumed_path = write_or_validate_manifest(
        spec,
        universes,
        resume=True,
    )

    assert resumed_path == first_path


def test_benchmark_lock_rejects_second_live_launcher(workspace_tmp):
    first = BenchmarkRunLock(workspace_tmp)
    second = BenchmarkRunLock(workspace_tmp)

    first.acquire()
    try:
        try:
            second.acquire()
        except RuntimeError as exc:
            assert "already running" in str(exc)
        else:
            raise AssertionError("Second launcher unexpectedly acquired the lock")
    finally:
        first.release()

    second.acquire()
    second.release()
