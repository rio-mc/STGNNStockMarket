from types import SimpleNamespace

import scripts.run_all_models as sweeper
from scripts.run_all_models import (
    ALL_MODEL_JOBS,
    MODEL_LABELS,
    STGNN_BACKENDS,
    ModelJob,
    SweepPlan,
)


def test_all_model_suite_has_fourteen_unique_configurations():
    assert len(ALL_MODEL_JOBS) == 14
    assert len(set(MODEL_LABELS)) == 14
    assert "rf" not in MODEL_LABELS
    assert "random_forest" in MODEL_LABELS


def test_stgnn_backends_are_separate_jobs():
    stgnn_jobs = [job for job in ALL_MODEL_JOBS if job.model == "stgnn"]

    assert tuple(job.graph_backend for job in stgnn_jobs) == STGNN_BACKENDS
    assert tuple(job.label for job in stgnn_jobs) == (
        "stgnn-gcn",
        "stgnn-graphsage",
        "stgnn-gat",
        "stgnn-nnconv",
    )


def test_command_delegates_to_headless_cli_with_backend_override():
    plan = SweepPlan(
        jobs=(ModelJob("stgnn", "gat"),),
        headless_argv=("--interval", "1h", "--target_stock", "AAPL"),
        headless_module="core.main",
        target_stock="AAPL",
        seed=42,
    )

    command = sweeper.build_command(plan, plan.jobs[0])

    assert command[1:3] == ("-m", "core.main")
    assert command[-4:] == ("--model", "stgnn", "--graph_model", "gat")
    assert "--interval" in command


def test_plan_removes_values_owned_by_the_sweeper():
    plan = sweeper.make_plan(
        [
            "--target_stock",
            "AAPL",
            "--seed=7",
            "--model",
            "lstm",
            "--graph_model",
            "gcn",
            "--interval",
            "1h",
            "--only",
            "random_forest",
        ]
    )

    assert plan is not None
    assert plan.headless_argv == ("--interval", "1h")
    assert plan.target_stock == "AAPL"
    assert plan.seed == 7


def test_sweeper_continues_after_failed_headless_process(monkeypatch):
    plan = SweepPlan(
        jobs=(ModelJob("arima"), ModelJob("random_forest")),
        headless_argv=("--target_stock", "AAPL"),
        headless_module="core.main",
        target_stock="AAPL",
        seed=42,
    )
    returncodes = iter((1, 0))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(sweeper.subprocess, "run", fake_run)

    outcomes = sweeper.run_sweep(plan)

    assert [outcome.returncode for outcome in outcomes] == [1, 0]
    assert len(calls) == 2
    assert calls[1][-2:] == ("--model", "random_forest")


def test_dry_run_does_not_report_jobs_as_executed(monkeypatch):
    plan = SweepPlan(
        jobs=(ModelJob("lstm"),),
        headless_argv=(),
        headless_module="core.main",
        target_stock="AAPL",
        seed=42,
        dry_run=True,
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("dry run must not create a child process")

    monkeypatch.setattr(sweeper.subprocess, "run", unexpected_run)

    outcomes = sweeper.run_sweep(plan)

    assert len(outcomes) == 1
    assert outcomes[0].returncode is None


def test_native_exit_is_reconciled_only_when_success_row_was_persisted(monkeypatch):
    plan = SweepPlan(
        jobs=(ModelJob("lstm"),),
        headless_argv=(),
        headless_module="core.main",
        target_stock="AAPL",
        seed=42,
    )
    monkeypatch.setattr(
        sweeper.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=3221226505),
    )
    monkeypatch.setattr(sweeper, "_persisted_success_for_job", lambda *_args, **_kwargs: True)

    outcomes = sweeper.run_sweep(plan)

    assert outcomes[0].returncode == 0
