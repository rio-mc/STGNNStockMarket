import logging
import os
import sys
import types
from types import SimpleNamespace

try:
    import seaborn  # noqa: F401
except ModuleNotFoundError:
    # core.main imports the GUI evaluator, but these lifecycle tests do not
    # render plots and should not require the optional plotting dependency.
    sys.modules["seaborn"] = types.ModuleType("seaborn")

try:
    import yfinance  # noqa: F401
except ModuleNotFoundError:
    sys.modules["yfinance"] = types.ModuleType("yfinance")

from core.experiment_runner import ExperimentRunner
from core.headless_app import FrontendMock, HeadlessApp, HeadlessEvaluator
from core.main import MainApp, _finish_successful_windows_headless_process


def test_main_app_does_not_construct_frontend_in_headless_mode(monkeypatch):
    args = SimpleNamespace(
        run_mode="headless",
        base_seed=42,
        seed=42,
        deterministic=False,
        universe_id="sp500",
        universe_provider="static_csv",
        price_provider="csv",
        results_dir="unused",
    )

    def resolve_universe(app):
        app.args.tickers = ["AAPL"]
        app.universe_definition = None
        app.universe_info = {}

    monkeypatch.setattr("core.main.ConfigManager.parseArgs", lambda: args)
    monkeypatch.setattr("core.main.MainApp._resolve_universe", resolve_universe)
    monkeypatch.setattr(
        "core.main.MainApp._set_all_seeds",
        lambda app, _seed=None: setattr(app, "current_seed", 42),
    )
    monkeypatch.setattr("core.main.Utils.resolve_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        "core.main.ExperimentStore",
        lambda root_dir: SimpleNamespace(root_dir=root_dir),
    )

    app = MainApp()

    assert app.frontendApp is None
    assert app.loader is None


def test_run_headless_uses_headless_evaluator_without_frontend(monkeypatch):
    app = object.__new__(MainApp)
    app.args = SimpleNamespace(
        tickers=["AAPL"],
        ablate_feature="none",
        price_provider="csv",
        prediction_window="1d",
        model="gru",
    )
    app.logger = logging.getLogger("test_gui_free_headless_run")
    app.frontendApp = None
    app.universe_definition = None
    app.experiment_store = SimpleNamespace(utc_now_iso=lambda: "2026-07-22T00:00:00Z")
    app.raw_feature_cols = []
    app._load_price_history = lambda *_args, **_kwargs: (
        {"AAPL": object()},
        SimpleNamespace(listTickers=lambda: ["AAPL"]),
    )

    observed = {}
    result = SimpleNamespace(model_name="GRU", direction="Upwards", confidence=51.0)

    class FakePipeline:
        def __init__(self, pipeline_app):
            observed["pipeline_app"] = pipeline_app

        def run(self, stock, window, stop_event=None):
            return {"target_stock": stock, "window": window}

    class FakeExperimentRunner:
        def __init__(self, runner_app):
            observed["runner_app"] = runner_app

        def run(self, **kwargs):
            observed["evaluator"] = kwargs["evaluator"]
            return result

    monkeypatch.setattr("core.main.Pipeline", FakePipeline)
    monkeypatch.setattr("core.main.ExperimentRunner", FakeExperimentRunner)
    app._store_run_result = lambda **kwargs: observed.setdefault("stored", kwargs)

    returned = app.run_headless(stock="AAPL", gui_window="1d", model_name="gru")

    assert returned is result
    assert observed["pipeline_app"] is app
    assert observed["runner_app"] is app
    assert isinstance(observed["evaluator"], HeadlessEvaluator)
    assert observed["stored"]["status"] == "success"


def test_experiment_runner_uses_gui_free_adapter_for_headless_main_app():
    args = SimpleNamespace(
        run_mode="headless",
        seed=42,
        deterministic=False,
        graph_ablation="none",
        ablate_feature="none",
        seq_len=10,
    )
    main_app = SimpleNamespace(
        args=args,
        device=SimpleNamespace(type="cpu"),
        frontendApp=None,
        _active_queue_job_id=None,
        raw_feature_cols=["close", "return", "volatility", "momentum"],
        raw_feature_dfs={"AAPL": object()},
    )
    state = {
        "train_feats": {},
        "val_feats": {},
        "test_feats": {},
        "tickers": ["AAPL"],
        "seq_len": 10,
        "horizon": 1,
    }

    runner = ExperimentRunner(main_app)
    adapter = runner._make_runner_app(
        state,
        stock="AAPL",
        force_headless=runner._should_use_headless_adapter(),
    )

    assert isinstance(adapter, HeadlessApp)
    assert isinstance(adapter.frontendApp, FrontendMock)


def test_headless_run_releases_resources_after_printing_result():
    app = object.__new__(MainApp)
    app.args = SimpleNamespace(
        run_mode="headless",
        target_stock="AAPL",
        prediction_window="1d",
        model="lstm",
        headless_report="compact",
    )
    app.logger = logging.getLogger("test_headless_run_cleanup")
    result = SimpleNamespace(model_name="LSTM", direction="Upwards", confidence=51.0)
    released = []

    app.run_headless = lambda **kwargs: result
    app._format_headless_result_line = lambda value: "result"
    app._release_headless_resources = lambda value: released.append(value)

    returned = app.run()

    assert returned is result
    assert released == [result]


def test_cli_headless_exit_happens_after_result_cleanup(monkeypatch):
    app = object.__new__(MainApp)
    app.args = SimpleNamespace(
        run_mode="headless",
        target_stock="AAPL",
        prediction_window="1d",
        model="lstm",
        headless_report="compact",
    )
    app.logger = logging.getLogger("test_cli_headless_exit_order")
    result = SimpleNamespace(model_name="LSTM", direction="Upwards", confidence=51.0)
    events = []

    app.run_headless = lambda **kwargs: result
    app._format_headless_result_line = lambda value: "result"
    app._release_headless_resources = lambda value: events.append("cleanup")
    monkeypatch.setattr("core.main._should_force_headless_cli_exit", lambda: True)
    monkeypatch.setattr(
        "core.main._finish_successful_windows_headless_process",
        lambda code: events.append(f"exit:{code}"),
    )

    returned = app.run()

    assert returned is result
    assert events == ["cleanup", "exit:0"]


def test_release_headless_resources_breaks_native_references(monkeypatch):
    app = object.__new__(MainApp)
    app.logger = logging.getLogger("test_release_headless_resources")
    closed = []

    class FakeModel:
        def __init__(self):
            self.moved_to_cpu = False

        def cpu(self):
            self.moved_to_cpu = True
            return self

    model = FakeModel()
    optimiser = SimpleNamespace(state={"tensor": object()})
    trainer = SimpleNamespace(
        optimiser=optimiser,
        criterion=object(),
        edge_index=object(),
        model=model,
    )
    result = SimpleNamespace(trainer=trainer, model=model)
    app.frontendApp = SimpleNamespace(
        _trainers={"lstm": trainer},
        _on_close=lambda: closed.append(True),
    )
    app.shutdown = lambda: None
    monkeypatch.setattr("core.main.torch.cuda.is_available", lambda: False)
    forced_collections = []
    monkeypatch.setattr("core.main.gc.collect", lambda: forced_collections.append(True))

    app._release_headless_resources(result)

    assert model.moved_to_cpu is True
    assert optimiser.state == {}
    assert trainer.model is None
    assert trainer.optimiser is None
    assert result.model is None
    assert result.trainer is None
    assert app.frontendApp._trainers == {}
    assert closed == [True]
    assert forced_collections == ([] if os.name == "nt" else [True])


def test_successful_windows_headless_exit_flushes_before_native_teardown(monkeypatch):
    events = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            events.append(f"flush:{self.name}")

    monkeypatch.setattr("core.main.logging.shutdown", lambda: events.append("logging"))
    monkeypatch.setattr("core.main.sys.stdout", FakeStream("stdout"))
    monkeypatch.setattr("core.main.sys.stderr", FakeStream("stderr"))
    monkeypatch.setattr("core.main.os._exit", lambda code: events.append(f"exit:{code}"))

    _finish_successful_windows_headless_process(0)

    assert events == ["logging", "flush:stdout", "flush:stderr", "exit:0"]


def test_compact_headless_result_is_one_machine_scannable_line():
    app = object.__new__(MainApp)
    app.args = SimpleNamespace(seed=42)
    result = SimpleNamespace(
        model_name="STGNN",
        direction="Upwards",
        confidence=50.11,
        metrics={
            "ticker": "AAPL",
            "threshold_operational": 0.500,
            "macro_f1_dense": 0.385,
            "macro_f1_dense_fixed_05": 0.379,
            "macro_f1_trade_aligned": 0.344,
            "sharpe": 1.230,
            "final_equity": 1.197,
        },
        eval_result=SimpleNamespace(
            metadata={
                "seed": 42,
                "graph_backend": "nnconv",
                "train_seconds": 4.40,
            }
        ),
    )

    line = app._format_headless_result_line(result)

    assert "\n" not in line
    assert "model=stgnn+nnconv" in line
    assert "macro_f1=0.385" in line
    assert "sharpe=1.230" in line
