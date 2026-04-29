"""Canonical model execution layer for GUI and headless runs."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from core.headless_app import HeadlessEvaluator, HeadlessStateAdapter
from models.registry import ModelRegistry


@dataclass
class ExperimentResult:
    model_name: str
    direction: str
    confidence: float
    metrics: Any = field(default_factory=dict)
    predictions: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    training_time_sec: float = 0.0
    eval_result: Optional[Any] = None
    trainer: Optional[Any] = None
    model: Optional[Any] = None
    extras: Optional[Dict[str, Any]] = None


class ExperimentRunner:
    def __init__(self, app=None, args=None, device: Optional[torch.device] = None):
        self.app = app
        self.logger = logging.getLogger(self.__class__.__name__)
        if app is not None:
            self.args = app.args
            self.device = app.device
            self.frontendApp = getattr(app, "frontendApp", None)
        else:
            self.args = args
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.frontendApp = None

    def run(self, model_name: str, stock: str, state: Dict[str, Any], evaluator=None, stop_event=None) -> ExperimentResult:
        if state is None:
            raise ValueError("ExperimentRunner.run requires a Pipeline state dictionary.")

        if self.args is None:
            self.args = state.get("args")
        if self.args is None and self.app is not None:
            self.args = self.app.args
        if self.args is None:
            raise ValueError("ExperimentRunner requires args. Pass app, args, or state['args'].")

        stock = str(stock or state.get("target_stock") or state.get("target_ticker")).strip().upper()
        self.logger.info("[ExperimentRunner] model=%s stock=%s device=%s", model_name, stock, self.device)

        runner = ModelRegistry.get_runner(model_name)
        use_headless_adapter = self._should_use_headless_adapter()
        runner_app = self._make_runner_app(state, stock=stock, force_headless=use_headless_adapter)
        price_df = self._resolve_price_df(stock, state, runner_app)

        if evaluator is None or use_headless_adapter:
            evaluator = HeadlessEvaluator()

        start_time = time.time()
        try:
            result = runner.run(
                app=runner_app,
                stock=stock,
                price_df=price_df,
                evaluator=evaluator,
                stop_event=stop_event,
            )
        except Exception:
            self.logger.exception("Experiment failed: %s", model_name)
            raise

        return self._normalise_result(model_name, result, time.time() - start_time)

    def _should_use_headless_adapter(self) -> bool:
        if self.app is None:
            return True
        if getattr(self.app, "_active_queue_job_id", None) is not None:
            return True
        return threading.current_thread() is not threading.main_thread()

    def _make_runner_app(self, state: Dict[str, Any], stock: str, force_headless: bool = False):
        if self.app is None or force_headless:
            adapter_state = dict(state)
            if self.app is not None:
                adapter_state.setdefault("raw_feature_cols", getattr(self.app, "raw_feature_cols", None))
                adapter_state.setdefault("raw_feature_dfs", getattr(self.app, "raw_feature_dfs", None))
                adapter_state.setdefault("args", getattr(self.app, "args", self.args))
            adapter_state.setdefault("target_stock", stock)
            adapter_state.setdefault("target_ticker", stock)
            adapter_state.setdefault("graph_window", int(adapter_state.get("seq_len", getattr(self.args, "seq_len", 10))))
            return HeadlessStateAdapter.create_app(args=self.args, state=adapter_state, device=self.device)

        self._overlay_state_on_app(self.app, state)
        return self.app

    def _overlay_state_on_app(self, app, state: Dict[str, Any]) -> None:
        app.train_feats = state.get("train_feats", {})
        app.val_feats = state.get("val_feats", {})
        app.tf_train = state.get("tf_train")
        app.tf_val = state.get("tf_val")
        app.graphBuilder = state.get("graphBuilder")
        app.edge_index = state.get("edge_index")
        app.init_edge_index = state.get("edge_index")
        app.tickers = state.get("tickers", getattr(app.args, "tickers", []))
        app.coords = state.get("coords")
        app.pruned = state.get("pruned")
        app.mst = state.get("mst")
        app.horizon = state.get("horizon")
        app.seq_len = state.get("seq_len")
        app.graph_window = int(state.get("graph_window", app.seq_len))

        app.target_stock = state.get("target_stock") or state.get("target_ticker")
        app.target_ticker = app.target_stock
        app.target_source = state.get("target_source", "unknown")

        if "raw_feature_cols" in state:
            app.raw_feature_cols = state["raw_feature_cols"]

        if app.train_feats:
            app.min_train_len = min(len(df) for df in app.train_feats.values())
        if app.val_feats:
            app.min_val_len = min(len(df) for df in app.val_feats.values())

        if not hasattr(app, "dl_gen") or app.dl_gen is None:
            seed = int(getattr(app.args, "seed", getattr(app, "current_seed", 42)))
            app.current_seed = seed
            app.dl_gen = torch.Generator()
            app.dl_gen.manual_seed(seed)

        if hasattr(app, "args"):
            if app.tickers:
                app.args.tickers = list(app.tickers)
            app.args.graph_window = app.graph_window

    def _resolve_price_df(self, stock: str, state: Dict[str, Any], runner_app):
        raw = state.get("raw_feature_dfs") or getattr(runner_app, "raw_feature_dfs", None)
        if isinstance(raw, dict) and stock in raw:
            return raw[stock]
        price_history = getattr(runner_app, "priceHistory", None)
        if isinstance(price_history, dict) and stock in price_history:
            return price_history[stock]
        val_feats = state.get("val_feats") or getattr(runner_app, "val_feats", None)
        if isinstance(val_feats, dict) and stock in val_feats:
            return val_feats[stock]
        train_feats = state.get("train_feats") or getattr(runner_app, "train_feats", None)
        if isinstance(train_feats, dict) and stock in train_feats:
            return train_feats[stock]
        return None

    def _normalise_result(self, model_name: str, result: Any, training_time_sec: float) -> ExperimentResult:
        if hasattr(result, "model_name"):
            return ExperimentResult(
                model_name=getattr(result, "model_name", model_name),
                direction=getattr(result, "direction", "unknown"),
                confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                metrics=getattr(result, "metrics", {}) or {},
                predictions=getattr(result, "predictions", None),
                probabilities=getattr(result, "probabilities", None),
                training_time_sec=float(training_time_sec),
                eval_result=getattr(result, "eval_result", None),
                trainer=getattr(result, "trainer", None),
                model=getattr(result, "model", None),
                extras=getattr(result, "extras", None),
            )
        if isinstance(result, (tuple, list)):
            return ExperimentResult(
                model_name=str(result[0]) if result else model_name,
                direction=str(result[1]) if len(result) > 1 else "unknown",
                confidence=float(result[2]) if len(result) > 2 else 0.0,
                training_time_sec=float(training_time_sec),
            )
        return ExperimentResult(model_name=model_name, direction="unknown", confidence=0.0, training_time_sec=float(training_time_sec), extras={"raw_result": repr(result)})

    @staticmethod
    def run_headless(model_name: str, state: Dict[str, Any], args, device: Optional[torch.device] = None, stock: Optional[str] = None, evaluator=None) -> ExperimentResult:
        target = stock or state.get("target_stock") or state.get("target_ticker")
        if not target:
            tickers = state.get("tickers") or getattr(args, "tickers", [])
            target = tickers[0] if tickers else None
        if not target:
            raise ValueError("No target stock supplied for headless run.")
        runner = ExperimentRunner(app=None, args=args, device=device)
        return runner.run(model_name=model_name, stock=target, state=state, evaluator=evaluator, stop_event=None)
