"""
Headless App Adapter

Provides the minimal MainApp-shaped interface required by the current model
runners, without starting Tkinter or depending on GUI state.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


class FrontendMock:
    def set_status(self, message: str):
        pass

    def updateProgress(self, progress: float):
        pass

    def update_progress(self, current: int, total: Optional[int] = None):
        pass

    def ui_call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def refresh_selected_tabs(self):
        pass


class HeadlessEvaluator:
    """Lightweight no-UI evaluator for CLI and GUI queue runs."""

    def evaluate(self, model_name: str, result, price_df=None):
        y_true = np.asarray(getattr(result, "y_true", []) or [], dtype=int)
        probs = np.asarray(getattr(result, "probs", []) or [], dtype=float)

        if y_true.size == 0 or probs.size == 0:
            return SimpleNamespace(
                model=str(model_name).upper(),
                threshold_fixed=0.5,
                threshold_macro_f1_dense=0.5,
                val_loss_dense=getattr(result, "dense_val_loss", None),
                accuracy_dense=0.0,
                f1_dense=0.0,
                macro_f1_dense=0.0,
                roc_auc_dense=None,
                ap_dense=None,
                accuracy_dense_macro_f1_threshold=0.0,
                f1_dense_macro_f1_threshold=0.0,
                macro_f1_dense_macro_f1_threshold=0.0,
                accuracy_trade_aligned=0.0,
                f1_trade_aligned=0.0,
                macro_f1_trade_aligned=0.0,
                roc_auc_trade_aligned=None,
                ap_trade_aligned=None,
                sharpe=None,
                n_trades=0,
                mean_trade_return=None,
                hit_rate=None,
                final_equity=None,
                max_drawdown=None,
                ticker=None,
                n_predictions_dense=0,
                n_predictions_trade_aligned=0,
                horizon=getattr(result, "horizon", None),
            )

        fixed_thr = float(getattr(result, "decision_threshold", 0.5) or 0.5)
        dense_pred = (probs >= fixed_thr).astype(int)
        best_thr = self._best_threshold_macro_f1(y_true, probs)
        tuned_pred = (probs >= best_thr).astype(int)

        roc_auc = None
        ap = None
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, probs))
            ap = float(average_precision_score(y_true, probs))

        _, _, f1, _ = precision_recall_fscore_support(
            y_true, dense_pred, average="binary", zero_division=0
        )
        _, _, tuned_f1, _ = precision_recall_fscore_support(
            y_true, tuned_pred, average="binary", zero_division=0
        )

        result.y_pred = dense_pred.astype(int).tolist()
        result.metadata["decision_threshold_policy"] = "fixed"

        return SimpleNamespace(
            model=str(model_name).upper(),
            threshold_fixed=fixed_thr,
            threshold_macro_f1_dense=best_thr,
            val_loss_dense=getattr(result, "dense_val_loss", None),
            accuracy_dense=float(accuracy_score(y_true, dense_pred)),
            f1_dense=float(f1),
            macro_f1_dense=float(f1_score(y_true, dense_pred, average="macro")),
            roc_auc_dense=roc_auc,
            ap_dense=ap,
            accuracy_dense_macro_f1_threshold=float(accuracy_score(y_true, tuned_pred)),
            f1_dense_macro_f1_threshold=float(tuned_f1),
            macro_f1_dense_macro_f1_threshold=float(f1_score(y_true, tuned_pred, average="macro")),
            accuracy_trade_aligned=0.0,
            f1_trade_aligned=0.0,
            macro_f1_trade_aligned=0.0,
            roc_auc_trade_aligned=None,
            ap_trade_aligned=None,
            sharpe=None,
            n_trades=0,
            mean_trade_return=None,
            hit_rate=None,
            final_equity=None,
            max_drawdown=None,
            ticker=None,
            n_predictions_dense=int(len(y_true)),
            n_predictions_trade_aligned=0,
            horizon=getattr(result, "horizon", None),
        )

    @staticmethod
    def _best_threshold_macro_f1(y_true, probs) -> float:
        thresholds = np.unique(probs)
        if thresholds.size == 0:
            return 0.5
        thresholds = np.concatenate(([0.0], thresholds, [1.0]))

        best_thr = 0.5
        best_score = -1.0
        for thr in thresholds:
            pred = (probs >= thr).astype(int)
            score = f1_score(y_true, pred, average="macro")
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        return best_thr


class HeadlessApp:
    """Minimal MainApp-compatible wrapper for current model runners."""

    def __init__(self, args, state: Dict[str, Any], device: torch.device, raw_feature_cols=None):
        self.args = args
        self.state = state
        self.device = device

        self.logger = logging.getLogger("HeadlessApp")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.train_feats = state.get("train_feats", {})
        self.val_feats = state.get("val_feats", {})
        self.raw_feature_dfs = state.get("raw_feature_dfs", {})
        self.tf_train = state.get("tf_train")
        self.tf_val = state.get("tf_val")
        self.graphBuilder = state.get("graphBuilder")
        self.edge_index = state.get("edge_index")
        self.init_edge_index = self.edge_index
        self.tickers = list(state.get("tickers", getattr(args, "tickers", [])))
        self.coords = state.get("coords")
        self.pruned = state.get("pruned")
        self.mst = state.get("mst")
        self.horizon = int(state.get("horizon", 1))
        self.seq_len = int(state.get("seq_len", getattr(args, "seq_len", 10)))
        self.graph_window = int(state.get("graph_window", self.seq_len))

        self.target_stock = state.get("target_stock") or state.get("target_ticker")
        self.target_ticker = self.target_stock
        self.target_source = state.get("target_source", "unknown")

        self.raw_feature_cols = raw_feature_cols or state.get(
            "raw_feature_cols",
            ["close", "return", "volatility", "momentum"],
        )

        self.min_train_len = min((len(df) for df in self.train_feats.values()), default=0)
        self.min_val_len = min((len(df) for df in self.val_feats.values()), default=0)

        if self.tickers:
            self.args.tickers = list(self.tickers)
        self.args.graph_window = self.graph_window

        self.frontendApp = FrontendMock()
        self.graph_ablation_mode = getattr(args, "graph_ablation", "none")
        self.ablate_feature = getattr(args, "ablate_feature", "none")

        self._setup_seed()
        self._setup_dl_gen()

    def _setup_seed(self):
        seed = int(getattr(self.args, "seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if getattr(self.args, "deterministic", False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)

    def _setup_dl_gen(self):
        seed = int(getattr(self.args, "seed", 42))
        self.dl_gen = torch.Generator()
        self.dl_gen.manual_seed(seed)

    @staticmethod
    def _seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _check_stop(self, stop_event):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Pipeline interrupted")

    def get_max_k(self):
        return int(getattr(self.args, "k", 0))


class HeadlessStateAdapter:
    @staticmethod
    def create_app(args, state: Dict[str, Any], device: torch.device) -> HeadlessApp:
        return HeadlessApp(
            args=args,
            state=state,
            device=device,
            raw_feature_cols=state.get("raw_feature_cols"),
        )
