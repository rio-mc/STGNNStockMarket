"""
Headless App Wrapper

Provides a MainApp-compatible interface for headless experiment execution.
This allows model runners to work without the GUI dependency.
"""

import logging
import random
from typing import Any, Dict

import numpy as np
import torch

from types import SimpleNamespace
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

class HeadlessEvaluator:
    """
    Lightweight evaluator for CLI/headless runs.

    Computes dense predictive metrics and trade-aligned backtest metrics without
    touching Tkinter.
    """

    def evaluate(self, model_name: str, result, price_df=None):
        from backtesting.engine import BacktestEngine

        y_true = np.asarray(getattr(result, "y_true", []) or [], dtype=int)
        probs = np.asarray(getattr(result, "probs", []) or [], dtype=float)

        ticker = (
            getattr(result, "ticker", None)
            or getattr(result, "target_stock", None)
            or getattr(result, "target_ticker", None)
        )

        horizon = getattr(result, "horizon", None)

        empty = self._empty_metrics(
            model_name=model_name,
            ticker=ticker,
            horizon=horizon,
            val_loss_dense=getattr(result, "dense_val_loss", None),
        )

        if y_true.size == 0 or probs.size == 0:
            return empty

        fixed_thr = float(getattr(result, "decision_threshold", 0.5) or 0.5)
        dense_pred = (probs >= fixed_thr).astype(int)

        best_thr = self._best_threshold_macro_f1(y_true, probs)
        tuned_pred = (probs >= best_thr).astype(int)

        roc_auc = None
        ap = None
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, probs))
            ap = float(average_precision_score(y_true, probs))

        _, _, f1_pos, _ = precision_recall_fscore_support(
            y_true,
            dense_pred,
            average="binary",
            zero_division=0,
        )

        _, _, tuned_f1_pos, _ = precision_recall_fscore_support(
            y_true,
            tuned_pred,
            average="binary",
            zero_division=0,
        )

        result.y_pred = dense_pred.astype(int).tolist()
        result.metadata["decision_threshold_policy"] = "fixed"

        trade_metrics = self._compute_backtest_metrics(
            model_name=model_name,
            result=result,
            price_df=price_df,
            y_true=y_true,
            probs=probs,
            dense_thr=fixed_thr,
        )

        return SimpleNamespace(
            model=str(model_name).upper(),

            threshold_fixed=fixed_thr,
            threshold_macro_f1_dense=best_thr,
            val_loss_dense=getattr(result, "dense_val_loss", None),

            accuracy_dense=float(accuracy_score(y_true, dense_pred)),
            f1_dense=float(f1_pos),
            macro_f1_dense=float(f1_score(y_true, dense_pred, average="macro")),
            roc_auc_dense=roc_auc,
            ap_dense=ap,

            accuracy_dense_macro_f1_threshold=float(accuracy_score(y_true, tuned_pred)),
            f1_dense_macro_f1_threshold=float(tuned_f1_pos),
            macro_f1_dense_macro_f1_threshold=float(
                f1_score(y_true, tuned_pred, average="macro")
            ),

            accuracy_trade_aligned=trade_metrics["accuracy_trade_aligned"],
            f1_trade_aligned=trade_metrics["f1_trade_aligned"],
            macro_f1_trade_aligned=trade_metrics["macro_f1_trade_aligned"],
            roc_auc_trade_aligned=trade_metrics["roc_auc_trade_aligned"],
            ap_trade_aligned=trade_metrics["ap_trade_aligned"],

            sharpe=trade_metrics["sharpe"],
            n_trades=trade_metrics["n_trades"],
            mean_trade_return=trade_metrics["mean_trade_return"],
            hit_rate=trade_metrics["hit_rate"],
            final_equity=trade_metrics["final_equity"],
            max_drawdown=trade_metrics["max_drawdown"],

            ticker=ticker,
            n_predictions_dense=int(len(y_true)),
            n_predictions_trade_aligned=trade_metrics["n_predictions_trade_aligned"],
            horizon=horizon,
        )

    def _compute_backtest_metrics(
        self,
        *,
        model_name: str,
        result,
        price_df,
        y_true,
        probs,
        dense_thr: float,
    ) -> dict:
        from backtesting.engine import BacktestEngine

        defaults = {
            "accuracy_trade_aligned": 0.0,
            "f1_trade_aligned": 0.0,
            "macro_f1_trade_aligned": 0.0,
            "roc_auc_trade_aligned": None,
            "ap_trade_aligned": None,
            "sharpe": None,
            "n_trades": 0,
            "mean_trade_return": None,
            "hit_rate": None,
            "final_equity": None,
            "max_drawdown": None,
            "n_predictions_trade_aligned": 0,
        }

        if price_df is None:
            return defaults

        if getattr(result, "horizon", None) is None:
            return defaults

        if not getattr(result, "prediction_dates", None):
            return defaults

        backtest = BacktestEngine(tc_per_side=0.0, rf_annual=0.0).run(
            model_name=model_name,
            evaluation_result=result,
            price_df=price_df,
            threshold=dense_thr,
        )

        if backtest is None:
            return defaults

        exec_idx = list(getattr(backtest, "executed_indices", []) or [])
        result.trade_aligned_indices = exec_idx

        metrics = {
            **defaults,
            "sharpe": getattr(backtest, "sharpe", None),
            "n_trades": int(getattr(backtest, "n_trades", 0)),
            "mean_trade_return": getattr(backtest, "mean_trade_return", None),
            "hit_rate": getattr(backtest, "hit_rate", None),
            "final_equity": getattr(backtest, "final_equity", None),
            "max_drawdown": getattr(backtest, "max_drawdown", None),
            "n_predictions_trade_aligned": int(len(exec_idx)),
        }

        if not exec_idx:
            return metrics

        y_true_trade = np.asarray(y_true)[exec_idx]
        probs_trade = np.asarray(probs)[exec_idx]
        y_pred_trade = (probs_trade >= dense_thr).astype(int)

        metrics["accuracy_trade_aligned"] = float(
            accuracy_score(y_true_trade, y_pred_trade)
        )
        _, _, f1_trade, _ = precision_recall_fscore_support(
            y_true_trade,
            y_pred_trade,
            average="binary",
            zero_division=0,
        )
        metrics["f1_trade_aligned"] = float(f1_trade)
        metrics["macro_f1_trade_aligned"] = float(
            f1_score(y_true_trade, y_pred_trade, average="macro")
        )

        if len(np.unique(y_true_trade)) > 1:
            metrics["roc_auc_trade_aligned"] = float(
                roc_auc_score(y_true_trade, probs_trade)
            )
            metrics["ap_trade_aligned"] = float(
                average_precision_score(y_true_trade, probs_trade)
            )

        return metrics

    @staticmethod
    def _empty_metrics(model_name, ticker=None, horizon=None, val_loss_dense=None):
        return SimpleNamespace(
            model=str(model_name).upper(),
            threshold_fixed=0.5,
            threshold_macro_f1_dense=0.5,
            val_loss_dense=val_loss_dense,

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

            ticker=ticker,
            n_predictions_dense=0,
            n_predictions_trade_aligned=0,
            horizon=horizon,
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
    """
    A minimal MainApp-compatible wrapper for headless execution.
    
    This class provides the interface that model runners expect,
    using data from the Pipeline state instead of GUI state.
    """
    
    def __init__(
        self,
        args,
        state: Dict[str, Any],
        device: torch.device,
        raw_feature_cols: list = None,
    ):
        """
        Initialize the headless app wrapper.
        
        Args:
            args: Configuration namespace (from argparse)
            state: Pipeline state dictionary
            device: Torch device for computation
            raw_feature_cols: Feature columns to use
        """
        self.args = args
        self.state = state
        self.device = device
        
        # Set up logger
        self.logger = logging.getLogger("HeadlessApp")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Extract state components
        self.train_feats = state.get("train_feats", {})
        self.val_feats = state.get("val_feats", {})
        self.edge_index = state.get("edge_index", None)
        self.tickers = state.get("tickers", [])
        self.graphBuilder = state.get("graphBuilder", None)
        self.init_edge_index = self.edge_index  # For graph ablation
        self.horizon = state.get("horizon", 1)
        self.seq_len = state.get("seq_len", 10)
        
        # Feature columns
        self.raw_feature_cols = raw_feature_cols or ["close", "return", "volatility", "momentum"]
        
        # Calculate min lengths
        if self.train_feats:
            self.min_train_len = min(len(df) for df in self.train_feats.values())
        else:
            self.min_train_len = 0
            
        if self.val_feats:
            self.min_val_len = min(len(df) for df in self.val_feats.values())
        else:
            self.min_val_len = 0
        
        # Set up random seed
        self._setup_seed()
        
        # Set up data loader generator
        self._setup_dl_gen()
        
        # Frontend app mock (for status updates)
        self.frontendApp = FrontendMock()
        
        # Graph ablation mode
        self.graph_ablation_mode = getattr(args, "graph_ablation", "none")
        
        # Feature ablation
        self.ablate_feature = getattr(args, "ablate_feature", "none")
    
    def _setup_seed(self):
        """Set up random seeds for reproducibility."""
        seed = getattr(self.args, "seed", 42)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if getattr(self.device, "type", None) == "cuda":
            torch.cuda.manual_seed_all(seed)
            # Deterministic settings
            if getattr(self.args, "deterministic", False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
    
    def _setup_dl_gen(self):
        """Set up DataLoader random generator."""
        seed = getattr(self.args, "seed", 42)
        self.dl_gen = torch.Generator()
        self.dl_gen.manual_seed(seed)
    
    @staticmethod
    def _seed_worker(worker_id):
        """Seed worker for DataLoader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def _check_stop(self, stop_event):
        """Check if stop was requested (for GUI compatibility)."""
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stop requested")


class FrontendMock:
    """No-op frontend used by Trainer and runners during headless execution."""

    def set_status(self, message: str):
        pass

    def updateProgress(self, progress: float):
        pass

    def update_progress(self, current: int, total=None):
        pass

    def ui_call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def refresh_selected_tabs(self):
        pass


class HeadlessStateAdapter:
    """
    Adapter that converts Pipeline state to MainApp-compatible interface.
    
    This is used by ExperimentRunner to wrap state before passing to model runners.
    """
    
    @staticmethod
    def create_app(args, state: Dict[str, Any], device: torch.device) -> HeadlessApp:
        """
        Create a HeadlessApp from pipeline state.
        
        Args:
            args: Configuration namespace
            state: Pipeline state dictionary
            device: Torch device
            
        Returns:
            HeadlessApp instance
        """
        raw_feature_cols = state.get("raw_feature_cols", ["close", "return", "volatility", "momentum"])
        
        return HeadlessApp(
            args=args,
            state=state,
            device=device,
            raw_feature_cols=raw_feature_cols,
        )
