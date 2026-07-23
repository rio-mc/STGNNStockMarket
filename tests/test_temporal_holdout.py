import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from core.headless_app import HeadlessEvaluator
from core.pipeline import Pipeline
from evaluation.evaluation_methods import EvaluationMethods
from config.config_manager import ConfigManager
from evaluation.evaluation_types import EvaluationMetrics, EvaluationResult
from models.runners.arima_runner import ARIMARunner


class TemporalHoldoutTests(unittest.TestCase):
    def test_split_ratio_defaults_and_validation(self):
        with patch.object(sys, "argv", ["prog", "--run_mode", "headless"]):
            args = ConfigManager.parseArgs()
        self.assertEqual(args.train_split_ratio, 0.6)
        self.assertEqual(args.validation_split_ratio, 0.2)
        self.assertEqual(args.decision_threshold_policy, "macro_f1_dense")

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--run_mode",
                "headless",
                "--train_split_ratio",
                "0.8",
                "--validation_split_ratio",
                "0.2",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "held-out test split"):
                ConfigManager.parseArgs()

    @staticmethod
    def _raw_prices(periods=300):
        index = pd.date_range("2024-01-01", periods=periods, freq="D")
        raw = {}
        for offset, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
            close = (
                100
                + offset
                + np.linspace(0, 10, periods)
                + np.sin(np.arange(periods) / (5 + offset))
            )
            raw[ticker] = pd.DataFrame(
                {
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + np.arange(periods) + offset,
                },
                index=index,
            )
        return raw

    @staticmethod
    def _args(tickers):
        return SimpleNamespace(
            seq_len=10,
            train_split_ratio=0.6,
            validation_split_ratio=0.2,
            ablate_feature="none",
            graph_embed="pca",
            graph_mode="knn_mst",
            graph_ablation="none",
            k=2,
            model="lstm",
            seed=42,
            results_dir="results",
            tickers=list(tickers),
            ticker_to_sector={ticker: "Synthetic" for ticker in tickers},
        )

    def test_pipeline_builds_two_purged_boundaries(self):
        raw = self._raw_prices()
        args = self._args(raw)

        with patch("graph.graph_builder.GraphBuilder.save_graph_stats"):
            state = Pipeline(args, raw_feature_dfs=raw).run("AAA", "1d")

        quality = state["data_quality"]
        train_end = pd.Timestamp(quality["train_end_date"])
        validation_start = pd.Timestamp(quality["validation_start_date"])
        validation_end = pd.Timestamp(quality["validation_end_date"])
        test_start = pd.Timestamp(quality["test_start_date"])

        self.assertLess(train_end, validation_start)
        self.assertLess(validation_end, test_start)
        self.assertEqual(quality["embargo_bars"], state["horizon"])
        shared_index = raw["AAA"].index
        self.assertEqual(
            shared_index.get_loc(validation_start) - shared_index.get_loc(train_end),
            quality["embargo_bars"],
        )
        self.assertEqual(
            shared_index.get_loc(test_start) - shared_index.get_loc(validation_end),
            quality["embargo_bars"],
        )
        self.assertTrue(state["train_feats"])
        self.assertTrue(state["val_feats"])
        self.assertTrue(state["test_feats"])
        self.assertLess(
            max(df.index.max() for df in state["val_feats"].values()),
            test_start,
        )
        self.assertGreaterEqual(
            min(df.index.min() for df in state["test_feats"].values()),
            test_start,
        )

    def test_evaluator_uses_validation_threshold_without_test_retuning(self):
        validation_threshold = 0.73
        result = EvaluationResult(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 0, 0, 1],
            probs=[0.10, 0.20, 0.60, 0.90],
            prediction_dates=list(pd.date_range("2025-01-01", periods=4, freq="D")),
            decision_threshold=validation_threshold,
            dense_val_loss=0.8,
            metadata={
                "evaluation_split": "test",
                "threshold_source": "validation",
                "threshold_macro_f1_validation": validation_threshold,
                "validation_loss_dense": 0.6,
                "test_loss_dense": 0.8,
            },
        )

        metrics = HeadlessEvaluator().evaluate("LSTM", result, price_df=None)

        self.assertEqual(metrics.threshold_fixed, validation_threshold)
        self.assertEqual(metrics.threshold_operational, validation_threshold)
        self.assertEqual(metrics.threshold_fixed_05, 0.5)
        self.assertEqual(metrics.threshold_macro_f1_dense, validation_threshold)
        self.assertEqual(metrics.val_loss_dense, 0.6)
        self.assertEqual(metrics.test_loss_dense, 0.8)
        self.assertEqual(metrics.threshold_source, "validation")
        self.assertEqual(metrics.evaluation_split, "test")

        gui_metrics = EvaluationMethods.compute_dense_metrics(
            object.__new__(EvaluationMethods),
            result,
        )
        self.assertEqual(
            gui_metrics["macro_f1_optimal_threshold"],
            validation_threshold,
        )
        fixed_predictions = (np.asarray(result.probs) >= 0.5).astype(int)
        expected_fixed_macro_f1 = f1_score(
            np.asarray(result.y_true),
            fixed_predictions,
            average="macro",
        )
        self.assertEqual(metrics.macro_f1_dense_fixed_05, expected_fixed_macro_f1)
        self.assertEqual(
            gui_metrics["fixed_05_metrics"]["macro_f1"],
            expected_fixed_macro_f1,
        )

        summary = EvaluationMethods.build_metrics_summary(
            object.__new__(EvaluationMethods),
            model_name="LSTM",
            result=result,
            price_df=None,
            dense_thr=gui_metrics["dense_thr"],
            macro_f1_optimal_threshold=gui_metrics["macro_f1_optimal_threshold"],
            dense_metrics=gui_metrics["dense_metrics"],
            fixed_05_metrics=gui_metrics["fixed_05_metrics"],
            macro_f1_optimised_metrics=gui_metrics["macro_f1_optimised_metrics"],
            trade_metrics={
                "accuracy": 0.0,
                "f1": 0.0,
                "macro_f1": 0.0,
                "roc_auc": None,
                "ap": None,
            },
            strategy_metrics={
                "sharpe": 0.0,
                "mean_trade_return": 0.0,
                "hit_rate": 0.0,
                "final_equity": None,
                "max_drawdown": None,
            },
        )
        self.assertIsInstance(summary, EvaluationMetrics)
        self.assertEqual(summary.threshold_operational, validation_threshold)
        self.assertEqual(summary.threshold_fixed_05, 0.5)

    def test_macro_f1_policy_selects_validation_threshold(self):
        validation = EvaluationResult(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 0, 0, 0],
            probs=[0.40, 0.45, 0.46, 0.47],
            prediction_dates=list(pd.date_range("2025-01-01", periods=4, freq="D")),
        )
        app = SimpleNamespace(
            args=SimpleNamespace(decision_threshold_policy="macro_f1_dense")
        )

        selection = ARIMARunner()._calibrate_threshold(app, validation)

        self.assertEqual(selection["split"], "validation")
        self.assertEqual(selection["policy"], "macro_f1_dense")
        self.assertNotEqual(selection["selected_threshold"], 0.5)
        self.assertEqual(
            selection["selected_threshold"],
            selection["macro_f1_threshold"],
        )


if __name__ == "__main__":
    unittest.main()
