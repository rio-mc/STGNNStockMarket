from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pandas as pd

from evaluation.evaluation_types import EvaluationResult

from ..base_runner import BaseModelRunner, ModelRunResult


class ARIMARunner(BaseModelRunner):
    model_name = "ARIMA"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Fitting ARIMA baseline...")

        train_ds, val_ds, train_df_stock, val_df_stock = self._build_recurrent_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, getattr(train_ds, "aligned_tickers", [stock]))

        order = self._resolve_order(app)
        train_close = self._close_series(train_df_stock)
        val_close = self._close_series(val_df_stock)
        scale = self._return_scale(train_close)

        app.logger.info("[ARIMARunner] order=%s | train=%d | val_windows=%d", order, len(train_close), len(val_ds))

        start = time.time()
        eval_result = self._evaluate_rolling(
            app=app,
            train_close=train_close,
            val_close=val_close,
            val_ds=val_ds,
            order=order,
            horizon=int(app.horizon),
            scale=scale,
            stop_event=stop_event,
        )
        train_seconds = time.time() - start

        self._attach_metadata(app, eval_result, stock=stock)
        eval_result.metadata.update({
            "arima_order": order,
            "energy_wh": 0.0,
            "train_seconds": train_seconds,
            "avg_power_w": 0.0,
            "energy_per_sample_wh": 0.0,
            "train_samples": len(train_close),
            "gpu_peak_memory_mb": None,
        })

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)
        app.frontendApp.set_status("Predicting with ARIMA baseline...")

        live_prob = self._live_probability(
            train_close=train_close,
            val_close=val_close,
            order=order,
            horizon=int(app.horizon),
            scale=scale,
        )
        threshold = self._resolve_threshold(
            metrics,
            policy=getattr(app.args, "decision_threshold_policy", "fixed"),
        )
        direction, confidence = self._direction_from_probability(live_prob, threshold)

        app.logger.info("[ARIMARunner] %s (%.1f%%)", direction, confidence)

        self._cleanup(train_ds, val_ds)
        return ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=None,
            model=None,
            extras={"arima_order": order},
        )

    def _evaluate_rolling(
        self,
        *,
        app,
        train_close: pd.Series,
        val_close: pd.Series,
        val_ds,
        order: tuple[int, int, int],
        horizon: int,
        scale: float,
        stop_event,
    ) -> EvaluationResult:
        y_true_all = []
        y_pred_all = []
        probs_all = []
        prediction_dates = []
        hist_val = []

        for i in range(len(val_ds)):
            app._check_stop(stop_event)

            history_end = i + int(app.seq_len)
            current_idx = history_end - 1
            if current_idx >= len(val_close):
                break

            history = pd.concat([train_close, val_close.iloc[:history_end]])
            current_price = float(val_close.iloc[current_idx])
            forecast = self._forecast(history, order=order, horizon=horizon)
            prob = self._probability_from_forecast(
                forecast=forecast,
                current_price=current_price,
                scale=scale,
            )

            _x, y = val_ds[i]
            truth = int(float(y.item()) >= 0.5)
            pred = int(prob >= 0.5)
            timestamp = pd.Timestamp(val_ds.get_timestamp(i)).tz_localize(None)
            loss = self._binary_cross_entropy(prob, truth)

            y_true_all.append(truth)
            y_pred_all.append(pred)
            probs_all.append(prob)
            prediction_dates.append(timestamp)
            hist_val.append({"date": timestamp, "loss": loss})

            if hasattr(app.frontendApp, "updateProgress"):
                app.frontendApp.updateProgress((i + 1) / max(len(val_ds), 1))

        mean_val_loss = float(np.mean([row["loss"] for row in hist_val])) if hist_val else float("nan")

        return EvaluationResult(
            y_true=y_true_all,
            y_pred=y_pred_all,
            probs=probs_all,
            prediction_dates=prediction_dates,
            decision_threshold=0.5,
            dense_val_loss=mean_val_loss,
            hist_train=[],
            hist_val=hist_val,
            horizon=horizon,
            model_name=self.model_name,
            metadata={
                "evaluation_mode": "dense_rolling",
                "decision_threshold_policy": "fixed",
                "ticker": app.args.target_stock if hasattr(app.args, "target_stock") else None,
            },
        )

    def _live_probability(
        self,
        *,
        train_close: pd.Series,
        val_close: pd.Series,
        order: tuple[int, int, int],
        horizon: int,
        scale: float,
    ) -> float:
        history = pd.concat([train_close, val_close])
        current_price = float(history.iloc[-1])
        forecast = self._forecast(history, order=order, horizon=horizon)
        return self._probability_from_forecast(
            forecast=forecast,
            current_price=current_price,
            scale=scale,
        )

    def _forecast(self, series: pd.Series, *, order: tuple[int, int, int], horizon: int) -> float:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise RuntimeError(
                "ARIMA baseline requires statsmodels. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc

        values = pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) <= max(sum(order), horizon, 3):
            return float(values.iloc[-1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted = ARIMA(
                    values,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                forecast = fitted.forecast(steps=max(int(horizon), 1))
                return float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1])
            except Exception:
                return float(values.iloc[-1])

    def _resolve_order(self, app) -> tuple[int, int, int]:
        order = getattr(app.args, "arima_order", None)
        if order:
            parts = [int(part.strip()) for part in str(order).split(",")]
            if len(parts) != 3 or any(part < 0 for part in parts):
                raise ValueError("--arima_order must have the form p,d,q with non-negative integer values")
            return tuple(parts)

        resolved = (
            int(getattr(app.args, "arima_p", 1)),
            int(getattr(app.args, "arima_d", 1)),
            int(getattr(app.args, "arima_q", 1)),
        )
        if min(resolved) < 0:
            raise ValueError("ARIMA order values must be non-negative")
        return resolved

    @staticmethod
    def _close_series(df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("ARIMA baseline requires a 'close' column.")
        series = df["close"].copy()
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        if getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_localize(None)
        return series.astype("float64")

    @staticmethod
    def _return_scale(series: pd.Series) -> float:
        diffs = series.astype("float64").diff().replace([np.inf, -np.inf], np.nan).dropna()
        scale = float(diffs.std(ddof=0)) if not diffs.empty else 0.0
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        return max(scale, 1e-6)

    @staticmethod
    def _probability_from_forecast(*, forecast: float, current_price: float, scale: float) -> float:
        if not np.isfinite(forecast) or not np.isfinite(current_price):
            return 0.5
        score = forecast - current_price
        z = max(min(score / max(scale, 1e-6), 30.0), -30.0)
        return float(1.0 / (1.0 + math.exp(-z)))

    @staticmethod
    def _binary_cross_entropy(prob: float, truth: int) -> float:
        p = min(max(float(prob), 1e-7), 1.0 - 1e-7)
        y = int(truth)
        return float(-(y * math.log(p) + (1 - y) * math.log(1.0 - p)))
