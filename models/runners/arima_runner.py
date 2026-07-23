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

        train_ds, val_ds, test_ds, train_df_stock, val_df_stock, test_df_stock = self._build_recurrent_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, test_ds, getattr(train_ds, "aligned_tickers", [stock]))

        order = self._resolve_order(app)
        refit_interval = self._resolve_refit_interval(app)
        train_close = self._close_series(train_df_stock)
        val_close = self._close_series(val_df_stock)
        test_close = self._close_series(test_df_stock)
        scale = self._return_scale(train_close)

        app.logger.info(
            "[ARIMARunner] order=%s | train=%d | val_windows=%d | refit_interval=%d",
            order,
            len(train_close),
            len(val_ds),
            refit_interval,
        )

        start = time.time()
        validation_result = self._evaluate_rolling(
            app=app,
            train_close=train_close,
            val_close=val_close,
            val_ds=val_ds,
            order=order,
            horizon=int(app.horizon),
            scale=scale,
            split_name="validation",
            decision_threshold=0.5,
            refit_interval=refit_interval,
            stop_event=stop_event,
        )
        threshold_selection = self._calibrate_threshold(app, validation_result)
        eval_result = self._evaluate_rolling(
            app=app,
            train_close=pd.concat([train_close, val_close]),
            val_close=test_close,
            val_ds=test_ds,
            order=order,
            horizon=int(app.horizon),
            scale=scale,
            split_name="test",
            decision_threshold=threshold_selection["selected_threshold"],
            refit_interval=refit_interval,
            stop_event=stop_event,
        )
        train_seconds = time.time() - start

        self._attach_metadata(app, eval_result, stock=stock)
        energy_metadata = self._cpu_energy_metadata(
            app,
            train_seconds=train_seconds,
            train_samples=len(train_close),
        )
        capacity = dict(eval_result.metadata.get("capacity", {}) or {})
        parameter_count = capacity.get("arima_parameter_count")
        eval_result.metadata.update({
            "arima_order": order,
            "arima_refit_interval": refit_interval,
            "arima_update_mode": "state_space_extend",
            "threshold_source": "validation",
            "validation_selection": threshold_selection,
            "validation_loss_dense": threshold_selection["dense_loss"],
            "test_loss_dense": getattr(eval_result, "dense_val_loss", None),
            "threshold_macro_f1_validation": threshold_selection["macro_f1_threshold"],
            **energy_metadata,
            "train_seconds": train_seconds,
            "train_samples": len(train_close),
            "training_sample_unit": "raw_time_observation",
            "train_examples_unique": len(train_close),
            "sample_exposures": None,
            "epochs_completed": None,
            "gpu_peak_memory_mb": None,
            "total_params": parameter_count,
            "trainable_params": parameter_count,
            "capacity": capacity,
        })

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)
        app.frontendApp.set_status("Predicting with ARIMA baseline...")

        live_prob = self._live_probability(
            train_close=pd.concat([train_close, val_close]),
            val_close=test_close,
            order=order,
            horizon=int(app.horizon),
            scale=scale,
        )
        threshold = self._resolve_threshold(
            metrics,
            policy=getattr(app.args, "decision_threshold_policy", "macro_f1_dense"),
        )
        direction, confidence = self._direction_from_probability(live_prob, threshold)

        app.logger.info("[ARIMARunner] %s (%.1f%%)", direction, confidence)

        self._cleanup(train_ds, val_ds, test_ds)
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
        split_name: str,
        decision_threshold: float,
        refit_interval: int,
        stop_event,
    ) -> EvaluationResult:
        y_true_all = []
        y_pred_all = []
        probs_all = []
        prediction_dates = []
        hist_val = []
        total_windows = len(val_ds)

        initial_end = min(int(app.seq_len), len(val_close))
        initial_history = pd.concat([train_close, val_close.iloc[:initial_end]])
        fitted = self._fit_arima(initial_history, order=order)
        capacity = self._arima_capacity_metadata(fitted, order=order)
        fit_count = int(fitted is not None)
        update_count = 0
        fallback_count = 0
        progress_interval = max(total_windows // 10, 1)

        app.logger.info(
            "[ARIMARunner] %s walk-forward started | windows=%d | initial_history=%d",
            split_name,
            total_windows,
            len(initial_history),
        )
        app.frontendApp.set_status(
            f"Evaluating ARIMA {split_name} (0/{total_windows})..."
        )

        for i in range(total_windows):
            app._check_stop(stop_event)

            history_end = i + int(app.seq_len)
            current_idx = history_end - 1
            if current_idx >= len(val_close):
                break

            current_price = float(val_close.iloc[current_idx])
            if i > 0:
                should_refit = refit_interval > 0 and i % refit_interval == 0
                if should_refit:
                    history = pd.concat([train_close, val_close.iloc[:history_end]])
                    fitted = self._fit_arima(history, order=order)
                    if capacity.get("arima_parameter_count") is None:
                        capacity = self._arima_capacity_metadata(fitted, order=order)
                    fit_count += int(fitted is not None)
                elif fitted is not None:
                    fitted = self._extend_arima(fitted, current_price)
                    update_count += int(fitted is not None)

            if fitted is None:
                forecast = current_price
                fallback_count += 1
            else:
                forecast = self._forecast_fitted(
                    fitted,
                    horizon=horizon,
                    fallback=current_price,
                )
            prob = self._probability_from_forecast(
                forecast=forecast,
                current_price=current_price,
                scale=scale,
            )

            _x, y = val_ds[i]
            truth = int(float(y.item()) >= 0.5)
            pred = int(prob >= float(decision_threshold))
            timestamp = pd.Timestamp(val_ds.get_timestamp(i)).tz_localize(None)
            loss = self._binary_cross_entropy(prob, truth)

            y_true_all.append(truth)
            y_pred_all.append(pred)
            probs_all.append(prob)
            prediction_dates.append(timestamp)
            hist_val.append({"date": timestamp, "loss": loss})

            if hasattr(app.frontendApp, "updateProgress"):
                app.frontendApp.updateProgress((i + 1) / max(total_windows, 1))

            completed = i + 1
            if completed == total_windows or completed % progress_interval == 0:
                app.logger.info(
                    "[ARIMARunner] %s progress %d/%d",
                    split_name,
                    completed,
                    total_windows,
                )
                app.frontendApp.set_status(
                    f"Evaluating ARIMA {split_name} ({completed}/{total_windows})..."
                )

        mean_val_loss = float(np.mean([row["loss"] for row in hist_val])) if hist_val else float("nan")

        return EvaluationResult(
            y_true=y_true_all,
            y_pred=y_pred_all,
            probs=probs_all,
            prediction_dates=prediction_dates,
            decision_threshold=float(decision_threshold),
            dense_val_loss=mean_val_loss,
            hist_train=[],
            hist_val=hist_val,
            horizon=horizon,
            model_name=self.model_name,
            metadata={
                "evaluation_mode": "dense_rolling",
                "evaluation_split": str(split_name),
                "decision_threshold_policy": "fixed",
                "arima_update_mode": "state_space_extend",
                "arima_refit_interval": int(refit_interval),
                "arima_fit_count": int(fit_count),
                "arima_update_count": int(update_count),
                "arima_fallback_count": int(fallback_count),
                "capacity": capacity,
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
        values = self._clean_series(series)
        if values.empty:
            return float("nan")

        fitted = self._fit_arima(values, order=order)
        return self._forecast_fitted(
            fitted,
            horizon=horizon,
            fallback=float(values.iloc[-1]),
        )

    def _fit_arima(self, series: pd.Series, *, order: tuple[int, int, int]):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise RuntimeError(
                "ARIMA baseline requires statsmodels. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc

        values = self._clean_series(series)
        if len(values) <= max(sum(order), 3):
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return ARIMA(
                    values.to_numpy(dtype="float64"),
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            except Exception:
                return None

    @staticmethod
    def _arima_capacity_metadata(fitted, *, order: tuple[int, int, int]) -> dict:
        params = getattr(fitted, "params", None) if fitted is not None else None
        parameter_count = None
        parameter_storage_bytes = None

        if params is not None:
            try:
                params_array = np.asarray(params, dtype="float64").reshape(-1)
                parameter_count = int(params_array.size)
                parameter_storage_bytes = int(params_array.nbytes)
            except Exception:
                parameter_count = None
                parameter_storage_bytes = None

        fitted_model = getattr(fitted, "model", None) if fitted is not None else None
        state_dimension = getattr(fitted_model, "k_states", None)
        if state_dimension is not None:
            try:
                state_dimension = int(state_dimension)
            except Exception:
                state_dimension = None

        def finite_float(value):
            try:
                value = float(value)
                return value if np.isfinite(value) else None
            except Exception:
                return None

        p, d, q = (int(part) for part in order)
        return {
            "family": "arima",
            "primary_measure": "fitted_coefficients",
            "primary_value": parameter_count,
            "parameter_storage_bytes": parameter_storage_bytes,
            "arima_p": p,
            "arima_d": d,
            "arima_q": q,
            "arima_parameter_count": parameter_count,
            "arima_state_dimension": state_dimension,
            "arima_aic": finite_float(getattr(fitted, "aic", None)),
            "arima_bic": finite_float(getattr(fitted, "bic", None)),
        }

    @staticmethod
    def _extend_arima(fitted, observation: float):
        try:
            return fitted.extend(np.asarray([observation], dtype="float64"))
        except Exception:
            return None

    @staticmethod
    def _forecast_fitted(fitted, *, horizon: int, fallback: float) -> float:
        if fitted is None:
            return float(fallback)
        try:
            forecast = fitted.forecast(steps=max(int(horizon), 1))
            value = float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1])
            return value if np.isfinite(value) else float(fallback)
        except Exception:
            return float(fallback)

    @staticmethod
    def _clean_series(series: pd.Series) -> pd.Series:
        return pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()

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
    def _resolve_refit_interval(app) -> int:
        interval = int(getattr(app.args, "arima_refit_interval", 0))
        if interval < 0:
            raise ValueError("--arima_refit_interval must be >= 0")
        return interval

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
