from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtesting.types import BacktestResult, TradeRecord
from evaluation.evaluation_types import EvaluationResult


class BacktestEngine:
    """
    Non-overlapping directional backtest engine.

    Trading rule:
    - prediction == 1 => long
    - prediction == 0 => short

    For research-defensible evaluation, the backtest can derive its binary
    signal from probabilities and an explicit threshold. The executed trade
    indices are returned so trade-aligned predictive metrics can be computed
    on the exact non-overlapping subset that was actually tradable.
    """

    def __init__(self, tc_per_side: float = 0.0, rf_annual: float = 0.0):
        self.tc_per_side = float(tc_per_side)
        self.rf_annual = float(rf_annual)

    def run(
        self,
        model_name: str,
        evaluation_result: EvaluationResult,
        price_df: pd.DataFrame,
        price_column: str = "close",
        threshold: Optional[float] = None,
    ) -> Optional[BacktestResult]:
        if evaluation_result.horizon is None:
            return None

        if not evaluation_result.prediction_dates or not evaluation_result.probs:
            return None

        if price_column not in price_df.columns:
            fallback_col = next(iter(price_df.columns), None)
            if fallback_col is None:
                return None
            price_column = fallback_col

        price_series = price_df[price_column].copy()
        if not isinstance(price_series.index, pd.DatetimeIndex):
            price_series.index = pd.to_datetime(price_series.index)

        if getattr(price_series.index, "tz", None) is not None:
            price_series.index = price_series.index.tz_localize(None)

        horizon = int(evaluation_result.horizon)
        pred_dates = [self._normalise_ts(d) for d in evaluation_result.prediction_dates]
        signal_threshold = float(
            evaluation_result.decision_threshold if threshold is None else threshold
        )
        probs = np.asarray(evaluation_result.probs, dtype=float)
        y_pred = (probs >= signal_threshold).astype(int)

        result = BacktestResult(
            model_name=str(model_name).upper(),
            horizon=horizon,
            tc_per_side=self.tc_per_side,
            rf_annual=self.rf_annual,
            signal_threshold=signal_threshold,
        )

        i = 0
        while i < len(pred_dates):
            date = pred_dates[i]

            if date not in price_series.index:
                i += 1
                continue

            try:
                entry_idx = price_series.index.get_loc(date)
                exit_idx = entry_idx + horizon
                if exit_idx >= len(price_series):
                    break

                entry_price = float(price_series.iloc[entry_idx])
                exit_price = float(price_series.iloc[exit_idx])

                if entry_price == 0:
                    i += 1
                    continue

                raw_ret = (exit_price - entry_price) / entry_price
                direction = 1 if int(y_pred[i]) == 1 else -1
                gross_ret = direction * raw_ret
                net_ret = gross_ret - (2.0 * self.tc_per_side)

                entry_time = price_series.index[entry_idx]
                exit_time = price_series.index[exit_idx]

                result.executed_indices.append(i)
                result.equity.append(result.equity[-1] * (1.0 + net_ret))
                result.trade_returns.append(float(net_ret))
                result.trade_times.append(entry_time)
                result.trades.append(
                    TradeRecord(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        raw_return=float(raw_ret),
                        net_return=float(net_ret),
                    )
                )

                i += horizon

            except Exception:
                i += 1
                continue

        result.sharpe = self._compute_annualised_sharpe(
            trade_returns=result.trade_returns,
            trade_times=result.trade_times,
            rf_annual=self.rf_annual,
        )

        return result

    def _compute_annualised_sharpe(
        self,
        trade_returns: list[float],
        trade_times: list[pd.Timestamp],
        rf_annual: float,
    ) -> float:
        if len(trade_returns) <= 1:
            return 0.0

        if len(trade_times) > 1:
            dt = np.median(
                np.diff(pd.to_datetime(trade_times)).astype("timedelta64[s]").astype(float)
            )
            secs_per_period = max(float(dt), 1.0)
        else:
            secs_per_period = 3600.0

        periods_per_year = (365.25 * 24 * 3600) / secs_per_period

        r = np.asarray(trade_returns, dtype=float)
        rf_per_period = (
            (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
            if rf_annual != 0.0
            else 0.0
        )
        excess = r - rf_per_period
        sd = np.std(excess, ddof=1)

        return float((excess.mean() / (sd + 1e-12)) * np.sqrt(periods_per_year))

    @staticmethod
    def _normalise_ts(value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
