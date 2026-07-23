import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models.runners.arima_runner import ARIMARunner


class _Dataset:
    def __init__(self, length):
        self._labels = torch.tensor([i % 2 for i in range(length)], dtype=torch.float32)
        self._timestamps = pd.date_range("2025-01-01", periods=length, freq="h")

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return torch.empty(0), self._labels[index]

    def get_timestamp(self, index):
        return self._timestamps[index]


class _Frontend:
    def __init__(self):
        self.statuses = []
        self.progress = []

    def set_status(self, status):
        self.statuses.append(status)

    def updateProgress(self, progress):
        self.progress.append(progress)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


class _Fitted:
    def __init__(self, last_value, tracker):
        self.last_value = float(last_value)
        self.tracker = tracker

    def forecast(self, steps):
        self.tracker["forecasts"] += 1
        return np.asarray([self.last_value + 0.25] * steps)

    def extend(self, observations):
        self.tracker["updates"] += 1
        return _Fitted(float(observations[-1]), self.tracker)


class ARIMARunnerTests(unittest.TestCase):
    def test_walk_forward_fits_once_and_extends_state(self):
        runner = ARIMARunner()
        tracker = {"fits": 0, "updates": 0, "forecasts": 0}

        def fake_fit(series, *, order):
            tracker["fits"] += 1
            return _Fitted(series.iloc[-1], tracker)

        runner._fit_arima = fake_fit
        frontend = _Frontend()
        app = SimpleNamespace(
            seq_len=3,
            args=SimpleNamespace(target_stock="AAPL"),
            logger=_Logger(),
            frontendApp=frontend,
            _check_stop=lambda _event: None,
        )
        train = pd.Series([10.0, 11.0, 12.0])
        validation = pd.Series([13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        dataset = _Dataset(length=4)

        result = runner._evaluate_rolling(
            app=app,
            train_close=train,
            val_close=validation,
            val_ds=dataset,
            order=(1, 1, 1),
            horizon=1,
            scale=1.0,
            split_name="validation",
            decision_threshold=0.5,
            refit_interval=0,
            stop_event=None,
        )

        self.assertEqual(tracker, {"fits": 1, "updates": 3, "forecasts": 4})
        self.assertEqual(len(result.probs), 4)
        self.assertEqual(result.metadata["arima_fit_count"], 1)
        self.assertEqual(result.metadata["arima_update_count"], 3)
        self.assertEqual(result.metadata["arima_fallback_count"], 0)
        self.assertEqual(frontend.progress[-1], 1.0)

    def test_refit_interval_reestimates_periodically(self):
        runner = ARIMARunner()
        tracker = {"fits": 0, "updates": 0, "forecasts": 0}

        def fake_fit(series, *, order):
            tracker["fits"] += 1
            return _Fitted(series.iloc[-1], tracker)

        runner._fit_arima = fake_fit
        app = SimpleNamespace(
            seq_len=2,
            args=SimpleNamespace(target_stock="AAPL"),
            logger=_Logger(),
            frontendApp=_Frontend(),
            _check_stop=lambda _event: None,
        )

        result = runner._evaluate_rolling(
            app=app,
            train_close=pd.Series([10.0, 11.0, 12.0]),
            val_close=pd.Series([13.0, 14.0, 15.0, 16.0, 17.0]),
            val_ds=_Dataset(length=4),
            order=(1, 1, 1),
            horizon=1,
            scale=1.0,
            split_name="test",
            decision_threshold=0.5,
            refit_interval=2,
            stop_event=None,
        )

        self.assertEqual(tracker["fits"], 2)
        self.assertEqual(tracker["updates"], 2)
        self.assertEqual(result.metadata["arima_fit_count"], 2)
        self.assertEqual(result.metadata["arima_update_count"], 2)


if __name__ == "__main__":
    unittest.main()
