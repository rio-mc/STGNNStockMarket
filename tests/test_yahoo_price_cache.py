import sys
import types

import pandas as pd

try:
    import yfinance  # noqa: F401
except ModuleNotFoundError:
    sys.modules["yfinance"] = types.ModuleType("yfinance")

from data.yahoo_price_loader import YahooPriceLoader


def test_frozen_date_window_uses_persistent_price_cache(workspace_tmp, monkeypatch):
    loader = YahooPriceLoader(price_cache_dir=workspace_tmp)
    calls = []
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.0, 2.0, 3.0],
            "volume": [10, 20, 30],
        }
    )

    def download(**_kwargs):
        calls.append(True)
        return raw.copy()

    monkeypatch.setattr(loader, "_download_single_ticker", download)

    first = loader.load_prices(
        ["AAPL"],
        start_date="2024-07-23",
        end_date="2026-07-22",
        interval="1h",
    )
    second = loader.load_prices(
        ["AAPL"],
        start_date="2024-07-23",
        end_date="2026-07-22",
        interval="1h",
    )

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first["AAPL"], second["AAPL"])
