from __future__ import annotations

from typing import Dict, List, Optional
import logging

import pandas as pd
import yfinance as yf

from data.price_loader import BasePriceLoader


class YahooPriceLoader(BasePriceLoader):
    """
    Yahoo Finance OHLCV loader with standardised cleaning.
    """

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_prices(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        period: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical OHLCV price data from Yahoo Finance.
        """
        if not tickers:
            raise ValueError("No tickers were provided to YahooPriceLoader.")

        cleaned_data: Dict[str, pd.DataFrame] = {}

        for ticker in [t.upper() for t in tickers]:
            try:
                df = self._download_single_ticker(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    period=period,
                )
                cleaned_data[ticker] = self._clean_dataframe(df)
            except Exception as exc:
                self.logger.warning("Failed to load %s: %s", ticker, exc)

        if not cleaned_data:
            raise RuntimeError("YahooPriceLoader failed for all requested tickers.")

        return cleaned_data

    def _download_single_ticker(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
        period: Optional[str],
    ) -> pd.DataFrame:
        """
        Download one ticker using either explicit dates or a Yahoo period.
        """
        yf_ticker = yf.Ticker(ticker)

        if start_date is not None or end_date is not None:
            raw = yf_ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
            )
        else:
            raw = yf_ticker.history(
                period=period or "1y",
                interval=interval,
                auto_adjust=False,
            )

        if raw.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        df = raw.reset_index().rename(
            columns={raw.index.name or "Date": "timestamp"}
        )

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise, validate, and clean an OHLCV dataframe.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")

        if df["close"].isna().all():
            raise ValueError("All close values are NaN")

        df = df.ffill().dropna()

        return df