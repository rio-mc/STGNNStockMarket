from __future__ import annotations

import logging
import re
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

from data.price_loader import BasePriceLoader


class YahooPriceLoader(BasePriceLoader):
    """
    Yahoo Finance OHLCV loader with standardised cleaning.

    Notes:
    - Returns only successfully loaded tickers.
    - Failed tickers are logged and omitted.
    - Caller should record requested vs loaded tickers for research provenance.
    """

    PROVIDER_NAME = "yahoo"
    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

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

        self._configure_yfinance_cache()

    def _configure_yfinance_cache(self) -> None:
        cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))

    def load_prices(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        period: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        if not tickers:
            raise ValueError("No tickers were provided to YahooPriceLoader.")

        cleaned_data: Dict[str, pd.DataFrame] = {}
        requested = [str(t).upper().strip() for t in tickers]
        effective_interval = self._normalise_interval_for_period(interval, period)

        for ticker in requested:
            try:
                df = self._download_single_ticker(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=effective_interval,
                    period=period,
                )
                cleaned_data[ticker] = self._clean_dataframe(df)
            except Exception as exc:
                self.logger.warning("Failed to load %s: %s", ticker, exc)

        if not cleaned_data:
            raise RuntimeError("YahooPriceLoader failed for all requested tickers.")

        self.logger.info(
            "Loaded prices for %d/%d tickers using provider=%s interval=%s.",
            len(cleaned_data),
            len(requested),
            self.PROVIDER_NAME,
            effective_interval,
        )
        return cleaned_data

    def _normalise_interval_for_period(self, interval: str, period: Optional[str]) -> str:
        interval_key = str(interval or "1d").strip().lower()
        if interval_key not in self.INTRADAY_INTERVALS:
            return interval

        period_days = self._period_to_days(period)
        if period_days is None or period_days <= 60:
            return interval

        self.logger.warning(
            "Yahoo Finance does not provide %s data for period=%s; using interval=1d.",
            interval,
            period,
        )
        return "1d"

    @staticmethod
    def _period_to_days(period: Optional[str]) -> Optional[int]:
        if not period:
            return None

        match = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]+)\s*", str(period))
        if not match:
            return None

        value = int(match.group(1))
        unit = match.group(2).lower()
        if unit in {"d", "day", "days"}:
            return value
        if unit in {"wk", "w", "week", "weeks"}:
            return value * 7
        if unit in {"mo", "mon", "month", "months"}:
            return value * 30
        if unit in {"y", "yr", "year", "years"}:
            return value * 365
        return None

    def _download_single_ticker(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
        period: Optional[str],
    ) -> pd.DataFrame:
        raw = pd.DataFrame()

        try:
            raw = self._download_with_chart_api(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                period=period,
            )
        except Exception as exc:
            self.logger.debug("Yahoo chart API failed for %s: %s", ticker, exc)

        try:
            if raw.empty:
                raw = self._download_with_ticker_history(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    period=period,
                )
        except Exception as exc:
            self.logger.debug("Ticker.history failed for %s: %s", ticker, exc)

        if raw.empty:
            try:
                raw = self._download_with_yf_download(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    period=period,
                )
            except Exception as exc:
                self.logger.debug("yf.download failed for %s: %s", ticker, exc)

        if raw.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        df = raw.reset_index().rename(columns={raw.index.name or "Date": "timestamp"})
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

    @staticmethod
    def _download_with_ticker_history(
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
        period: Optional[str],
    ) -> pd.DataFrame:
        yf_ticker = yf.Ticker(ticker)

        if start_date is not None or end_date is not None:
            return yf_ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
            )

        return yf_ticker.history(
            period=period or "1y",
            interval=interval,
            auto_adjust=False,
        )

    @staticmethod
    def _download_with_yf_download(
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
        period: Optional[str],
    ) -> pd.DataFrame:
        kwargs = {
            "tickers": ticker,
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }

        if start_date is not None or end_date is not None:
            kwargs["start"] = start_date
            kwargs["end"] = end_date
        else:
            kwargs["period"] = period or "1y"

        raw = yf.download(**kwargs)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    @staticmethod
    def _download_with_chart_api(
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
        period: Optional[str],
    ) -> pd.DataFrame:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "interval": interval,
            "includePrePost": "false",
            "events": "history",
        }

        if start_date is not None or end_date is not None:
            start_ts = pd.Timestamp(start_date or "1970-01-01", tz=timezone.utc)
            end_ts = pd.Timestamp(end_date or pd.Timestamp.utcnow(), tz=timezone.utc)
            params["period1"] = int(start_ts.timestamp())
            params["period2"] = int(end_ts.timestamp())
        else:
            params["range"] = period or "1y"

        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        result = payload.get("chart", {}).get("result")
        if not result:
            return pd.DataFrame()

        result = result[0]
        timestamps = result.get("timestamp") or []
        quotes = result.get("indicators", {}).get("quote") or []
        if not timestamps or not quotes:
            return pd.DataFrame()

        quote = quotes[0]
        df = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True),
        )
        df.index.name = "Date"
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        if df["close"].isna().all():
            raise ValueError("All close values are NaN")

        df = df.ffill().dropna()

        if df.empty:
            raise ValueError("Dataframe is empty after cleaning")

        return df
