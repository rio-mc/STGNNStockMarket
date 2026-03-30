import os
import glob
import pandas as pd
import requests
import logging
from typing import List, Optional, Dict
import yfinance as yf
import alpha_vantage as av

class RawDataHandler:
    """
    Ingests time-series data from CSV, yfinance or Alpha Vantage.
    Cleans, applies horizon slicing, and exposes DataFrames per ticker.
    """

    def __init__(
        self,
        dataDir: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        source: str = "csv",
        requiredColumns: Optional[List[str]] = None,
        avApiKey: Optional[str] = None,
        yfInterval: Optional[str] = None,
        yfPeriod: str = "1mo",
    ):
        # === STEP 1: Validate And Store Source Settings ===
        # ------------------------------------
        self.source = source.lower()
        if self.source not in {"csv", "yfinance", "av"}:
            raise ValueError(f"Invalid source '{source}'. Must be one of 'csv', 'yfinance', 'av'.")

		# === STEP 2: Assign Parameters and Environment Keys ===
        # ------------------------------------
        self.requiredColumns = requiredColumns or ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.dataDir = dataDir
        self.tickers = [t.upper() for t in tickers] if tickers else []
        self.avApiKey = avApiKey or os.environ.get("ALPHAVANTAGE_API_KEY")
        self.yfPeriod = yfPeriod
        self.yfInterval = yfInterval
        self._dataframes: Dict[str, pd.DataFrame] = {}

        # === STEP 3: Initialise Logger ===
        # ------------------------------------
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)

        # === STEP 4: Load Ticker Data ===
        # ------------------------------------
        self.loadAllTickers()

    def loadFromYfinance(self) -> None:
        # === STEP 1: Set Data Interval ===
        # ------------------------------------
        interval = self.yfInterval

        # === STEP 2: Verify Stocks Exist In YahooFinance ===
        # ------------------------------------
        if not self.tickers:
            raise ValueError("No tickers provided for Alpha Vantage mode.")
        
        # === STEP 3: Collect Data From YahooFinance ===
        # ------------------------------------
        for tkr in self.tickers:
            try:
                #   1. Collect history per stock (ticker)
                raw = yf.Ticker(tkr).history(period=self.yfPeriod,
                                            interval=interval)
                if raw.empty:
                    raise RuntimeError(f"No data returned for {tkr}")

                #   2. Reset index: column named "timestamp"
                df = raw.reset_index().rename(
                    columns={raw.index.name or "Date": "timestamp"}
                )
                #   3. Lower-case names for consistency
                df = df.rename(columns={
                    "Open":   "open",
                    "High":   "high",
                    "Low":    "low",
                    "Close":  "close",
                    "Volume": "volume"
                })

                #   4. Clean and store data
                df_clean = self.cleanDataframe(df)
                self._dataframes[tkr] = df_clean

            except Exception as e:
                self.logger.warning(f"[Warning] {tkr} failed: {e!s}")

        # === STEP 4: Exception Handling ===
        # ------------------------------------
        if not self._dataframes:
            raise RuntimeError(f"No data could be loaded using source='{self.source}' — all tickers failed.")
        else:
            failed = set(self.tickers) - set(self._dataframes.keys())
            if failed:
                print(f"[WARNING] These tickers failed and will be skipped: {sorted(failed)}")

    def cleanDataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # === STEP 1: Enforce Required Columns ===
        # ------------------------------------
        missing_cols = [col for col in self.requiredColumns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df[self.requiredColumns].copy()

        # === STEP 2: Parse Valid Timestamps ===
        # ------------------------------------
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')

        # === STEP 3: Set Timestamp Indices ===
        # ------------------------------------
        df = df.set_index('timestamp')

        # === STEP 4: Check For NaNs ===
        # ------------------------------------
        
        #   1. Find NaNs
        if df['close'].isna().all():
            raise ValueError("All 'close' values are NaN — invalid ticker data.")
        
        #   2. Carry previous value to fill any Na
        df = df.ffill().dropna()

        # === STEP 5: Return Clean DataFrame ===
        # ------------------------------------
        return df
    
    def loadFromCsv(self) -> None:
        # === STEP 1: Verify Directory ===
        # ------------------------------------
        if not self.dataDir:
            raise ValueError("dataDir must be provided for CSV mode.")
		
        # === STEP 2: Collect CSV Files ===
        # ------------------------------------

        #   1. Collect all ticker CSVs
        for path in glob.glob(os.path.join(self.dataDir, "*.csv")):
            
            #   2. Parse as ticker names (not ticker.csv)
            ticker = os.path.basename(path).replace(".csv", "").upper()
            try:
                #   3. Read and clean data
                df = pd.read_csv(path)
                dfClean = self.cleanDataframe(df)

                #   4. Create DataFrame
                self._dataframes[ticker] = dfClean

                #   5. Correct formatting
                df.columns = [c.lower() for c in df.columns]
            except Exception as e:
                self.logger.warning(f"[Warning] {ticker} failed: {e}")

    def loadFromAlphaVantage(self) -> None:
        # === STEP 1: Verify AlphaVantage Key ===
        # ------------------------------------
        if not self.avApiKey:
            raise ValueError("Alpha Vantage API key not provided.")
            
        # === STEP 2: Verify Stocks Exist In AlphaVantage ===
        # ------------------------------------
        if not self.tickers:
            raise ValueError("No tickers provided for Alpha Vantage mode.")

        # === STEP 3: Collect Data From AlphaVantage ===
        # ------------------------------------
        baseUrl = "https://www.alphavantage.co/query"
        for tkr in self.tickers:
            #   1. AlphaVantage data configuration
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol":   tkr,
                "outputsize": "compact",
                "datatype": "json",
                "apikey":    self.avApiKey
            }
            try:
                #   2. Format records
                resp    = requests.get(baseUrl, params=params, timeout=10)
                data    = resp.json().get("Time Series (Daily)", {})
                records = [
                    {
                        "timestamp": pd.to_datetime(date),
                        "open":      float(vals["1. open"]),
                        "high":      float(vals["2. high"]),
                        "low":       float(vals["3. low"]),
                        "close":     float(vals["4. close"]),
                        "volume":    int(vals["6. volume"])
                    }
                    for date, vals in data.items()
                    if all(k in vals for k in ["1. open", "2. high", "3. low", "4. close", "6. volume"])
                ]
                #   3. Clean and store DataFrames
                if records:
                    dfClean = self.cleanDataframe(pd.DataFrame.from_records(records))
                    self._dataframes[tkr] = dfClean

            # === STEP 4: Exception Handling ===
            # ------------------------------------
            except Exception as e:
                self.logger.warning(f"[Warning] {tkr} failed: {e}")

    def loadAllTickers(self) -> None:
        # ====================================
		# === Helper to load raw data

        #   1. Use chosen source
        if self.source == "csv":
            self.loadFromCsv()
        elif self.source == "yfinance":
            self.loadFromYfinance()
        else:
            self.loadFromAlphaVantage()

        #   2. Ensure data exists
        if not self._dataframes:
            raise RuntimeError(f"No data could be loaded using source='{self.source}'.")
        
    def listTickers(self) -> List[str]:
        # ====================================
		# === Helper to list tickers
        return list(self._dataframes.keys())

    def getDataframe(self, ticker: str) -> pd.DataFrame:
        # ====================================
		# === Helper to get DataFrame

        #   1. Search all ticker DataFrames
        t = ticker.upper()
        if t not in self._dataframes:
            raise KeyError(f"Ticker '{t}' not loaded.")
        
        #   2. Copy ticker (if valid) DataFrame
        df = self._dataframes[t].copy()

        return df

