"""
Dataset Registry Module

Provides a standardized interface for loading datasets with a consistent output format.
Decouples dataset logic from the pipeline to prevent data leakage and ensure reproducibility.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DatasetResult:
    """
    Standardized output format for all datasets.
    
    Attributes:
        data: Dict mapping ticker symbols to their DataFrames
        tickers: List of valid ticker symbols
        index: DatetimeIndex of shared timestamps
        metadata: Additional information about the dataset
    """
    data: Dict[str, pd.DataFrame]
    tickers: List[str]
    index: pd.DatetimeIndex
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": {t: df.to_dict() for t, df in self.data.items()},
            "tickers": self.tickers,
            "index": self.index.isoformat() if hasattr(self.index, 'isoformat') else str(self.index),
            "metadata": self.metadata,
        }


class BaseDataset(ABC):
    """Abstract base class for datasets."""
    
    @abstractmethod
    def load(self) -> DatasetResult:
        """Load and return the dataset."""
        raise NotImplementedError


class YahooFinanceDataset(BaseDataset):
    """Dataset that loads data from Yahoo Finance."""
    
    def __init__(
        self,
        tickers: List[str],
        period: str = "729d",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.tickers = [t.upper() for t in tickers]
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> DatasetResult:
        """Load price data from Yahoo Finance."""
        import yfinance as yf
        
        data: Dict[str, pd.DataFrame] = {}
        valid_tickers: List[str] = []
        
        for ticker in self.tickers:
            try:
                if self.start_date and self.end_date:
                    df = yf.Ticker(ticker).history(
                        start=self.start_date,
                        end=self.end_date,
                        interval=self.interval,
                    )
                else:
                    df = yf.Ticker(ticker).history(
                        period=self.period,
                        interval=self.interval,
                    )
                
                if df.empty:
                    self.logger.warning(f"No data returned for {ticker}")
                    continue
                
                # Reset index and standardize column names
                df = df.reset_index().rename(columns={df.index.name or "Date": "timestamp"})
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
                
                # Ensure timestamp is datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                
                data[ticker] = df
                valid_tickers.append(ticker)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {ticker}: {e}")
        
        if not data:
            raise RuntimeError(f"No data could be loaded for any ticker: {self.tickers}")
        
        # Build shared index
        shared_index = self._build_shared_index(data, valid_tickers)
        
        metadata = {
            "source": "yfinance",
            "period": self.period,
            "interval": self.interval,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "requested_tickers": self.tickers,
            "valid_tickers": valid_tickers,
            "dropped_tickers": [t for t in self.tickers if t not in valid_tickers],
            "load_timestamp": datetime.now().isoformat(),
        }
        
        return DatasetResult(
            data=data,
            tickers=valid_tickers,
            index=shared_index,
            metadata=metadata,
        )
    
    def _build_shared_index(self, data: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DatetimeIndex:
        """Build a shared DatetimeIndex across all tickers."""
        if not tickers:
            return pd.DatetimeIndex([])
        
        shared = set.intersection(*(set(df.index) for df in data.values() if not df.empty))
        return pd.DatetimeIndex(sorted(shared))


class CSVDirectoryDataset(BaseDataset):
    """Dataset that loads data from CSV files in a directory."""
    
    def __init__(
        self,
        data_dir: str,
        tickers: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tickers = [t.upper() for t in tickers] if tickers else None
        self.required_columns = required_columns or ["timestamp", "open", "high", "low", "close", "volume"]
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> DatasetResult:
        """Load price data from CSV files."""
        data: Dict[str, pd.DataFrame] = {}
        valid_tickers: List[str] = []
        
        if self.tickers:
            # Load specific tickers
            for ticker in self.tickers:
                csv_path = self.data_dir / f"{ticker}.csv"
                if not csv_path.exists():
                    self.logger.warning(f"CSV file not found: {csv_path}")
                    continue
                
                try:
                    df = pd.read_csv(csv_path)
                    df = self._clean_dataframe(df)
                    if df is not None and not df.empty:
                        data[ticker] = df
                        valid_tickers.append(ticker)
                except Exception as e:
                    self.logger.warning(f"Failed to load {ticker}: {e}")
        else:
            # Load all CSV files in directory
            for csv_path in self.data_dir.glob("*.csv"):
                ticker = csv_path.stem.upper()
                try:
                    df = pd.read_csv(csv_path)
                    df = self._clean_dataframe(df)
                    if df is not None and not df.empty:
                        data[ticker] = df
                        valid_tickers.append(ticker)
                except Exception as e:
                    self.logger.warning(f"Failed to load {ticker}: {e}")
        
        if not data:
            raise RuntimeError(f"No data could be loaded from {self.data_dir}")
        
        # Build shared index
        shared_index = self._build_shared_index(data, valid_tickers)
        
        metadata = {
            "source": "csv_directory",
            "data_dir": str(self.data_dir),
            "requested_tickers": self.tickers,
            "valid_tickers": valid_tickers,
            "dropped_tickers": [t for t in (self.tickers or []) if t not in valid_tickers],
            "load_timestamp": datetime.now().isoformat(),
        }
        
        return DatasetResult(
            data=data,
            tickers=valid_tickers,
            index=shared_index,
            metadata=metadata,
        )
    
    def _clean_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and validate a DataFrame."""
        # Check required columns
        df_cols_lower = {c.lower() for c in df.columns}
        required_lower = {c.lower() for c in self.required_columns}
        
        if not required_lower.issubset(df_cols_lower):
            return None
        
        # Rename columns to lowercase
        rename_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename_map)
        
        # Ensure timestamp column exists and parse it
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        
        # Select and order columns
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available]
        
        return df
    
    def _build_shared_index(self, data: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DatetimeIndex:
        """Build a shared DatetimeIndex across all tickers."""
        if not tickers:
            return pd.DatetimeIndex([])
        
        shared = set.intersection(*(set(df.index) for df in data.values() if not df.empty))
        return pd.DatetimeIndex(sorted(shared))


class DatasetRegistry:
    """
    Registry for loading datasets with a standardized interface.
    
    Usage:
        dataset = DatasetRegistry.load("yahoo", tickers=["AAPL", "GOOGL"])
        result = dataset.load()
        
        # Standardized output:
        # result.data - Dict[str, DataFrame]
        # result.tickers - List[str]
        # result.index - DatetimeIndex
        # result.metadata - Dict
    """
    
    _providers: Dict[str, type] = {
        "yahoo": YahooFinanceDataset,
        "yfinance": YahooFinanceDataset,
        "csv": CSVDirectoryDataset,
        "csv_dir": CSVDirectoryDataset,
    }
    
    @classmethod
    def register(cls, name: str, dataset_class: type) -> None:
        """Register a new dataset provider."""
        cls._providers[name.lower()] = dataset_class
    
    @classmethod
    def available_providers(cls) -> List[str]:
        """List all available dataset providers."""
        return sorted(cls._providers.keys())
    
    @classmethod
    def load(cls, provider: str, *args, **kwargs) -> BaseDataset:
        """
        Load a dataset by provider name.
        
        Args:
            provider: Name of the provider ("yahoo", "csv", etc.)
            *args: Positional arguments passed to the dataset constructor
            **kwargs: Keyword arguments passed to the dataset constructor
            
        Returns:
            BaseDataset: The dataset instance (call .load() to get data)
            
        Raises:
            ValueError: If the provider is not supported
        """
        key = str(provider).strip().lower()
        
        if key not in cls._providers:
            supported = ", ".join(cls.available_providers())
            raise ValueError(
                f"Unknown dataset provider '{provider}'. "
                f"Available: {supported}"
            )
        
        return cls._providers[key](*args, **kwargs)
    
    @classmethod
    def load_and_get_data(cls, provider: str, *args, **kwargs) -> DatasetResult:
        """
        Convenience method to load and return data directly.
        
        Args:
            provider: Name of the provider
            *args: Positional arguments passed to the dataset constructor
            **kwargs: Keyword arguments passed to the dataset constructor
            
        Returns:
            DatasetResult: The loaded dataset
        """
        dataset = cls.load(provider, *args, **kwargs)
        return dataset.load()