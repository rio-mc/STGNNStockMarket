from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BasePriceLoader(ABC):
    @abstractmethod
    def load_prices(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        period: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for the requested tickers.

        Returns:
            Dict[ticker, DataFrame]
        """
        raise NotImplementedError