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


class PriceLoaderRegistry:
    """
    Small provider registry for price loaders.

    Keeps the rest of the app independent from the concrete implementation
    used to fetch prices.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, BasePriceLoader] = {}

    def register(self, provider_name: str, loader: BasePriceLoader) -> None:
        key = str(provider_name).strip().lower()
        self._providers[key] = loader

    def get(self, provider_name: str) -> BasePriceLoader:
        key = str(provider_name).strip().lower()
        if key not in self._providers:
            supported = ", ".join(sorted(self._providers.keys()))
            raise ValueError(
                f"Unsupported price_provider='{provider_name}'. Supported: {supported}"
            )
        return self._providers[key]

    def list_supported(self) -> List[str]:
        return sorted(self._providers.keys())