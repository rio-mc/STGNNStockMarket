from __future__ import annotations

from typing import Dict, List, Optional

from data.yahoo_universe import YahooUniverseProvider


class UniverseService:
    """
    Dispatch layer for asset universe selection.
    """

    def __init__(self) -> None:
        self.yahoo_provider = YahooUniverseProvider()

    def resolve(
        self,
        universe_name: str,
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> Dict:
        return self.yahoo_provider.get_universe(
            universe_name=universe_name,
            top_n=top_n,
            as_of_date=as_of_date,
            custom_tickers=custom_tickers,
        )