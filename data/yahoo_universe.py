from __future__ import annotations

from typing import Dict, List, Optional

from data.universe_service import UniverseService


class LegacyUniverseProviderAdapter:
    """
    Backwards-compatible wrapper for older code paths.

    Historical naming used 'YahooUniverseProvider', but benchmark universes
    are no longer resolved from Yahoo or from embedded lists. They are now
    resolved through UniverseService using explicit universe and provider ids.
    """

    def __init__(self) -> None:
        self._service = UniverseService()

    def get_universe(
        self,
        universe_name: str,
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> Dict:
        return self._service.resolve(
            universe_id=universe_name,
            universe_provider="static_csv",
            top_n=top_n,
            as_of_date=as_of_date,
            custom_tickers=custom_tickers,
        )


# Optional legacy alias to avoid breaking old imports immediately.
YahooUniverseProvider = LegacyUniverseProviderAdapter