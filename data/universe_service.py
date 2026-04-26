from __future__ import annotations

from typing import Dict, List, Optional

from data.static_universe import BaseUniverseProvider, StaticCsvUniverseProvider
from data.universe_types import UniverseDefinition


class UniverseService:
    """
    Dispatch layer for explicit, versioned asset universe selection.

    Separates:
    - universe_id: logical basket (sp500, nasdaq100, custom)
    - universe_provider: how that basket is resolved (static_csv, future APIs, etc.)
    """

    def __init__(self) -> None:
        self.providers: Dict[str, BaseUniverseProvider] = {
            "static_csv": StaticCsvUniverseProvider(),
        }

    def resolve_definition(
        self,
        universe_id: str,
        universe_provider: str = "static_csv",
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> UniverseDefinition:
        provider_key = str(universe_provider).strip().lower()
        if provider_key not in self.providers:
            supported = ", ".join(sorted(self.providers.keys()))
            raise ValueError(
                f"Unsupported universe_provider='{universe_provider}'. Supported: {supported}"
            )

        provider = self.providers[provider_key]
        return provider.get_universe(
            universe_id=universe_id,
            top_n=top_n,
            as_of_date=as_of_date,
            custom_tickers=custom_tickers,
        )

    def resolve(
        self,
        universe_id: str,
        universe_provider: str = "static_csv",
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> Dict:
        definition = self.resolve_definition(
            universe_id=universe_id,
            universe_provider=universe_provider,
            top_n=top_n,
            as_of_date=as_of_date,
            custom_tickers=custom_tickers,
        )
        return definition.to_dict()