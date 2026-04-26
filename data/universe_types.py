from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class UniverseMetadata:
    universe_id: str
    universe_provider: str
    source_note: str
    selection_method: str
    snapshot_date: str
    historical_constituent_reconstruction: bool = False
    universe_role: Optional[str] = None
    requested_top_n: Optional[int] = None
    actual_count: int = 0
    dropped_tickers: List[str] = field(default_factory=list)
    universe_file: Optional[str] = None


@dataclass(frozen=True)
class UniverseDefinition:
    universe_id: str
    tickers: List[str]
    ticker_to_sector: Dict[str, str]
    ticker_to_industry: Dict[str, str]
    metadata: UniverseMetadata

    def with_dropped_tickers(self, dropped_tickers: List[str]) -> "UniverseDefinition":
        updated_metadata = UniverseMetadata(
            universe_id=self.metadata.universe_id,
            universe_provider=self.metadata.universe_provider,
            source_note=self.metadata.source_note,
            selection_method=self.metadata.selection_method,
            snapshot_date=self.metadata.snapshot_date,
            historical_constituent_reconstruction=self.metadata.historical_constituent_reconstruction,
            universe_role=self.metadata.universe_role,
            requested_top_n=self.metadata.requested_top_n,
            actual_count=self.metadata.actual_count,
            dropped_tickers=list(dropped_tickers or []),
            universe_file=self.metadata.universe_file,
        )
        return UniverseDefinition(
            universe_id=self.universe_id,
            tickers=list(self.tickers),
            ticker_to_sector=dict(self.ticker_to_sector),
            ticker_to_industry=dict(self.ticker_to_industry),
            metadata=updated_metadata,
        )

    def to_dict(self) -> Dict:
        return {
            "tickers": list(self.tickers),
            "universe_id": self.universe_id,
            "universe_name": self.universe_id,  # compatibility
            "selection_date": self.metadata.snapshot_date,
            "selection_method": self.metadata.selection_method,
            "requested_top_n": self.metadata.requested_top_n,
            "actual_count": self.metadata.actual_count,
            "ticker_to_sector": dict(self.ticker_to_sector),
            "ticker_to_industry": dict(self.ticker_to_industry),
            "metadata": {
                "universe_id": self.metadata.universe_id,
                "universe_provider": self.metadata.universe_provider,
                "source_note": self.metadata.source_note,
                "historical_constituent_reconstruction": self.metadata.historical_constituent_reconstruction,
                "universe_role": self.metadata.universe_role,
                "dropped_tickers": list(self.metadata.dropped_tickers),
                "universe_file": self.metadata.universe_file,
            },
        }