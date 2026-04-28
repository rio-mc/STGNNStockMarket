from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data.universe_types import UniverseDefinition, UniverseMetadata


class BaseUniverseProvider(ABC):
    @abstractmethod
    def get_universe(
        self,
        universe_id: str,
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> UniverseDefinition:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticUniverseConfig:
    universe_id: str
    csv_filename: str
    source_note: str
    universe_role: Optional[str] = None


class StaticCsvUniverseProvider(BaseUniverseProvider):
    """
    Loads versioned universe definitions from CSV files.

    Expected CSV columns:
    - ticker
    - sector

    Optional columns:
    - company_name
    - industry
    - rank
    """

    REQUIRED_COLUMNS = {"ticker", "sector"}
    PROVIDER_NAME = "static_csv"

    def __init__(self, project_root: Optional[Path] = None) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        self.project_root = Path(project_root)
        self.universe_dir = self.project_root / "static" / "universes"

        self.registry: Dict[str, StaticUniverseConfig] = {
            "sp500": StaticUniverseConfig(
                universe_id="sp500",
                csv_filename="sp500_tickers.csv",
                source_note="static_snapshot_sp500",
                universe_role="primary",
            ),
            "nasdaq100": StaticUniverseConfig(
                universe_id="nasdaq100",
                csv_filename="nasdaq100_ticker.csv",
                source_note="static_snapshot_nasdaq100",
                universe_role="secondary",
            ),
        }

    def get_universe(
        self,
        universe_id: str,
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> UniverseDefinition:
        universe_id = self._normalise_universe_id(universe_id)

        if universe_id == "custom":
            return self._build_custom_universe(
                custom_tickers=custom_tickers,
                as_of_date=as_of_date,
            )

        if universe_id not in self.registry:
            supported = ", ".join(sorted(list(self.registry.keys()) + ["custom"]))
            raise ValueError(f"Unsupported universe_id='{universe_id}'. Supported: {supported}")

        cfg = self.registry[universe_id]
        csv_path = self.universe_dir / cfg.csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(f"Universe file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Universe file {csv_path.name} missing required columns: {sorted(missing)}"
            )

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["sector"] = df["sector"].fillna("Unknown").astype(str).str.strip()

        if "industry" not in df.columns:
            df["industry"] = ""
        else:
            df["industry"] = df["industry"].fillna("").astype(str).str.strip()

        if "rank" in df.columns:
            df = df.sort_values("rank", kind="stable")
        else:
            df = df.sort_values("ticker", kind="stable")

        df = df.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

        if top_n is not None:
            if int(top_n) <= 0:
                raise ValueError("top_n must be positive when provided")
            df = df.iloc[: int(top_n)].copy()

        tickers = df["ticker"].tolist()
        ticker_to_sector = dict(zip(df["ticker"], df["sector"]))
        ticker_to_industry = dict(zip(df["ticker"], df["industry"]))

        snapshot_date = as_of_date or self._infer_snapshot_date_from_filename(cfg.csv_filename)

        metadata = UniverseMetadata(
            universe_id=cfg.universe_id,
            universe_provider=self.PROVIDER_NAME,
            source_note=cfg.source_note,
            selection_method="snapshot_file",
            snapshot_date=snapshot_date,
            historical_constituent_reconstruction=False,
            universe_role=cfg.universe_role,
            requested_top_n=top_n,
            actual_count=len(tickers),
            universe_file=str(csv_path),
        )

        return UniverseDefinition(
            universe_id=cfg.universe_id,
            tickers=tickers,
            ticker_to_sector=ticker_to_sector,
            ticker_to_industry=ticker_to_industry,
            metadata=metadata,
        )

    def _build_custom_universe(
        self,
        custom_tickers: Optional[List[str]],
        as_of_date: Optional[str],
    ) -> UniverseDefinition:
        if not custom_tickers:
            raise ValueError("custom_tickers must be provided for custom universe")

        tickers = [str(t).upper().strip() for t in custom_tickers if str(t).strip()]
        tickers = list(dict.fromkeys(tickers))

        metadata = UniverseMetadata(
            universe_id="custom",
            universe_provider=self.PROVIDER_NAME,
            source_note="user_input",
            selection_method="user_provided",
            snapshot_date=as_of_date or "unspecified",
            historical_constituent_reconstruction=False,
            universe_role="custom",
            requested_top_n=None,
            actual_count=len(tickers),
            universe_file=None,
        )

        return UniverseDefinition(
            universe_id="custom",
            tickers=tickers,
            ticker_to_sector={t: "Unknown" for t in tickers},
            ticker_to_industry={t: "" for t in tickers},
            metadata=metadata,
        )

    @staticmethod
    def _infer_snapshot_date_from_filename(filename: str) -> str:
        stem = Path(filename).stem
        parts = stem.split("_")
        if parts:
            last = parts[-1]
            if len(last) == 10 and last.count("-") == 2:
                return last
        return "unspecified"

    @staticmethod
    def _normalise_universe_id(universe_id: str) -> str:
        raw = str(universe_id).strip().lower()

        aliases = {
            "sp_500": "sp500",
            "s&p500": "sp500",
            "s_and_p_500": "sp500",
            "nasdaq_100": "nasdaq100",
            "ndx": "nasdaq100",
        }
        return aliases.get(raw, raw)