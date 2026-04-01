from __future__ import annotations

from typing import Dict, List, Optional
import logging
from datetime import date


class YahooUniverseProvider:
    """
    Resolves ticker universes and records provenance metadata.

    Notes:
    - This version uses static curated constituent lists.
    - It is suitable for contemporary benchmarking, not historical constituent reconstruction.
    - It avoids fragile live market-cap lookups from Yahoo metadata.
    """

    SP500_CURATED = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
        "JPM", "XOM", "UNH", "V", "COST", "MA", "JNJ", "PG", "HD", "MRK",
        "ABBV", "CVX", "ADBE", "PEP", "KO", "AMD", "BAC", "WMT", "NFLX", "TMO",
        "CRM", "ACN", "MCD", "LIN", "CSCO", "ABT", "CMCSA", "DHR", "INTC", "TXN",
        "WFC", "DIS", "PM", "AMGN", "VZ", "INTU", "UNP", "QCOM", "CAT", "IBM"
    ]

    NASDAQ100_CURATED = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "COST", "NFLX",
        "ADBE", "PEP", "AMD", "CSCO", "TMUS", "INTC", "CMCSA", "QCOM", "AMGN", "TXN",
        "INTU", "HON", "AMAT", "BKNG", "ISRG", "ADP", "VRTX", "GILD", "ADI", "LRCX",
        "PANW", "MU", "KLAC", "MELI", "SNPS", "CDNS", "ASML", "MAR", "CRWD", "FTNT",
        "ORLY", "ABNB", "CTAS", "REGN", "MDLZ", "CSX", "MNST", "KDP", "PAYX", "AEP",
        "DXCM", "ODFL", "WDAY", "MRVL", "ADSK", "PCAR", "ROST", "BIIB", "EA", "XEL",
        "CPRT", "FAST", "DDOG", "TEAM", "CHTR", "FANG", "KHC", "GEHC", "EXC", "IDXX",
        "BKR", "DLTR", "GFS", "LULU", "ON", "CTSH", "MCHP", "AZN", "TTD", "CCEP",
        "ANSS", "ZS", "CDW", "ILMN", "MDB", "SBUX", "PDD", "PYPL", "NXPI", "APP",
        "TTWO", "CEG", "CSGP", "VRSK", "JD", "MSTR", "ARM", "ROP", "CPNG", "DASH"
    ]

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

    def get_universe(
        self,
        universe_name: str,
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
        custom_tickers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Returns a standardised universe payload.
        """
        selection_date = as_of_date or str(date.today())
        universe_name = universe_name.lower()

        if universe_name == "custom":
            if not custom_tickers:
                raise ValueError("custom_tickers must be provided for custom universe")
            tickers = [t.upper() for t in custom_tickers]
            method = "user_provided"
            source_note = "user_input"

        elif universe_name == "sp500_top_n":
            base = self.SP500_CURATED.copy()
            tickers = base[:top_n] if top_n is not None else base
            method = "curated_static_order"
            source_note = "static_curated_sp500_large_caps"

        elif universe_name == "nasdaq100":
            base = self.NASDAQ100_CURATED.copy()
            tickers = base[:top_n] if top_n is not None else base
            method = "membership_list"
            source_note = "static_curated_nasdaq100"

        else:
            raise ValueError(f"Unsupported universe_name: {universe_name}")

        return {
            "tickers": tickers,
            "universe_name": universe_name,
            "selection_date": selection_date,
            "selection_method": method,
            "source": "yfinance_universe_service",
            "requested_top_n": top_n,
            "actual_count": len(tickers),
            "metadata": {
                "source_note": source_note,
                "historical_constituent_reconstruction": False,
            },
        }