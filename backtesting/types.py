from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    raw_return: float
    net_return: float


@dataclass
class BacktestResult:
    model_name: str
    horizon: int
    equity: List[float] = field(default_factory=lambda: [1.0])
    trade_returns: List[float] = field(default_factory=list)
    trade_times: List[pd.Timestamp] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    executed_indices: List[int] = field(default_factory=list)
    sharpe: Optional[float] = None
    tc_per_side: float = 0.0
    rf_annual: float = 0.0
    signal_threshold: Optional[float] = None

    @property
    def n_trades(self) -> int:
        return len(self.trade_returns)

    @property
    def final_equity(self) -> Optional[float]:
        return self.equity[-1] if self.equity else None

    @property
    def mean_trade_return(self) -> float:
        if not self.trade_returns:
            return 0.0
        return float(np.mean(self.trade_returns))

    @property
    def hit_rate(self) -> float:
        if not self.trade_returns:
            return 0.0
        r = np.asarray(self.trade_returns, dtype=float)
        return float(np.mean(r > 0.0))

    @property
    def max_drawdown(self) -> float:
        if not self.equity:
            return 0.0
        eq = np.asarray(self.equity, dtype=float)
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq / np.maximum(running_max, 1e-12)) - 1.0
        return float(np.min(drawdowns))

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "n_trades": self.n_trades,
            "final_equity": self.final_equity,
            "mean_trade_return": self.mean_trade_return,
            "hit_rate": self.hit_rate,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "tc_per_side": self.tc_per_side,
            "rf_annual": self.rf_annual,
            "signal_threshold": self.signal_threshold,
        }
