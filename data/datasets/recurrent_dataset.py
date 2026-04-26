from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.tensor_factory import TensorFactory


class RecurrentDataset(Dataset):
    """
    Dataset wrapper for LSTM/GRU-style models.

    This dataset is intentionally thin:
    - TensorFactory handles window generation and alignment
    - Dataset handles indexing and timestamp retrieval
    """

    def __init__(
        self,
        feature_dict: Dict[str, pd.DataFrame],
        tickers: Optional[List[str]] = None,
        target_ticker: str = "AAPL",
        feature_cols: Optional[List[str]] = None,
        seq_len: int = 32,
        prediction_horizon: int = 1,
    ) -> None:
        self.feature_dict = feature_dict
        self.requested_tickers = tickers or list(feature_dict.keys())
        self.target_ticker = target_ticker
        self.feature_cols = feature_cols or ["close", "return", "volatility", "momentum"]
        self.seq_len = int(seq_len)
        self.horizon = int(prediction_horizon)

        self.x, self.y, self.timestamps, self.metadata = TensorFactory.build_recurrent_windows(
            features=self.feature_dict,
            tickers=self.requested_tickers,
            target_ticker=self.target_ticker,
            feature_cols=self.feature_cols,
            seq_len=self.seq_len,
            prediction_horizon=self.horizon,
        )

        self.aligned_tickers = list(self.metadata["aligned_tickers"])
        assert len(self.x) == len(self.y)
        
    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            self.x[idx].clone().detach().float(),
            self.y[idx].clone().detach().float(),
        )

    def get_timestamp(self, idx: int):
        return self.timestamps[idx]