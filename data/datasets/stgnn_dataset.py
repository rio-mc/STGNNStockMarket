from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data.tensor_factory import TensorFactory


class STGNNDataset(Dataset):
    """
    Dataset wrapper for STGNN-style graph-temporal models.

    Important:
    - Ticker alignment is resolved before graph sample construction
    - The dataset uses only the aligned ticker set
    """

    def __init__(
        self,
        graph_builder,
        feature_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        edge_index: torch.Tensor,
        target_ticker: str,
        feature_cols: Optional[List[str]] = None,
        seq_len: int = 32,
        horizon: int = 1,
        include_target_flag: bool = True,
    ) -> None:
        self.graph_builder = graph_builder
        self.feature_dict = feature_dict
        self.requested_tickers = list(tickers)
        self.target_ticker = target_ticker
        self.feature_cols = feature_cols or ["close", "return", "volatility", "momentum"]
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.include_target_flag = bool(include_target_flag)

        x_tensor, y_tensor, timestamps, metadata = TensorFactory.build_stgnn_windows(
            features=self.feature_dict,
            tickers=self.requested_tickers,
            target_ticker=self.target_ticker,
            feature_cols=self.feature_cols,
            seq_len=self.seq_len,
            prediction_horizon=self.horizon,
            include_target_flag=self.include_target_flag,
        )

        self.x_list = x_tensor
        self.y_list = y_tensor
        self.timestamps = timestamps
        self.metadata = metadata
        self.tickers = list(metadata["aligned_tickers"])

        if self.target_ticker not in self.tickers:
            raise ValueError(
                f"Target ticker '{self.target_ticker}' missing after alignment."
            )

        self.target_idx = self.tickers.index(self.target_ticker)

        self.edge_index = edge_index.detach().clone().cpu().to(torch.long)
        self.edge_attr = self.graph_builder.build_edge_weight_tensor(self.edge_index)
        self.edge_attr = self.edge_attr.detach().clone().cpu().to(torch.float32)
        assert len(self.x_list) == len(self.y_list)

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int):
        return Data(
            x=self.x_list[idx].clone().detach().float(),
            edge_index=self.edge_index.clone().contiguous(),
            y=self.y_list[idx].clone().detach().float(),
            edge_attr=self.edge_attr.clone().contiguous(),
        )

    def get_timestamp(self, idx: int):
        return self.timestamps[idx]