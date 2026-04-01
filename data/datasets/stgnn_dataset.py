from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class STGNNDataset(Dataset):
    def __init__(
        self,
        tensor_factory,
        graph_builder,
        feature_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        edge_index: torch.Tensor,
        target_ticker: str,
        horizon: int
    ):
        self.tensor_factory = tensor_factory
        self.feature_dict = feature_dict
        self.tickers = tickers
        self.target_ticker = target_ticker
        self.target_idx = tickers.index(target_ticker)
        self.horizon = horizon

        self.graph_builder = graph_builder
        self.edge_index = edge_index.detach().clone().cpu().to(torch.long)
        self.edge_attr = self.graph_builder.build_edge_weight_tensor(self.edge_index)
        self.edge_attr = self.edge_attr.detach().clone().cpu().to(torch.float32)

        self.x_list, self.y_list = tensor_factory.getStgnnAllWindows(
            features=feature_dict,
            tickers=tickers,
            target_ticker=target_ticker,
            feature_cols=tensor_factory.featureCols
        )

        df = feature_dict[target_ticker]
        self.timestamps = df.index[tensor_factory.seq_len + horizon - 1:]
        self.timestamps = self.timestamps[:len(self.x_list)]

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return Data(
            x=self.x_list[idx].clone().detach().float(),         # [N, T, F]
            edge_index=self.edge_index.clone().contiguous(),     # [2, E]
            y=self.y_list[idx].clone().detach().float(),         # scalar
            edge_attr=self.edge_attr.clone().contiguous()        # [E, 1]
        )

    def get_timestamp(self, idx):
        return self.timestamps[idx]