from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class LSTMDataset(Dataset):
    def __init__(
        self,
        tensor_factory,
        df_feats: Dict[str, pd.DataFrame],
        target_ticker: str = "AAPL",
        prediction_horizon: int = 1
    ):
        # === STEP 1: Initialisation for LSTM ===
        # ------------------------------------

        #   1. Feature and tensor initialisation
        self.tensor_factory = tensor_factory
        self.df_feats = df_feats

        #   2. Stock initialisation
        self.target_ticker = target_ticker
        self.horizon = prediction_horizon
        self.seq_len = tensor_factory.seq_len

        # === STEP 2: Feature Tensors And Labels ===
        # ------------------------------------
        self.x, self.y = tensor_factory.getLstmAllWindows(
            features=self.df_feats,
            tickers=list(self.df_feats.keys()),
            target_ticker=self.target_ticker,
            feature_cols=tensor_factory.featureCols
        )

        # === STEP 3: Timestamp Extraction ===
        # ------------------------------------
        df = self.df_feats[self.target_ticker]
        full_index = df.index
        pred_offset = self.seq_len + self.horizon - 1
        self.timestamps = full_index[pred_offset - 1:]
        self.timestamps = self.timestamps[:len(self.x)]

    def __len__(self):
        # ====================================
		# === Helper to pass dataset length
        return len(self.x)

    def __getitem__(self, idx):
        # ====================================
		# === Helper to get data
        return (
            self.x[idx].clone().detach().float(),  # [T, F]
            self.y[idx].clone().detach().float()  # [1]
        )

    def get_timestamp(self, idx):
        # ====================================
		# === Helper to collect data timestamp
        return self.timestamps[idx]
    
class STGNNDataset(Dataset):
    def __init__(self,
                 tensor_factory,
                 graph_builder,
                 feature_dict: Dict[str, pd.DataFrame],
                 tickers: List[str],
                 edge_index: torch.Tensor,
                 target_ticker: str,
                 horizon: int):
        # === STEP 1: Initialisation for STGNN ===
        # ------------------------------------

        #   1. Feature and tensor initialisation
        self.tensor_factory = tensor_factory
        self.feature_dict = feature_dict

        #   2. Stock initialisation
        self.tickers = tickers
        self.target_ticker = target_ticker
        self.target_idx = tickers.index(target_ticker)
        self.horizon = horizon

        #   3. Graph initialisation
        self.graph_builder = graph_builder
        self.edge_index = edge_index.detach().clone().cpu().to(torch.long)          # [2, E] long
        self.edge_attr  = self.graph_builder.build_edge_weight_tensor(self.edge_index)
        self.edge_attr  = self.edge_attr.detach().clone().cpu().to(torch.float32)   # [E, F]

        # === STEP 2: Feature Tensors And Labels ===
        # ------------------------------------
        self.x_list, self.y_list = tensor_factory.getStgnnAllWindows(
            features=feature_dict,
            tickers=tickers,
            target_ticker=target_ticker,
            feature_cols=tensor_factory.featureCols
        )

        # === STEP 3: Timestamp Extraction ===
        # ------------------------------------
        df = feature_dict[target_ticker]
        self.timestamps = df.index[tensor_factory.seq_len + horizon - 1:]
        self.timestamps = self.timestamps[:len(self.x_list)]

    def __len__(self):
        # ====================================
		# === Helper to pass dataset length
        return len(self.x_list)

    def __getitem__(self, idx):
        # ====================================
        # === Helper to get data
        return Data(
            x=self.x_list[idx].clone().detach().float(),                   # [N, T, F]
            edge_index=self.edge_index.clone().contiguous(),               # [2, E] long, CPU
            y=self.y_list[idx].clone().detach().float(),                   # [1]
            edge_attr=self.edge_attr.clone().contiguous()                  # [E, F] float, CPU
        )

    def get_timestamp(self, idx):
        # ====================================
		# === Helper to collect data timestamp
        return self.timestamps[idx]
