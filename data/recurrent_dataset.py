from typing import Dict
import pandas as pd
from torch.utils.data import Dataset


class RecurrentDataset(Dataset):
    def __init__(
        self,
        tensor_factory,
        df_feats: Dict[str, pd.DataFrame],
        target_ticker: str = "AAPL",
        prediction_horizon: int = 1
    ):
        self.tensor_factory = tensor_factory
        self.df_feats = df_feats
        self.target_ticker = target_ticker
        self.horizon = prediction_horizon
        self.seq_len = tensor_factory.seq_len

        self.x, self.y = tensor_factory.getLstmAllWindows(
            features=self.df_feats,
            tickers=list(self.df_feats.keys()),
            target_ticker=self.target_ticker,
            feature_cols=tensor_factory.featureCols
        )

        df = self.df_feats[self.target_ticker]
        full_index = df.index
        pred_offset = self.seq_len + self.horizon - 1
        self.timestamps = full_index[pred_offset - 1:]
        self.timestamps = self.timestamps[:len(self.x)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx].clone().detach().float(),
            self.y[idx].clone().detach().float()
        )

    def get_timestamp(self, idx):
        return self.timestamps[idx]