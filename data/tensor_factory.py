from typing import Dict, List, Tuple
import pandas as pd
import torch
import numpy as np


class TensorFactory:
    """
    Generate windows for LSTM/GRU and STGNN.

    Important:
    Keep tensors on CPU here.
    Move to CUDA only inside the trainer/model step.
    """

    def __init__(self, tickers, featureCols, seq_len, prediction_horizon, device="cpu"):
        self.tickers = tickers
        self.featureCols = featureCols
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.device = device

    def getLstmAllWindows(
        self,
        features: Dict[str, pd.DataFrame],
        tickers: List[str],
        target_ticker: str,
        feature_cols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.seq_len
        horizon = self.prediction_horizon

        L = min(len(features[t]) for t in tickers)
        arr = features[target_ticker][feature_cols].iloc[: L + horizon].values.astype(np.float32)

        x, y = [], []
        for i in range(0, L - seq_len - horizon + 1):
            x_window = arr[i : i + seq_len]

            current_price = arr[i + seq_len - 1][0]
            future_price = arr[i + seq_len + horizon - 1][0]
            label = 1.0 if future_price > current_price else 0.0

            x.append(x_window)
            y.append(label)

        if x:
            x_tensor = torch.tensor(np.stack(x), dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            x_tensor = torch.empty((0, seq_len, len(feature_cols)), dtype=torch.float32)
            y_tensor = torch.empty((0,), dtype=torch.float32)

        return x_tensor, y_tensor

    def getStgnnAllWindows(
        self,
        features: Dict[str, pd.DataFrame],
        tickers: List[str],
        target_ticker: str,
        feature_cols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.seq_len
        horizon = self.prediction_horizon

        num_nodes = len(tickers)
        target_idx = tickers.index(target_ticker)
        L = min(len(features[t]) for t in tickers)
        close = features[target_ticker]["close"].iloc[: L + horizon].values

        flags = torch.zeros(num_nodes, seq_len, 1, dtype=torch.float32)
        flags[target_idx, :, 0] = 1.0

        stacked_feats = torch.stack([
            torch.tensor(
                features[t][feature_cols].iloc[:L].values,
                dtype=torch.float32
            )
            for t in tickers
        ], dim=1)  # [L, N, F]

        labels = (close[seq_len + horizon - 1:] > close[seq_len - 1 : -horizon]).astype(float)

        X_list, Y_list = [], []
        for i in range(0, L - seq_len - horizon + 1):
            x_window = stacked_feats[i : i + seq_len]   # [T, N, F]
            x_window = x_window.permute(1, 0, 2)        # [N, T, F]
            x_window = torch.cat([x_window, flags], dim=-1)  # [N, T, F+1]

            X_list.append(x_window)
            Y_list.append(torch.tensor(labels[i], dtype=torch.float32))

        if X_list:
            x_tensor = torch.stack(X_list)
            y_tensor = torch.stack(Y_list)
        else:
            x_tensor = torch.empty((0, num_nodes, seq_len, len(feature_cols) + 1), dtype=torch.float32)
            y_tensor = torch.empty((0,), dtype=torch.float32)

        return x_tensor, y_tensor