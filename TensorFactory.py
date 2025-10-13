from typing import Dict, List, Tuple
import pandas as pd
import torch
import numpy as np

class TensorFactory:
    """
    Generate windows for LSTM and STGNN.

    LSTM is per-stock.
    STGNN is across all stocks with an extra channel indicating the target node.
    """
    def __init__(self, tickers, featureCols, seq_len, prediction_horizon, device='cuda'):
        # === STEP 1: Initialise Parameters ===
        # ------------------------------------
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
        # === STEP 1: Set Window Parameters For LSTM ===
        # ------------------------------------
        
        #   1. Set window parameters
        seq_len = self.seq_len
        horizon = self.prediction_horizon

        #   2. Set stock parameters
        L = min(len(features[t]) for t in tickers)
        arr = features[target_ticker][feature_cols].iloc[:L + horizon].values.astype(np.float32)

        # === STEP 2: Create Window For Chosen Stock ===
        # ------------------------------------
        x, y = [], []
        for i in range(0, L - seq_len - horizon + 1):
            #   1. Set window size equal to sequence length
            x_window = arr[i : i + seq_len]

            #   2. Create label by price comparison
            current_price = arr[i + seq_len - 1][0]
            future_price = arr[i + seq_len + horizon - 1][0]
            label = 1.0 if future_price > current_price else 0.0

            #   3. Append to list
            x.append(x_window)
            y.append(label)

        # === STEP 3: Convert to PyTorch Tensors ===
        # ------------------------------------
        x_tensor = torch.tensor(np.stack(x), dtype=torch.float32, device=self.device) if x else torch.empty(0)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device) if y else torch.empty(0)

        return x_tensor, y_tensor
        
    def getStgnnAllWindows(
        self,
        features: Dict[str, pd.DataFrame],
        tickers: List[str],
        target_ticker: str,
        feature_cols: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # === STEP 1: Set Window Parameters For STGNN ===
        # ------------------------------------
        
        #   1. Set window parameters
        seq_len = self.seq_len
        horizon = self.prediction_horizon

        #   2. Establish stock parameters
        num_nodes = len(tickers)
        target_idx = tickers.index(target_ticker)
        L = min(len(features[t]) for t in tickers)
        close = features[target_ticker]['close'].iloc[: L + horizon].values

        #   3. Set flag for stock to be predicted over
        flags = torch.zeros(num_nodes, seq_len, 1, device=self.device)
        flags[target_idx, :, 0] = 1.0

        # === STEP 2: Stack Features ===
        # ------------------------------------
        stacked_feats = torch.stack([
            torch.tensor(
                features[t][feature_cols].iloc[:L].values,
                dtype=torch.float32,
                device=self.device
            )
            for t in tickers
        ], dim=1)  # [L, N, F]
    
        # === STEP 3: Create Window For All Stocks ===
        # ------------------------------------

        #   1. Choose labels for chosen stock
        labels = (close[seq_len + horizon - 1:] > close[seq_len - 1 : -horizon]).astype(float)
        X_list, Y_list = [], []
        for i in range(0, L - seq_len - horizon + 1):
            #   1. Set window size equal to sequence length
            x_window = stacked_feats[i : i + seq_len]           # [T, N, F]

            #   2. Rearrange tensor for passing
            x_window = x_window.permute(1, 0, 2)                # [N, T, F]

            #   3. Add flags to identfiy chosen stock during training
            x_window = torch.cat([x_window, flags], dim=-1)     # [N, T, F+1]
            
            #   4. Append to list
            X_list.append(x_window)
            Y_list.append(torch.tensor(labels[i], dtype=torch.float32, device=self.device))

        # === STEP 4: Convert to PyTorch Tensors ===
        # ------------------------------------
        x_tensor = torch.stack(X_list) if X_list else torch.empty(0)
        y_tensor = torch.stack(Y_list) if Y_list else torch.empty(0)

        return x_tensor, y_tensor
