from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


class TensorFactory:
    """
    Stateless tensor/window construction for recurrent models and STGNNs.

    Important:
    - Tensors remain on CPU here.
    - CUDA transfer should happen only inside trainer/model code.
    - Label definition is directional:
        1.0 if future close > current close else 0.0
    """

    @staticmethod
    def build_recurrent_windows(
        features: Dict[str, pd.DataFrame],
        tickers: List[str],
        target_ticker: str,
        feature_cols: Sequence[str],
        seq_len: int,
        prediction_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, Dict]:
        TensorFactory._validate_inputs(
            features=features,
            tickers=tickers,
            target_ticker=target_ticker,
            feature_cols=feature_cols,
            seq_len=seq_len,
            prediction_horizon=prediction_horizon,
        )

        aligned_tickers, aligned_length = TensorFactory._resolve_aligned_tickers(
            features=features,
            requested_tickers=tickers,
            min_required_length=seq_len + prediction_horizon,
        )

        if target_ticker not in aligned_tickers:
            raise ValueError(
                f"Target ticker '{target_ticker}' is not in aligned_tickers={aligned_tickers}"
            )

        target_df = features[target_ticker].iloc[:aligned_length].copy()
        arr = target_df[list(feature_cols)].to_numpy(dtype=np.float32)
        close = target_df["close"].to_numpy(dtype=np.float32)

        n_samples = aligned_length - seq_len - prediction_horizon + 1
        if n_samples <= 0:
            x_tensor = torch.empty((0, seq_len, len(feature_cols)), dtype=torch.float32)
            y_tensor = torch.empty((0,), dtype=torch.float32)
            timestamps = pd.DatetimeIndex([], dtype="datetime64[ns]")
        else:
            x_list = []
            y_list = []

            for i in range(n_samples):
                x_window = arr[i:i + seq_len]
                current_price = close[i + seq_len - 1]
                future_price = close[i + seq_len + prediction_horizon - 1]
                label = 1.0 if future_price > current_price else 0.0

                x_list.append(x_window)
                y_list.append(label)

            x_tensor = torch.tensor(np.stack(x_list), dtype=torch.float32)
            y_tensor = torch.tensor(y_list, dtype=torch.float32)

            pred_offset = seq_len + prediction_horizon - 1
            timestamps = target_df.index[pred_offset: pred_offset + n_samples]

        metadata = {
            "requested_tickers": list(tickers),
            "aligned_tickers": list(aligned_tickers),
            "requested_count": len(tickers),
            "aligned_count": len(aligned_tickers),
            "dropped_tickers": [t for t in tickers if t not in aligned_tickers],
            "aligned_length": aligned_length,
            "feature_cols": list(feature_cols),
            "seq_len": seq_len,
            "prediction_horizon": prediction_horizon,
            "target_ticker": target_ticker,
            "n_samples": len(x_tensor),
        }
        return x_tensor, y_tensor, timestamps, metadata

    @staticmethod
    def build_stgnn_windows(
        features: Dict[str, pd.DataFrame],
        tickers: List[str],
        target_ticker: str,
        feature_cols: Sequence[str],
        seq_len: int,
        prediction_horizon: int,
        include_target_flag: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, Dict]:
        TensorFactory._validate_inputs(
            features=features,
            tickers=tickers,
            target_ticker=target_ticker,
            feature_cols=feature_cols,
            seq_len=seq_len,
            prediction_horizon=prediction_horizon,
        )

        aligned_tickers, aligned_length = TensorFactory._resolve_aligned_tickers(
            features=features,
            requested_tickers=tickers,
            min_required_length=seq_len + prediction_horizon,
        )

        if target_ticker not in aligned_tickers:
            raise ValueError(
                f"Target ticker '{target_ticker}' is not in aligned_tickers={aligned_tickers}"
            )

        num_nodes = len(aligned_tickers)
        target_idx = aligned_tickers.index(target_ticker)
        target_df = features[target_ticker].iloc[:aligned_length].copy()
        close = target_df["close"].to_numpy(dtype=np.float32)

        stacked_feats = torch.stack(
            [
                torch.tensor(
                    features[t][list(feature_cols)].iloc[:aligned_length].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )
                for t in aligned_tickers
            ],
            dim=1,
        )  # [L, N, F]

        n_samples = aligned_length - seq_len - prediction_horizon + 1
        feature_dim = len(feature_cols) + (1 if include_target_flag else 0)

        if include_target_flag:
            flags = torch.zeros(num_nodes, seq_len, 1, dtype=torch.float32)
            flags[target_idx, :, 0] = 1.0
        else:
            flags = None

        if n_samples <= 0:
            x_tensor = torch.empty((0, num_nodes, seq_len, feature_dim), dtype=torch.float32)
            y_tensor = torch.empty((0,), dtype=torch.float32)
            timestamps = pd.DatetimeIndex([], dtype="datetime64[ns]")
        else:
            labels = (
                close[seq_len + prediction_horizon - 1:] >
                close[seq_len - 1:-prediction_horizon]
            ).astype(float)

            x_list = []
            y_list = []

            for i in range(n_samples):
                x_window = stacked_feats[i:i + seq_len]    # [T, N, F]
                x_window = x_window.permute(1, 0, 2)       # [N, T, F]

                if include_target_flag:
                    x_window = torch.cat([x_window, flags], dim=-1)

                x_list.append(x_window)
                y_list.append(torch.tensor(labels[i], dtype=torch.float32))

            x_tensor = torch.stack(x_list)
            y_tensor = torch.stack(y_list)

            pred_offset = seq_len + prediction_horizon - 1
            timestamps = target_df.index[pred_offset: pred_offset + n_samples]

        metadata = {
            "requested_tickers": list(tickers),
            "aligned_tickers": list(aligned_tickers),
            "requested_count": len(tickers),
            "aligned_count": len(aligned_tickers),
            "dropped_tickers": [t for t in tickers if t not in aligned_tickers],
            "aligned_length": aligned_length,
            "feature_cols": list(feature_cols),
            "seq_len": seq_len,
            "prediction_horizon": prediction_horizon,
            "target_ticker": target_ticker,
            "target_idx": target_idx,
            "include_target_flag": include_target_flag,
            "n_samples": len(x_tensor),
        }
        return x_tensor, y_tensor, timestamps, metadata

    @staticmethod
    def _resolve_aligned_tickers(
        features: Dict[str, pd.DataFrame],
        requested_tickers: List[str],
        min_required_length: int,
    ) -> Tuple[List[str], int]:
        aligned_tickers: List[str] = []
        lengths: List[int] = []

        for ticker in requested_tickers:
            df = features.get(ticker)
            if df is None or df.empty:
                continue
            if len(df) < min_required_length:
                continue
            aligned_tickers.append(ticker)
            lengths.append(len(df))

        if not aligned_tickers:
            raise ValueError(
                "No aligned tickers available after length filtering. "
                f"min_required_length={min_required_length}"
            )

        aligned_length = min(lengths)
        return aligned_tickers, aligned_length

    @staticmethod
    def _validate_inputs(
        features: Dict[str, pd.DataFrame],
        tickers: List[str],
        target_ticker: str,
        feature_cols: Sequence[str],
        seq_len: int,
        prediction_horizon: int,
    ) -> None:
        if not tickers:
            raise ValueError("tickers must be non-empty")
        if target_ticker not in tickers:
            raise ValueError(f"target_ticker='{target_ticker}' not present in tickers")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty")

        missing_tickers = [t for t in tickers if t not in features]
        if missing_tickers:
            raise ValueError(f"Missing feature frames for tickers: {missing_tickers}")

        for ticker in tickers:
            df = features[ticker]
            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Ticker '{ticker}' missing feature columns: {missing_cols}"
                )
            if "close" not in df.columns:
                raise ValueError(f"Ticker '{ticker}' missing required 'close' column")