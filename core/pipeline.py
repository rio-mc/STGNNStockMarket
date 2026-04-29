import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from core.utils.utils import Utils
from features.feature_extractor import FeatureExtractor
from graph.graph_builder import GraphBuilder


class Pipeline:
    """
    Runs data splitting, feature engineering, graph construction, and tensor config.

    Can be initialised with either:
    - MainApp instance for GUI compatibility
    - argparse namespace plus raw_feature_dfs for headless mode
    """

    def __init__(self, app_or_args, raw_feature_dfs: Optional[Dict[str, pd.DataFrame]] = None):
        if hasattr(app_or_args, "args"):
            self.app = app_or_args
            self.args = app_or_args.args
            self.logger = app_or_args.logger
            self.raw_feature_dfs = app_or_args.raw_feature_dfs
        else:
            self.app = None
            self.args = app_or_args
            self.raw_feature_dfs = raw_feature_dfs
            self.logger = logging.getLogger("Pipeline")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        if not self.raw_feature_dfs:
            raise RuntimeError("Pipeline requires non-empty raw_feature_dfs.")

    def run(self, stock: str, gui_window: str, stop_event=None):
        stock = str(stock).strip().upper()
        if stock not in self.raw_feature_dfs:
            raise ValueError(f"Target stock '{stock}' not found in raw_feature_dfs.")

        price_df = self.raw_feature_dfs[stock]
        deltas = price_df.index.to_series().diff().dropna()
        if deltas.empty:
            raise RuntimeError("Not enough data to infer sampling interval")

        mode = deltas.mode()
        sample_interval = mode.iloc[0] if not mode.empty else deltas.median()
        bars_per_day = max(1, int(pd.Timedelta("1D") / sample_interval))

        horizon = Utils.parse_window(gui_window, bars_per_day)
        seq_len = int(self.args.seq_len)

        # Single source of truth: graph_window is always seq_len.
        graph_window = seq_len
        self.args.graph_window = graph_window

        if self.app is not None:
            self.app.horizon = horizon
            self.app.seq_len = seq_len
            self.app.graph_window = graph_window

        self._check_stop(stop_event)

        shared_index = set.intersection(*(set(df.index) for df in self.raw_feature_dfs.values()))
        shared_index = sorted(shared_index)
        if len(shared_index) < 2:
            raise RuntimeError("Not enough shared timestamps across tickers to build a split.")

        cutoff_idx = int(0.7 * len(shared_index))
        cutoff_idx = min(max(cutoff_idx, 1), len(shared_index) - 1)
        cutoff_date = shared_index[cutoff_idx]

        embargo_bars = max(1, int(horizon))
        train_end_idx = max(0, cutoff_idx - embargo_bars)
        train_end_date = shared_index[train_end_idx]

        train_raw_map = {ticker: df[df.index < train_end_date].copy() for ticker, df in self.raw_feature_dfs.items()}
        val_raw_map = {ticker: df[df.index >= cutoff_date].copy() for ticker, df in self.raw_feature_dfs.items()}

        self._check_stop(stop_event)

        feat_ext_train = FeatureExtractor(
            train_raw_map,
            rollingVolWindow=seq_len,
            norm_stats=None,
            fit_normaliser=True,
            ablate_feature=self.args.ablate_feature,
        )
        feat_ext_train.buildFeatureDfs()
        train_feats = feat_ext_train.dfFeats
        train_norm_stats = feat_ext_train.get_norm_stats()

        feat_ext_val = FeatureExtractor(
            val_raw_map,
            rollingVolWindow=seq_len,
            norm_stats=train_norm_stats,
            fit_normaliser=False,
            ablate_feature=self.args.ablate_feature,
        )
        feat_ext_val.buildFeatureDfs()
        val_feats = feat_ext_val.dfFeats

        if not train_feats:
            raise RuntimeError("Feature extraction produced an empty training feature map.")
        if not val_feats:
            raise RuntimeError("Feature extraction produced an empty validation feature map.")

        min_train_len = min(len(df) for df in train_feats.values())
        min_val_len = min(len(df) for df in val_feats.values())
        train_feats = {ticker: df.iloc[-min_train_len:].copy() for ticker, df in train_feats.items()}
        val_feats = {ticker: df.iloc[-min_val_len:].copy() for ticker, df in val_feats.items()}

        self._check_stop(stop_event)

        tf_train = self._build_tensor_factory(horizon=horizon, seq_len=seq_len)
        tf_val = self._build_tensor_factory(horizon=horizon, seq_len=seq_len)

        ticker_to_sector = Utils.load_ticker_to_sector("sp500_tickers.csv")
        max_k = int(getattr(self.args, "k", 10))

        graph_builder = GraphBuilder(
            dfFeats=train_feats,
            max_k=max_k,
            n_pca=3,
            ticker_to_sector=ticker_to_sector,
            graph_embed=self.args.graph_embed,
            ablate_feature=self.args.ablate_feature,
            graph_window=graph_window,
            graph_mode=self.args.graph_mode,
        )

        tickers, coords3d, pruned, mst = graph_builder.getLightGraph()
        graph_stats_path = Path(getattr(self.args, "results_dir", "./results")) / "graph_logging" / "graph_stats.json"
        graph_builder.save_graph_stats(str(graph_stats_path))

        edge_pairs = [(i, j) for i, j, _ in pruned]
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        if edge_index.numel() == 0:
            idx = torch.arange(len(tickers), dtype=torch.long)
            edge_index = torch.stack([idx, idx], dim=0)

        self._check_stop(stop_event)

        state = {
            "train_feats": train_feats,
            "val_feats": val_feats,
            "tf_train": tf_train,
            "tf_val": tf_val,
            "graphBuilder": graph_builder,
            "edge_index": edge_index,
            "tickers": tickers,
            "coords": coords3d,
            "pruned": pruned,
            "mst": mst,
            "horizon": horizon,
            "seq_len": seq_len,
            "graph_window": graph_window,
            "graph_stats": graph_builder.graph_stats,
            "graph_stats_path": str(graph_stats_path),
            "target_ticker": stock,
            "target_stock": stock,
        }
        return state

    def _check_stop(self, stop_event):
        if stop_event is not None and self.app is not None:
            self.app._check_stop(stop_event)

    def _build_tensor_factory(self, horizon: int, seq_len: int = None):
        engineered = ["return", "volatility", "momentum"]
        if getattr(self.args, "ablate_feature", "none") in engineered:
            engineered = [f for f in engineered if f != self.args.ablate_feature]
        feature_cols = ["close"] + engineered

        class ConfiguredTensorFactory:
            def __init__(self, horizon, seq_len, feature_cols):
                self.horizon = int(horizon)
                self.seq_len = int(seq_len)
                self.feature_cols = list(feature_cols)

            def build_recurrent_windows(self, features, tickers, target_ticker):
                from data.tensor_factory import TensorFactory
                return TensorFactory.build_recurrent_windows(
                    features=features,
                    tickers=tickers,
                    target_ticker=target_ticker,
                    feature_cols=self.feature_cols,
                    seq_len=self.seq_len,
                    prediction_horizon=self.horizon,
                )

            def build_stgnn_windows(self, features, tickers, target_ticker):
                from data.tensor_factory import TensorFactory
                return TensorFactory.build_stgnn_windows(
                    features=features,
                    tickers=tickers,
                    target_ticker=target_ticker,
                    feature_cols=self.feature_cols,
                    seq_len=self.seq_len,
                    prediction_horizon=self.horizon,
                )

        return ConfiguredTensorFactory(horizon, seq_len, feature_cols)
