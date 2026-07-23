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
            self.logger.propagate = False

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
        self.args.effective_interval = Utils.infer_interval_label(price_df.index)

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

        train_ratio = float(getattr(self.args, "train_split_ratio", 0.6))
        validation_ratio = float(getattr(self.args, "validation_split_ratio", 0.2))
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_split_ratio must be > 0 and < 1")
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_split_ratio must be > 0 and < 1")
        if train_ratio + validation_ratio >= 1.0:
            raise ValueError("train_split_ratio + validation_split_ratio must be < 1")

        train_cutoff_idx = int(train_ratio * len(shared_index))
        test_cutoff_idx = int((train_ratio + validation_ratio) * len(shared_index))
        embargo_bars = max(1, int(horizon))
        train_end_idx = train_cutoff_idx - embargo_bars
        validation_end_idx = test_cutoff_idx - embargo_bars

        if not (0 < train_end_idx < train_cutoff_idx < validation_end_idx < test_cutoff_idx < len(shared_index)):
            raise RuntimeError(
                "Not enough shared timestamps for purged train/validation/test splits: "
                f"n={len(shared_index)}, horizon={horizon}, "
                f"train_ratio={train_ratio}, validation_ratio={validation_ratio}."
            )

        train_end_date = shared_index[train_end_idx]
        validation_start_date = shared_index[train_cutoff_idx]
        validation_end_date = shared_index[validation_end_idx]
        test_start_date = shared_index[test_cutoff_idx]

        train_raw_map = {ticker: df[df.index < train_end_date].copy() for ticker, df in self.raw_feature_dfs.items()}
        val_raw_map = {
            ticker: df[(df.index >= validation_start_date) & (df.index < validation_end_date)].copy()
            for ticker, df in self.raw_feature_dfs.items()
        }
        test_raw_map = {
            ticker: df[df.index >= test_start_date].copy()
            for ticker, df in self.raw_feature_dfs.items()
        }

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

        feat_ext_test = FeatureExtractor(
            test_raw_map,
            rollingVolWindow=seq_len,
            norm_stats=train_norm_stats,
            fit_normaliser=False,
            ablate_feature=self.args.ablate_feature,
        )
        feat_ext_test.buildFeatureDfs()
        test_feats = feat_ext_test.dfFeats

        common_feature_tickers = sorted(
            set(train_feats) & set(val_feats) & set(test_feats)
        )
        if stock not in common_feature_tickers:
            raise RuntimeError(
                f"Target ticker '{stock}' is not available in every temporal split after "
                "feature extraction."
            )
        if not common_feature_tickers:
            raise RuntimeError(
                "Feature extraction produced no tickers shared by train, validation and test."
            )

        dropped_during_feature_extraction = sorted(
            ticker for ticker in self.raw_feature_dfs if ticker not in common_feature_tickers
        )
        train_feats = {ticker: train_feats[ticker] for ticker in common_feature_tickers}
        val_feats = {ticker: val_feats[ticker] for ticker in common_feature_tickers}
        test_feats = {ticker: test_feats[ticker] for ticker in common_feature_tickers}

        min_train_len = min(len(df) for df in train_feats.values())
        min_val_len = min(len(df) for df in val_feats.values())
        min_test_len = min(len(df) for df in test_feats.values())
        data_quality = {
            "requested_tickers": list(getattr(self.args, "tickers", list(self.raw_feature_dfs.keys()))),
            "raw_loaded_tickers": sorted(list(self.raw_feature_dfs.keys())),
            "raw_loaded_count": len(self.raw_feature_dfs),
            "shared_timestamp_count": len(shared_index),
            # cutoff_date is retained as a backwards-compatible alias for
            # the start of validation.
            "cutoff_date": pd.Timestamp(validation_start_date).isoformat(),
            "train_end_date": pd.Timestamp(train_end_date).isoformat(),
            "validation_start_date": pd.Timestamp(validation_start_date).isoformat(),
            "validation_end_date": pd.Timestamp(validation_end_date).isoformat(),
            "test_start_date": pd.Timestamp(test_start_date).isoformat(),
            "embargo_bars": int(embargo_bars),
            "train_split_ratio": train_ratio,
            "validation_split_ratio": validation_ratio,
            "test_split_ratio": 1.0 - train_ratio - validation_ratio,
            "train_feature_tickers": sorted(list(train_feats.keys())),
            "val_feature_tickers": sorted(list(val_feats.keys())),
            "test_feature_tickers": sorted(list(test_feats.keys())),
            "train_feature_count": len(train_feats),
            "val_feature_count": len(val_feats),
            "test_feature_count": len(test_feats),
            "min_train_len": int(min_train_len),
            "min_val_len": int(min_val_len),
            "min_test_len": int(min_test_len),
            "dropped_before_pipeline": [
                ticker
                for ticker in list(getattr(self.args, "tickers", []))
                if ticker not in self.raw_feature_dfs
            ],
            "dropped_during_feature_extraction": dropped_during_feature_extraction,
        }
        train_feats = {ticker: df.iloc[-min_train_len:].copy() for ticker, df in train_feats.items()}
        val_feats = {ticker: df.iloc[-min_val_len:].copy() for ticker, df in val_feats.items()}
        test_feats = {ticker: df.iloc[-min_test_len:].copy() for ticker, df in test_feats.items()}

        self._check_stop(stop_event)

        tf_train = self._build_tensor_factory(horizon=horizon, seq_len=seq_len)
        tf_val = self._build_tensor_factory(horizon=horizon, seq_len=seq_len)
        tf_test = self._build_tensor_factory(horizon=horizon, seq_len=seq_len)

        ticker_to_sector = self._resolve_ticker_to_sector()
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

        graph_stats_path = (
            Path(getattr(self.args, "results_dir", "./results"))
            / "graph_logging"
            / f"graph_stats_{stock}_{self.args.model}_k{max_k}_seed{self.args.seed}.json"
        )
        graph_builder.save_graph_stats(str(graph_stats_path))

        # Build PyG edge_index.
        # GraphBuilder returns conceptual undirected edges, but PyG message
        # passing is directional, so add reverse edges explicitly.
        edge_pairs = [(i, j) for i, j, _ in pruned]

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

            reverse_edges = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, reverse_edges], dim=1)

            num_nodes = len(tickers)
            edge_keys = edge_index[0] * num_nodes + edge_index[1]
            unique_keys = torch.unique(edge_keys, sorted=True)

            edge_index = torch.stack(
                [
                    unique_keys // num_nodes,
                    unique_keys % num_nodes,
                ],
                dim=0,
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph_ablation = getattr(self.args, "graph_ablation", "none")
        edge_index = Utils.apply_graph_ablation(
            edge_index=edge_index,
            num_nodes=len(tickers),
            mode=graph_ablation,
        ).cpu()

        # If no cross-asset edges remain, retain identity self-loops so
        # graph operators remain well-defined without relational mixing.
        if edge_index.numel() == 0 and str(graph_ablation).strip().lower() != "empty":
            idx = torch.arange(len(tickers), dtype=torch.long)
            edge_index = torch.stack([idx, idx], dim=0)

        self._check_stop(stop_event)

        state = {
            "raw_feature_dfs": self.raw_feature_dfs,
            "train_feats": train_feats,
            "val_feats": val_feats,
            "test_feats": test_feats,
            "tf_train": tf_train,
            "tf_val": tf_val,
            "tf_test": tf_test,
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
            "ticker_to_sector": ticker_to_sector,
            "data_quality": data_quality,
            "target_ticker": stock,
            "target_stock": stock,
        }
        return state

    def _resolve_ticker_to_sector(self) -> Dict[str, str]:
        if self.app is not None:
            universe_definition = getattr(self.app, "universe_definition", None)
            sector_map = getattr(universe_definition, "ticker_to_sector", None)
            if sector_map:
                return {str(t).upper(): str(s) for t, s in sector_map.items()}

            universe_info = getattr(self.app, "universe_info", None)
            if isinstance(universe_info, dict) and universe_info.get("ticker_to_sector"):
                return {
                    str(t).upper(): str(s)
                    for t, s in universe_info["ticker_to_sector"].items()
                }

        sector_map = getattr(self.args, "ticker_to_sector", None)
        if sector_map:
            return {str(t).upper(): str(s) for t, s in sector_map.items()}

        return Utils.load_ticker_to_sector("sp500_tickers.csv")

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
