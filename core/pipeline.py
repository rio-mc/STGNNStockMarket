import pandas as pd
import torch

from features.feature_extractor import FeatureExtractor
from graph.graph_builder import GraphBuilder
from core.utils.utils import Utils


class Pipeline:
    def __init__(self, app):
        self.app = app
        self.args = app.args
        self.logger = app.logger

    def run(self, stock: str, gui_window: str, stop_event=None):
        """
        Runs the full data + graph + tensor pipeline.
        Returns a state object (dict) used by model runners.
        """

        app = self.app

        # ====================================
        # STEP 1: Determine horizon
        price_df = app.raw_feature_dfs[stock]

        deltas = price_df.index.to_series().diff().dropna()
        if deltas.empty:
            raise RuntimeError("Not enough data to infer sampling interval")

        mode = deltas.mode()
        sample_interval = mode.iloc[0] if not mode.empty else deltas.median()
        bars_per_day = max(1, int(pd.Timedelta("1D") / sample_interval))

        horizon = Utils.parse_window(gui_window, bars_per_day)
        seq_len = int(app.args.seq_len)

        # Make the state explicit early so downstream helpers never depend
        # on an attribute that has not yet been attached back onto MainApp.
        app.horizon = horizon
        app.seq_len = seq_len

        if stop_event is not None:
            app._check_stop(stop_event)

        # ====================================
        # STEP 2: Train / validation split
        shared_index = set.intersection(
            *(set(df.index) for df in app.raw_feature_dfs.values())
        )
        shared_index = sorted(shared_index)

        if len(shared_index) < 2:
            raise RuntimeError("Not enough shared timestamps across tickers to build a split.")

        cutoff_idx = int(0.7 * len(shared_index))
        cutoff_idx = min(max(cutoff_idx, 1), len(shared_index) - 1)
        cutoff_date = shared_index[cutoff_idx]

        embargo_bars = max(1, int(horizon))
        train_end_idx = max(0, cutoff_idx - embargo_bars)
        train_end_date = shared_index[train_end_idx]

        train_raw_map = {
            t: df[df.index < train_end_date].copy()
            for t, df in app.raw_feature_dfs.items()
        }
        val_raw_map = {
            t: df[df.index >= cutoff_date].copy()
            for t, df in app.raw_feature_dfs.items()
        }

        if stop_event is not None:
            app._check_stop(stop_event)

        # ====================================
        # STEP 3: Feature engineering
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

        train_feats = {
            t: df.iloc[-min_train_len:].copy()
            for t, df in train_feats.items()
        }
        val_feats = {
            t: df.iloc[-min_val_len:].copy()
            for t, df in val_feats.items()
        }

        if stop_event is not None:
            app._check_stop(stop_event)

        # ====================================
        # STEP 4: Tensor factories
        tf_train = app.build_tensor_factory(horizon=horizon, seq_len=seq_len)
        tf_val = app.build_tensor_factory(horizon=horizon, seq_len=seq_len)

        # ====================================
        # STEP 5: Graph construction
        ticker_to_sector = Utils.load_ticker_to_sector("tickers.csv")

        graphBuilder = GraphBuilder(
            dfFeats=train_feats,
            max_k=app.get_max_k(),
            n_pca=3,
            ticker_to_sector=ticker_to_sector,
            graph_embed=self.args.graph_embed,
            ablate_feature=self.args.ablate_feature,
        )

        tickers, coords3d, pruned, mst = graphBuilder.getLightGraph()

        edge_pairs = [(i, j) for i, j, _ in pruned]
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        if edge_index.numel() == 0:
            idx = torch.arange(len(tickers), dtype=torch.long)
            edge_index = torch.stack([idx, idx], dim=0)

        if stop_event is not None:
            app._check_stop(stop_event)

        # ====================================
        # STEP 6: Return pipeline state
        return {
            "train_feats": train_feats,
            "val_feats": val_feats,
            "tf_train": tf_train,
            "tf_val": tf_val,
            "graphBuilder": graphBuilder,
            "edge_index": edge_index,
            "tickers": tickers,
            "coords": coords3d,
            "pruned": pruned,
            "mst": mst,
            "horizon": horizon,
            "seq_len": seq_len,
        }