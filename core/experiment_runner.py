import numpy as np
import pandas as pd
import torch

from models import ModelRegistry
from core.utils.utils import Utils


class ExperimentRunner:
    """
    Transitional experiment runner.

    It takes pipeline state and attaches the minimum compatibility fields
    back onto `app` so the existing per-model runners can execute unchanged.
    """

    def __init__(self, app):
        self.app = app
        self.args = app.args
        self.logger = app.logger

    def run(self, model_name: str, stock: str, state: dict, evaluator, stop_event=None):
        app = self.app

        selected_model = str(model_name).strip().lower()
        self.args.model = selected_model

        self.logger.info("[ExperimentRunner] model=%s stock=%s", selected_model, stock)

        # ====================================
        # STEP 1: Restore compatibility state expected by existing runners
        app.train_feats = state["train_feats"]
        app.val_feats = state["val_feats"]
        app.tf_train = state["tf_train"]
        app.tf_val = state["tf_val"]
        app.graphBuilder = state["graphBuilder"]
        app.init_edge_index = state["edge_index"]
        app.horizon = state["horizon"]
        app.seq_len = state["seq_len"]

        app.min_train_len = min(len(df) for df in app.train_feats.values())
        app.min_val_len = min(len(df) for df in app.val_feats.values())

        # ====================================
        # STEP 2: Restore graph/UI side effects for compatibility
        ticker_to_sector = Utils.load_ticker_to_sector("tickers.csv")
        app.frontendApp.set_sector_map(ticker_to_sector)

        tickers = state.get("tickers", app.args.tickers)
        coords3d = state.get("coords")
        pruned = state.get("pruned", [])
        mst = state.get("mst", [])

        if coords3d is not None:
            app.frontendApp.root.after(
                0,
                lambda: app.frontendApp.plot3d_on_ax(
                    tickers=tickers,
                    coords=coords3d,
                    pruned_edges=pruned,
                    mst_edges=mst,
                )
            )

        latest_feats = []
        for t in app.args.tickers:
            df = app.train_feats.get(t)
            if df is not None and len(df) > 0:
                latest_row = df.iloc[-1]
                latest_feats.append({
                    "return": latest_row.get("return", np.nan),
                    "volatility": latest_row.get("volatility", np.nan),
                    "volume": latest_row.get("volume", np.nan),
                    "momentum": latest_row.get("momentum", np.nan),
                })

        if latest_feats:
            node_df = pd.DataFrame(latest_feats, index=app.args.tickers)
            node_df.index = node_df.index.astype(str)
            app.frontendApp.root.after(0, lambda: app.frontendApp.updateTable(node_df))

        if stop_event is not None:
            app._check_stop(stop_event)

        # ====================================
        # STEP 3: Seeded DataLoader generator
        app.dl_gen = torch.Generator(device="cpu")
        seed_for_loaders = getattr(app, "current_seed", int(app.args.seed)) % (2 ** 32)
        app.dl_gen.manual_seed(seed_for_loaders)
        self.logger.info("[ExperimentRunner] dataloader_seed=%d", seed_for_loaders)

        # ====================================
        # STEP 4: Execute selected runner
        runner = ModelRegistry.get_runner(selected_model)

        result = runner.run(
            app,
            stock=stock,
            price_df=app.raw_feature_dfs[stock],
            evaluator=evaluator,
            stop_event=stop_event,
        )

        if stop_event is not None:
            app._check_stop(stop_event)

        return result