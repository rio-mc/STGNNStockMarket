import numpy as np
import pandas as pd
import torch

from models import ModelRegistry
from core.utils.utils import Utils
from data.universe_service import UniverseService


class ExperimentRunner:
    """
    Compatibility runner that restores the state expected by the model runners,
    while sourcing universe metadata from the new data layer.
    """

    def __init__(self, app):
        self.app = app
        self.args = app.args
        self.logger = app.logger
        self.universe_service = UniverseService()

    def run(self, model_name: str, stock: str, state: dict, evaluator, stop_event=None):
        app = self.app

        selected_model = str(model_name).strip().lower()
        self.args.model = selected_model

        self.logger.info("[ExperimentRunner] model=%s stock=%s", selected_model, stock)

        # ====================================
        # STEP 1: Restore compatibility state expected by existing runners
        app.train_feats = state["train_feats"]
        app.val_feats = state["val_feats"]
        app.graphBuilder = state["graphBuilder"]
        app.init_edge_index = state["edge_index"]
        app.horizon = state["horizon"]
        app.seq_len = state["seq_len"]

        state_tickers = list(state.get("tickers", getattr(app.args, "tickers", [])))
        app.args.tickers = state_tickers

        app.min_train_len = min(len(df) for df in app.train_feats.values())
        app.min_val_len = min(len(df) for df in app.val_feats.values())

        # ====================================
        # STEP 2: Universe metadata and sector mapping
        universe_definition = getattr(app, "universe_definition", None)

        if universe_definition is None:
            universe_id = getattr(app.args, "universe_id", "sp500")
            universe_provider = getattr(app.args, "universe_provider", "static_csv")

            try:
                universe_definition = self.universe_service.resolve_definition(
                    universe_id=universe_id,
                    universe_provider=universe_provider,
                    top_n=len(state_tickers) if state_tickers else getattr(app.args, "top_n", None),
                    as_of_date=getattr(app.args, "universe_as_of", None),
                    custom_tickers=state_tickers if universe_id == "custom" else None,
                )
            except Exception as exc:
                self.logger.warning(
                    "[ExperimentRunner] Universe metadata resolution failed for %s via %s: %s",
                    universe_id,
                    universe_provider,
                    exc,
                )
                universe_definition = None

        if universe_definition is not None:
            ticker_to_sector = dict(universe_definition.ticker_to_sector)
            app.universe_info = universe_definition.to_dict()
        else:
            ticker_to_sector = {}

        if hasattr(app.frontendApp, "set_sector_map"):
            app.frontendApp.set_sector_map(ticker_to_sector)
        else:
            app.frontendApp.ticker_to_sector = dict(ticker_to_sector or {})

        # ====================================
        # STEP 3: Restore graph/UI side effects for compatibility
        tickers = state_tickers
        coords3d = state.get("coords")
        pruned = state.get("pruned", [])
        mst = state.get("mst", [])

        plot_tickers = tickers
        plot_coords = coords3d
        plot_pruned = pruned
        plot_mst = mst

        if coords3d is not None:
            if selected_model in ("lstm", "gru"):
                if stock in tickers:
                    target_idx = tickers.index(stock)
                    plot_tickers = [stock]
                    plot_coords = coords3d[target_idx:target_idx + 1]
                else:
                    plot_tickers = [stock]
                    plot_coords = coords3d[:1]

                plot_pruned = []
                plot_mst = []

            elif selected_model in ("panel_gru", "panel_lstm"):
                plot_tickers = tickers
                plot_coords = coords3d
                plot_pruned = []
                plot_mst = []

            else:
                plot_tickers = tickers
                plot_coords = coords3d
                plot_pruned = pruned
                plot_mst = mst

            app.frontendApp.root.after(
                0,
                lambda: app.frontendApp.plot3d_on_ax(
                    tickers=plot_tickers,
                    coords=plot_coords,
                    pruned_edges=plot_pruned,
                    mst_edges=plot_mst,
                )
            )

            if selected_model in ("panel_gru", "panel_lstm", "nnconv", "stgnn"):
                app.frontendApp.root.after(
                    0,
                    lambda s=stock: app.frontendApp.highlight_ticker(s)
                )

        # ====================================
        # STEP 4: Seeded DataLoader generator
        app.dl_gen = torch.Generator(device="cpu")
        seed_for_loaders = getattr(app, "current_seed", int(app.args.seed)) % (2 ** 32)
        app.dl_gen.manual_seed(seed_for_loaders)
        self.logger.info("[ExperimentRunner] dataloader_seed=%d", seed_for_loaders)

        # ====================================
        # STEP 5: Execute selected runner
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