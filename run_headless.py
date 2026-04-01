from core.experiment_runner import ExperimentRunner
from core.pipeline import Pipeline


def run_headless(self, stock: str = None, gui_window: str = None, model_name: str = None):
    """
    Run a single experiment without entering the Tk mainloop.

    Transitional mode:
    - still instantiates FrontEnd because current evaluator / plotting /
      runner code depends on it
    - does NOT start the GUI event loop
    """

    # ====================================
    # STEP 1: Resolve defaults
    selected_stock = stock or (self.args.tickers[0] if self.args.tickers else None)
    if not selected_stock:
        raise RuntimeError("No stock available for headless execution.")

    selected_window = gui_window or "1d"
    selected_model = (model_name or getattr(self.args, "model", "lstm")).strip().lower()
    self.args.model = selected_model

    self.logger.info(
        "[Headless] stock=%s window=%s model=%s",
        selected_stock,
        selected_window,
        selected_model,
    )

    # ====================================
    # STEP 2: Load raw data synchronously
    engineered = ["return", "volatility", "momentum"]
    if self.args.ablate_feature != "none":
        engineered = [f for f in engineered if f != self.args.ablate_feature]

    self.raw_feature_cols = ["close"] + engineered

    self.logger.info(
        "[Ablation] ablate_feature=%s | node_raw_feature_cols=%s",
        self.args.ablate_feature,
        self.raw_feature_cols,
    )

    self.priceHistory, load_result = self._load_price_history(
        self.args.tickers,
        return_handler=True
    )

    valid_tickers = load_result.listTickers()
    dropped_tickers = [t for t in self.args.tickers if t not in valid_tickers]
    if dropped_tickers:
        self.logger.warning(
            "Dropped tickers (no valid data): %s",
            ", ".join(dropped_tickers)
        )

    self.args.tickers = valid_tickers
    self.raw_feature_dfs = {
        t: self.priceHistory[t] for t in valid_tickers
    }

    if not self.raw_feature_dfs:
        raise RuntimeError("No usable asset data was loaded.")

    if selected_stock not in self.raw_feature_dfs:
        raise ValueError(
            f"Requested stock '{selected_stock}' not available after data load. "
            f"Available: {', '.join(self.raw_feature_dfs.keys())}"
        )

    # ====================================
    # STEP 3: Bind front-end compatibility state without mainloop
    self.frontendApp.bindMainApp(self)
    self.frontendApp.modelVar.set(selected_model.upper())
    self.frontendApp.stockVar.set(selected_stock)
    self.frontendApp.windowVar.set(selected_window)

    evaluator = self.frontendApp.evaluator
    evaluator.reset_histories()

    # ====================================
    # STEP 4: Build pipeline state
    pipeline = Pipeline(self)
    state = pipeline.run(selected_stock, selected_window, stop_event=None)

    # ====================================
    # STEP 5: Execute shared experiment runner
    experiment_runner = ExperimentRunner(self)
    result = experiment_runner.run(
        model_name=selected_model,
        stock=selected_stock,
        state=state,
        evaluator=evaluator,
        stop_event=None,
    )

    self.frontendApp.updateResults(
        result.model_name,
        result.direction,
        result.confidence,
    )
    self.frontendApp.root.update_idletasks()

    return result