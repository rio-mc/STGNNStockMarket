import gc
import logging
import os
import platform
import threading

import numpy as np
import pandas as pd
import torch

from core.pipeline import Pipeline
from front_end import FrontEnd
from data.tensor_factory import TensorFactory
from loading_overlay import LoadingOverlay
from utils import Utils
from config_manager import ConfigManager
from data.universe_service import UniverseService
from data.yahoo_price_loader import YahooPriceLoader
from core.experiment_runner import ExperimentRunner

# cuBLAS determinism
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


class LoadedPriceResult:
    """
    Lightweight compatibility wrapper so the rest of the app
    can still call listTickers() like it did with RawDataHandler.
    """

    def __init__(self, loaded_tickers):
        self._loaded_tickers = list(loaded_tickers)

    def listTickers(self):
        return self._loaded_tickers


class MainApp:
    """
    Orchestrates universe resolution, data loading, preprocessing,
    graph-building, and model execution.
    """

    def __init__(self):
        # ====================================
        # === STEP 1: Configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = ConfigManager.parseArgs()

        self.args.base_seed = getattr(self.args, "base_seed", 42)
        self.args.seed = getattr(self.args, "seed", self.args.base_seed)
        self.args.deterministic = getattr(self.args, "deterministic", False)

        self.graph_homophily = float("nan")
        self.pipeline_running = False
        self.data_ready = threading.Event()
        self.universe_info = None

        # ====================================
        # === STEP 2: Logging
        self.logger = logging.getLogger("MainApp")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self._set_all_seeds(self.args.seed)
        self.results_log = []

        # ====================================
        # === STEP 3: Resolve universe
        self._resolve_universe()

        # ====================================
        # === STEP 4: Front-end / compatibility layer
        self.args.tickers = sorted(self.args.tickers)

        run_mode = str(getattr(self.args, "run_mode", "gui")).strip().lower()
        self.frontendApp = FrontEnd(self.args.tickers)

        if run_mode == "headless":
            # Transitional compatibility:
            # keep FrontEnd/evaluator available for existing runners,
            # but do not show a visible Tk window.
            self.frontendApp.root.withdraw()
            self.loader = None
            self.logger.info("[Init] FrontEnd created in hidden mode for headless execution")
        else:
            avi_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "loading.avi"
            )
            self.loader = LoadingOverlay(self.frontendApp.root, avi_path, delay=24)

    def _resolve_universe(self):
        """
        Resolve the requested dataset / universe into a concrete ticker list.
        """
        universe_service = UniverseService()
        self.universe_info = universe_service.resolve(
            universe_name=self.args.dataset_name,
            top_n=self.args.top_n,
            as_of_date=self.args.universe_as_of,
            custom_tickers=self.args.custom_tickers,
        )

        self.args.tickers = self.universe_info["tickers"]

        self.logger.info(
            "[Universe] dataset=%s method=%s selection_date=%s count=%d",
            self.universe_info["universe_name"],
            self.universe_info["selection_method"],
            self.universe_info["selection_date"],
            self.universe_info["actual_count"],
        )

    def _load_data_and_start_gui(self):
        # ====================================
        # === STEP 1: Establish feature columns
        engineered = ["return", "volatility", "momentum"]

        if self.args.ablate_feature != "none":
            engineered = [f for f in engineered if f != self.args.ablate_feature]

        self.raw_feature_cols = ["close"] + engineered

        self.logger.info(
            f"[Ablation] ablate_feature={self.args.ablate_feature} | "
            f"node_raw_feature_cols={self.raw_feature_cols}"
        )

        def background_task():
            # ====================================
            # === STEP 2: Load raw price history
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

            # ====================================
            # === STEP 3: Build feature DF map
            self.raw_feature_dfs = {
                t: self.priceHistory[t] for t in valid_tickers
            }

            if not self.raw_feature_dfs:
                raise RuntimeError("No usable asset data was loaded.")

            # ====================================
            # === STEP 4: Bind front-end
            self.frontendApp.bindMainApp(self)
            self.loader.trigger_fade_and_destroy()

            self.frontendApp.bindTickerClick(self.frontendApp.on_ticker_click)
            self.frontendApp.setComputeCallback(self.startPipeline)

            self.data_ready.set()

        threading.Thread(target=background_task, daemon=True).start()

    def startPipeline(self, gui_window: str, stock: str, stop_event: threading.Event):
        if self.pipeline_running:
            self.logger.warning("Pipeline already running.")
            return ("-", 0.0, "-", 0.0, "-", 0.0)

        self.pipeline_running = True

        try:
            # ====================================
            # STEP 1: Resolve model
            selected_model = self.frontendApp.get_selected_model()
            self.args.model = selected_model
            self.logger.info("[ModelSelection] %s", selected_model)

            # ====================================
            # STEP 2: Reset evaluation/UI state
            evaluator = self.frontendApp.evaluator
            evaluator.reset_histories()

            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.frontendApp.set_status(f"Starting {selected_model.upper()}...")

            # ====================================
            # STEP 3: Build pipeline state
            pipeline = Pipeline(self)
            state = pipeline.run(stock, gui_window, stop_event)

            # ====================================
            # STEP 4: Run experiment through shared runner
            experiment_runner = ExperimentRunner(self)
            result = experiment_runner.run(
                model_name=selected_model,
                stock=stock,
                state=state,
                evaluator=evaluator,
                stop_event=stop_event,
            )

            self.frontendApp.root.after(
                0,
                lambda: self.frontendApp.updateResults(
                    result.model_name,
                    result.direction,
                    result.confidence,
                )
            )

            return (
                result.model_name,
                result.direction,
                result.confidence,
            )

        except InterruptedError:
            self.logger.info("[startPipeline] Pipeline interrupted by stop_event")
            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            return (self.frontendApp.modelVar.get(), "-", 0.0)

        except Exception:
            self.logger.exception("Pipeline failed")
            raise

        finally:
            self.pipeline_running = False
            self.frontendApp.btnCompute.config(state="normal", text="Compute ▶")
            self.frontendApp.btnStop.config(state="disabled")
            self.frontendApp.clear_status()

    def run(self):
        """
        Entry point for application execution.

        Modes:
        - gui: start background data loading and enter the Tk mainloop
        - headless: run a single CLI experiment without entering the GUI loop
        """
        run_mode = str(getattr(self.args, "run_mode", "gui")).strip().lower()

        if run_mode == "gui":
            self.logger.info("[RunMode] Starting GUI mode")
            threading.Thread(target=self._load_data_and_start_gui, daemon=True).start()
            self.frontendApp.root.mainloop()
            return None

        if run_mode == "headless":
            self.logger.info("[RunMode] Starting headless mode")

            selected_stock = getattr(self.args, "target_stock", None)
            selected_window = getattr(self.args, "prediction_window", "1d")
            selected_model = getattr(self.args, "model", "lstm")

            result = self.run_headless(
                stock=selected_stock,
                gui_window=selected_window,
                model_name=selected_model,
            )

            self.logger.info(
                "[HeadlessResult] model=%s direction=%s confidence=%.2f",
                result.model_name,
                result.direction,
                result.confidence,
            )

            print("\n=== Experiment Result ===")
            print(f"Model      : {result.model_name}")
            print(f"Direction  : {result.direction}")
            print(f"Confidence : {result.confidence:.2f}%")

            if getattr(result, "metrics", None):
                print("Metrics:")
                for key, value in result.metrics.items():
                    print(f"  {key}: {value}")

            return result

        raise ValueError(
            f"Unknown run_mode '{self.args.run_mode}'. "
            f"Expected one of: gui, headless."
        )
    

    def run_headless(self, stock: str = None, gui_window: str = None, model_name: str = None):
        """
        Run a single experiment without entering the Tk mainloop.

        Transitional headless mode:
        - reuses the existing FrontEnd/evaluator objects for compatibility
        - keeps the Tk root hidden
        - does not start the GUI event loop
        """

        # ====================================
        # STEP 1: Ensure the compatibility UI stays hidden
        try:
            self.frontendApp.root.withdraw()
            self.frontendApp.root.update_idletasks()
        except Exception:
            pass

        # ====================================
        # STEP 2: Resolve defaults
        selected_stock = stock or (self.args.tickers[0] if self.args.tickers else None)
        if not selected_stock:
            raise RuntimeError("No stock available for headless execution.")

        selected_stock = str(selected_stock).strip().upper()
        selected_window = str(gui_window or getattr(self.args, "prediction_window", "1d")).strip()
        selected_model = str(model_name or getattr(self.args, "model", "lstm")).strip().lower()
        self.args.model = selected_model

        self.logger.info(
            "[Headless] stock=%s window=%s model=%s",
            selected_stock,
            selected_window,
            selected_model,
        )

        # ====================================
        # STEP 3: Establish feature columns
        engineered = ["return", "volatility", "momentum"]
        if self.args.ablate_feature != "none":
            engineered = [f for f in engineered if f != self.args.ablate_feature]

        self.raw_feature_cols = ["close"] + engineered

        self.logger.info(
            "[Ablation] ablate_feature=%s | node_raw_feature_cols=%s",
            self.args.ablate_feature,
            self.raw_feature_cols,
        )

        # ====================================
        # STEP 4: Load raw price history synchronously
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
                f"Available: {', '.join(sorted(self.raw_feature_dfs.keys()))}"
            )

        # ====================================
        # STEP 5: Bind compatibility state without starting mainloop
        self.frontendApp.bindMainApp(self)
        self.frontendApp.modelVar.set(selected_model.upper())
        self.frontendApp.stockVar.set(selected_stock)
        self.frontendApp.windowVar.set(selected_window)

        evaluator = self.frontendApp.evaluator
        evaluator.reset_histories()

        # ====================================
        # STEP 6: Build pipeline state
        pipeline = Pipeline(self)
        state = pipeline.run(selected_stock, selected_window, stop_event=None)

        # ====================================
        # STEP 7: Execute shared experiment runner
        experiment_runner = ExperimentRunner(self)
        result = experiment_runner.run(
            model_name=selected_model,
            stock=selected_stock,
            state=state,
            evaluator=evaluator,
            stop_event=None,
        )

        # ====================================
        # STEP 8: Flush pending UI work while remaining hidden
        try:
            self.frontendApp.updateResults(
                result.model_name,
                result.direction,
                result.confidence,
            )
            self.frontendApp.root.update_idletasks()
        except Exception as exc:
            self.logger.warning("[Headless] UI compatibility update failed: %s", exc)

        return result
    
    def _load_price_history(self, tickers, period="729d", return_handler=False):
        """
        Load OHLCV data using the new Yahoo-only price loader.
        """
        loader = YahooPriceLoader()

        data = loader.load_prices(
            tickers=tickers,
            start_date=self.args.date_start,
            end_date=self.args.date_end,
            interval=self.args.interval,
            period=period if self.args.date_start is None and self.args.date_end is None else None,
        )

        wrapper = LoadedPriceResult(loaded_tickers=list(data.keys()))
        return (data, wrapper) if return_handler else data

    def _check_stop(self, stop_event: threading.Event):
        if stop_event and stop_event.is_set():
            self.logger.info("[MainApp] Pipeline stopped via stop_event.")
            raise InterruptedError("Pipeline interrupted")

    def get_max_k(self):
        return self.args.k

    def build_edge_index(self, coords, edges, *_):
        if edges:
            rows, cols = zip(*[(i, j) for i, j, _ in edges])
            return torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        return torch.zeros((2, 0), dtype=torch.long, device=self.device)

    def build_tensor_factory(self, horizon, seq_len=None):
        engineered = ["return", "volatility", "momentum"]
        if getattr(self.args, "ablate_feature", "none") in engineered:
            engineered = [f for f in engineered if f != self.args.ablate_feature]

        feature_cols = ["close"] + engineered

        resolved_seq_len = int(seq_len) if seq_len is not None else int(self.args.seq_len)

        return TensorFactory(
            tickers=self.args.tickers,
            featureCols=feature_cols,
            seq_len=resolved_seq_len,
            prediction_horizon=horizon,
            device="cpu",
        )

    def _set_all_seeds(self, run_seed: int | None = None):
        seed = int(run_seed) if run_seed is not None else int(self.args.base_seed)
        used = Utils.set_seed(seed, deterministic=self.args.deterministic)
        self.current_seed = int(used)
        self.logger.info(
            f"[Seed] Using seed={self.current_seed} "
            f"(deterministic={self.args.deterministic})"
        )

    def _seed_worker(self, worker_id: int):
        worker_seed = (self.current_seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)

        import random as _random
        _random.seed(worker_seed)

    def run_experiments(self):
        self.data_ready.wait()
        self.logger.info(
            "run_experiments() not yet refactored into the new model-runner flow."
        )


if __name__ == "__main__":
    app = MainApp()
    app.run()