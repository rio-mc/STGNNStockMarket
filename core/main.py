import gc
import logging
import os
import platform
import threading

import numpy as np
import pandas as pd
import torch

from core.job_queue import JobQueueController, QueueJob
from core.pipeline import Pipeline
from ui.front_end import FrontEnd
from data.tensor_factory import TensorFactory
from ui.loading_overlay import LoadingOverlay
from core.utils.utils import Utils
from config.config_manager import ConfigManager
from data.price_loader import PriceLoaderRegistry
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

        # Canonical selection / provider fields with backward compatibility
        self.args.universe_id = self._resolve_universe_id_from_args()
        self.args.universe_provider = str(
            getattr(self.args, "universe_provider", "static_csv")
        ).strip().lower()
        self.args.price_provider = str(
            getattr(self.args, "price_provider", "yahoo")
        ).strip().lower()

        self.graph_homophily = float("nan")
        self.pipeline_running = False
        self.data_ready = threading.Event()

        self.universe_definition = None
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

        # ====================================
        # === STEP 3: Services
        self.universe_service = UniverseService()
        self.price_loader_registry = PriceLoaderRegistry()
        self.price_loader_registry.register("yahoo", YahooPriceLoader())

        self._set_all_seeds(self.args.seed)
        self.results_log = []
        self.queue_controller = JobQueueController()
        self.queue_running = False
        self.queue_stop_event = threading.Event()

        # ====================================
        # === STEP 4: Resolve universe
        self._resolve_universe()

        # ====================================
        # === STEP 5: Front-end / compatibility layer
        self.args.tickers = sorted(self.args.tickers)

        run_mode = str(getattr(self.args, "run_mode", "gui")).strip().lower()
        self.frontendApp = FrontEnd(self.args.tickers)
        self.frontendApp.setQueueAddCallback(self.enqueue_jobs)
        self.frontendApp.setQueueRunCallback(self.run_queue)
        self.frontendApp.setQueueRemoveCallback(self.remove_job_at)
        self.frontendApp.setQueueClearCallback(self.clear_queue)

        if run_mode == "headless":
            # Transitional compatibility:
            # keep FrontEnd/evaluator available for existing runners,
            # but do not show a visible Tk window.
            self.frontendApp.root.withdraw()
            self.loader = None
            self.logger.info("[Init] FrontEnd created in hidden mode for headless execution")
        else:
            from pathlib import Path

            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            avi_path = PROJECT_ROOT / "assets" / "loading.avi"

            self.loader = LoadingOverlay(self.frontendApp.root, str(avi_path), delay=24)
            if not avi_path.exists():
                raise FileNotFoundError(f"Loading animation not found at: {avi_path}")

    def _resolve_universe_id_from_args(self) -> str:
        """
        Canonicalise historical argument names into a single concept: universe_id.
        """
        candidates = [
            getattr(self.args, "universe_id", None),
            getattr(self.args, "universe_name", None),
            getattr(self.args, "dataset_name", None),
            getattr(self.args, "dataset", None),
        ]

        raw = next((v for v in candidates if v is not None and str(v).strip()), "sp500")
        raw = str(raw).strip().lower()

        aliases = {
            "sp_500": "sp500",
            "s&p500": "sp500",
            "s_and_p_500": "sp500",
            "nasdaq_100": "nasdaq100",
            "ndx": "nasdaq100",
        }
        return aliases.get(raw, raw)

    def _resolve_universe(self):
        self.universe_definition = self.universe_service.resolve_definition(
            universe_id=self.args.universe_id,
            universe_provider=self.args.universe_provider,
            top_n=self.args.top_n,
            as_of_date=getattr(self.args, "universe_as_of", None),
            custom_tickers=getattr(self.args, "custom_tickers", None),
        )

        self.universe_info = self.universe_definition.to_dict()
        self.args.tickers = list(self.universe_definition.tickers)

        self.logger.info(
            "[Universe] universe_id=%s universe_provider=%s count=%d snapshot_date=%s",
            self.universe_definition.universe_id,
            self.universe_definition.metadata.universe_provider,
            len(self.universe_definition.tickers),
            self.universe_definition.metadata.snapshot_date,
        )


    def enqueue_jobs(self, jobs):
        jobs = list(jobs)
        if not jobs:
            return
        self.queue_controller.enqueue_many(jobs)
        self.frontendApp.refresh_queue_table(self.queue_controller.snapshot())
        self.frontendApp.set_status(f"Queued {len(jobs)} job(s)")

    def remove_job_at(self, index: int) -> None:
        removed = self.queue_controller.remove_at(index)
        self.frontendApp.refresh_queue_table(self.queue_controller.snapshot())
        if removed is not None:
            self.frontendApp.set_status(f"Removed {removed.job_id}")

    def clear_queue(self) -> None:
        self.queue_controller.clear()
        self.frontendApp.refresh_queue_table([])
        self.frontendApp.set_status("Queue cleared")

    def run_queue(self) -> None:
        if self.queue_running:
            self.frontendApp.set_status("Queue already running")
            return

        self.queue_running = True
        self.queue_stop_event.clear()

        def _worker():
            try:
                while not self.queue_stop_event.is_set():
                    job = self.queue_controller.pop_next()
                    self.frontendApp.refresh_queue_table(self.queue_controller.snapshot())

                    if job is None:
                        break

                    self.frontendApp.set_status(
                        f"Running queued job {job.model.upper()} {job.ticker} {job.prediction_window}"
                    )
                    self._run_single_job(job)

                self.frontendApp.set_status("Queue finished")
            except Exception as exc:
                self.logger.exception("Queue failed")
                self.frontendApp.set_status(f"Queue failed: {exc}")
            finally:
                self.queue_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _run_single_job(self, job: QueueJob):
        prev_model = self.args.model
        prev_seed = self.args.seed
        prev_graph_model = getattr(self.args, "graph_model", "gcn")

        try:
            self.args.model = str(job.model).strip().lower()
            self.args.seed = int(job.seed)
            self.args.graph_model = str(job.graph_model).strip().lower()

            self._set_all_seeds(self.args.seed)

            self.frontendApp.ui_call(self.frontendApp.modelVar.set, self.args.model.upper())
            self.frontendApp.ui_call(self.frontendApp.stockVar.set, job.ticker)
            self.frontendApp.ui_call(self.frontendApp.windowVar.set, job.prediction_window)

            if hasattr(self.frontendApp, "graphModelVar"):
                graph_value = self.args.graph_model.upper() if self.args.model == "stgnn" else "GCN"
                self.frontendApp.ui_call(self.frontendApp.graphModelVar.set, graph_value)

            if hasattr(self.frontendApp, "seedVar"):
                self.frontendApp.ui_call(self.frontendApp.seedVar.set, str(self.args.seed))

            self.startPipeline(job.prediction_window, job.ticker, self.queue_stop_event)
        finally:
            self.args.model = prev_model
            self.args.seed = prev_seed
            self.args.graph_model = prev_graph_model

    def _load_data_and_start_gui(self):
        engineered = ["return", "volatility", "momentum"]

        if self.args.ablate_feature != "none":
            engineered = [f for f in engineered if f != self.args.ablate_feature]

        self.raw_feature_cols = ["close"] + engineered

        self.logger.info(
            f"[Ablation] ablate_feature={self.args.ablate_feature} | "
            f"node_raw_feature_cols={self.raw_feature_cols}"
        )

        def background_task():
            self.priceHistory, load_result = self._load_price_history(
                self.args.tickers,
                return_handler=True,
            )

            valid_tickers = load_result.listTickers()
            dropped_tickers = [t for t in self.args.tickers if t not in valid_tickers]
            if dropped_tickers:
                self.logger.warning(
                    "Dropped tickers (no valid data from %s): %s",
                    self.args.price_provider,
                    ", ".join(dropped_tickers),
                )

            self.args.tickers = valid_tickers

            if self.universe_definition is not None:
                self.universe_definition = self.universe_definition.with_dropped_tickers(
                    dropped_tickers=dropped_tickers
                )
                self.universe_info = self.universe_definition.to_dict()

            self.raw_feature_dfs = {
                t: self.priceHistory[t] for t in valid_tickers
            }

            if not self.raw_feature_dfs:
                raise RuntimeError("No usable asset data was loaded.")

            def finish_gui_init():
                self.frontendApp.bindMainApp(self)
                self.frontendApp.setComputeCallback(self.startPipeline)
                if self.loader is not None:
                    self.loader.trigger_fade_and_destroy()
                self.data_ready.set()

            self.frontendApp.root.after(0, finish_gui_init)

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

            self.frontendApp.root.after(
                0,
                lambda: self.frontendApp.set_active_model_titles(selected_model)
            )

            self.logger.info("[ModelSelection] %s", selected_model)

            # ====================================
            # STEP 2: Reset evaluation/UI state
            evaluator = self.frontendApp.evaluator
            evaluator.reset_histories()

            self.frontendApp.root.after(0, self.frontendApp._reset_ui)
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
            self.frontendApp.root.after(0, self.frontendApp._reset_ui)
            return (self.frontendApp.modelVar.get(), "-", 0.0)

        except Exception:
            self.logger.exception("Pipeline failed")
            raise

        finally:
            self.pipeline_running = False

            def _finish_ui():
                self.frontendApp.btnCompute.config(state="normal", text="Compute ▶")
                self.frontendApp.btnStop.config(state="disabled")
                self.frontendApp.clear_status()

            try:
                self.frontendApp.root.after(0, _finish_ui)
            except RuntimeError:
                pass
            
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
                "Dropped tickers (no valid data from %s): %s",
                self.args.price_provider,
                ", ".join(dropped_tickers)
            )

        self.args.tickers = valid_tickers

        if self.universe_definition is not None:
            self.universe_definition = self.universe_definition.with_dropped_tickers(
                dropped_tickers=dropped_tickers
            )
            self.universe_info = self.universe_definition.to_dict()

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
        self.frontendApp.set_active_model_titles(selected_model)

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
        Load OHLCV data using the configured price provider.
        """
        loader = self.price_loader_registry.get(self.args.price_provider)

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

        return {
            "tickers": self.args.tickers,
            "feature_cols": feature_cols,
            "seq_len": resolved_seq_len,
            "prediction_horizon": horizon,
            "device": "cpu",
        }

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