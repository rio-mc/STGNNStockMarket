import gc
import logging
import os
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dataclasses import asdict, is_dataclass

from config.config_manager import ConfigManager
from core.experiment_runner import ExperimentRunner
from core.headless_app import HeadlessEvaluator
from core.experiment_store import ExperimentStore, RunRecord
from core.job_queue import JobQueueController, QueueJob
from core.pipeline import Pipeline
from core.utils.utils import Utils
from data.price_loader import PriceLoaderRegistry
from data.tensor_factory import TensorFactory
from data.universe_service import UniverseService
from data.yahoo_price_loader import YahooPriceLoader
from ui.front_end import FrontEnd
from ui.loading_overlay import LoadingOverlay

from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    graph-building, model execution, queue execution, and result storage.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = ConfigManager.parseArgs()

        self.args.base_seed = getattr(self.args, "base_seed", 42)
        self.args.seed = getattr(self.args, "seed", self.args.base_seed)
        self.args.deterministic = getattr(self.args, "deterministic", False)

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

        self.logger = logging.getLogger("MainApp")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.universe_service = UniverseService()
        self.price_loader_registry = PriceLoaderRegistry()
        self.price_loader_registry.register("yahoo", YahooPriceLoader())

        self._set_all_seeds(self.args.seed)
        self.results_log = []

        self.queue_controller = JobQueueController()
        self.queue_running = False
        self.queue_stop_event = threading.Event()
        self._active_queue_job_id = None
        self._active_queue_run_id = None
        self._active_queue_run_started_at = None
        self._active_queue_manifest_jobs = []
        self._active_queue_job_summaries = []
        self._active_queue_completed = 0
        self._active_queue_failed = 0
        self._active_queue_cancelled = 0

        self.experiment_store = ExperimentStore(
            root_dir=getattr(self.args, "results_dir", "./results")
        )

        self._resolve_universe()

        self.args.tickers = sorted(self.args.tickers)

        run_mode = str(getattr(self.args, "run_mode", "gui")).strip().lower()
        self.frontendApp = FrontEnd(self.args.tickers, project_root=PROJECT_ROOT)
        self.frontendApp.setQueueAddCallback(self.enqueue_jobs)
        self.frontendApp.setQueueRunCallback(self.run_queue)
        self.frontendApp.setQueueRemoveCallback(self.remove_job_at)
        self.frontendApp.setQueueClearCallback(self.clear_queue)

        if run_mode == "headless":
            self.frontendApp.root.withdraw()
            self.loader = None
            self.logger.info("[Init] FrontEnd created in hidden mode for headless execution")
        else:
            from pathlib import Path

            project_root = Path(__file__).resolve().parent.parent
            avi_path = project_root / "assets" / "loading.avi"

            self.loader = LoadingOverlay(self.frontendApp.root, str(avi_path), delay=24)
            if not avi_path.exists():
                raise FileNotFoundError(f"Loading animation not found at: {avi_path}")

    def _resolve_universe_id_from_args(self) -> str:
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

        snapshot = self.queue_controller.snapshot()
        if not snapshot:
            self.frontendApp.set_status("Queue is empty. Add jobs before running.")
            return

        self.queue_running = True
        self.queue_stop_event.clear()

        # Queue lifecycle:
        # - a non-empty queue starts a new queue run group
        # - every popped job belongs to this queue_run_id
        # - when the queue drains again, the manifest is finalised once
        self._active_queue_run_id = self.experiment_store.make_queue_run_id()
        self._active_queue_run_started_at = self.experiment_store.utc_now_iso()
        self._active_queue_manifest_jobs = list(snapshot)
        self._active_queue_job_summaries = []
        self._active_queue_completed = 0
        self._active_queue_failed = 0
        self._active_queue_cancelled = 0

        self.experiment_store.write_queue_manifest(
            queue_run_id=self._active_queue_run_id,
            status="running",
            timestamp_start=self._active_queue_run_started_at,
            jobs=self._active_queue_manifest_jobs,
            extras={
                "universe_id": str(getattr(self.args, "universe_id", "unknown")),
                "interval": str(getattr(self.args, "interval", "unknown")),
            },
        )

        self.frontendApp.set_status(f"Started queue run {self._active_queue_run_id}")

        def _worker():
            queue_status = "completed"
            try:
                while not self.queue_stop_event.is_set():
                    job = self.queue_controller.pop_next()
                    self.frontendApp.refresh_queue_table(self.queue_controller.snapshot())

                    if job is None:
                        break

                    self.frontendApp.set_status(
                        f"Running queued job {job.model.upper()} {job.ticker} {job.prediction_window}"
                    )

                    try:
                        self._run_single_job(job)
                        self._active_queue_completed += 1
                    except InterruptedError:
                        self._active_queue_cancelled += 1
                        queue_status = "cancelled"
                        raise
                    except Exception:
                        self._active_queue_failed += 1
                        queue_status = "partial_failed"
                        self.logger.exception("Queued job failed: %s", job.job_id)
                        # Continue queue execution so one failed job does not
                        # discard the rest of the batch.
                        continue

                if self.queue_stop_event.is_set():
                    queue_status = "cancelled"

                if self._active_queue_failed and queue_status == "completed":
                    queue_status = "partial_failed"

                self.frontendApp.set_status(f"Queue finished: {queue_status}")

            except Exception as exc:
                self.logger.exception("Queue failed")
                self.frontendApp.set_status(f"Queue failed: {exc}")

            finally:
                queue_run_id = self._active_queue_run_id
                started_at = self._active_queue_run_started_at

                if queue_run_id and started_at:
                    summary_paths = self.experiment_store.write_queue_seed_summaries(
                        queue_run_id=queue_run_id,
                        job_summaries=self._active_queue_job_summaries,
                    )

                    self.experiment_store.write_queue_manifest(
                        queue_run_id=queue_run_id,
                        status=queue_status,
                        timestamp_start=started_at,
                        timestamp_end=self.experiment_store.utc_now_iso(),
                        jobs=self._active_queue_manifest_jobs,
                        completed=self._active_queue_completed,
                        failed=self._active_queue_failed,
                        cancelled=self._active_queue_cancelled,
                        extras={
                            "remaining_queue_count": len(self.queue_controller),
                            "universe_id": str(getattr(self.args, "universe_id", "unknown")),
                            "interval": str(getattr(self.args, "interval", "unknown")),
                            "summary_paths": summary_paths,
                        },
                    )

                self.queue_running = False
                self._active_queue_run_id = None
                self._active_queue_run_started_at = None
                self._active_queue_manifest_jobs = []
                self._active_queue_job_summaries = []

        threading.Thread(target=_worker, daemon=True).start()

    def _run_single_job(self, job: QueueJob):
        prev_model = self.args.model
        prev_seed = self.args.seed
        prev_graph_model = getattr(self.args, "graph_model", "gcn")
        prev_job_id = self._active_queue_job_id
        prev_queue_run_id = self._active_queue_run_id

        try:
            self.args.model = str(job.model).strip().lower()
            self.args.seed = int(job.seed)
            self.args.graph_model = str(job.graph_model).strip().lower()
            self._active_queue_job_id = job.job_id

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
            self._active_queue_job_id = prev_job_id
            self._active_queue_run_id = prev_queue_run_id

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

            self.raw_feature_dfs = {t: self.priceHistory[t] for t in valid_tickers}

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

    def _serialise_metrics(self, metrics_obj):
        if metrics_obj is None:
            return {}

        if isinstance(metrics_obj, dict):
            return dict(metrics_obj)

        if hasattr(metrics_obj, "__dict__"):
            return dict(metrics_obj.__dict__)

        return {"value": str(metrics_obj)}

    def _store_run_result(
        self,
        *,
        stock: str,
        gui_window: str,
        selected_model: str,
        result,
        status: str,
        ts_start: str,
        job_id=None,
        queue_run_id=None,
        error_message=None,
    ) -> None:
        run_id = self.experiment_store.make_run_id()
        ts_end = self.experiment_store.utc_now_iso()

        start_dt = datetime.fromisoformat(ts_start)
        end_dt = datetime.fromisoformat(ts_end)
        duration_sec = (end_dt - start_dt).total_seconds()

        history_path = None
        eval_result = getattr(result, "eval_result", None) if result is not None else None

        # For queue research suites, keep outputs seed-level rather than
        # per-iteration/per-date loss histories. Single manual runs still save
        # histories as before.
        if eval_result is not None and not queue_run_id:
            history_path = self.experiment_store.save_history(
                run_id=run_id,
                hist_train=getattr(eval_result, "hist_train", None),
                hist_val=getattr(eval_result, "hist_val", None),
            )

        metrics_payload = self._serialise_metrics(
            getattr(result, "metrics", None) if result is not None else None
        )

        queue_group = None
        job_payload_path = None
        if queue_run_id and job_id:
            queue_group = self.experiment_store.model_group(
                selected_model,
                str(getattr(self.args, "graph_model", "gcn")).lower(),
            )
            job_payload_path = self.experiment_store.save_job_payload(
                queue_run_id=queue_run_id,
                job_id=job_id,
                model=selected_model,
                graph_model=str(getattr(self.args, "graph_model", "gcn")).lower(),
                filename="result.json",
                payload={
                    "run_id": run_id,
                    "job_id": job_id,
                    "queue_run_id": queue_run_id,
                    "status": status,
                    "ticker": str(stock).upper(),
                    "prediction_window": str(gui_window),
                    "model": str(selected_model).lower(),
                    "seed": int(self.args.seed),
                    "graph_model": str(getattr(self.args, "graph_model", "gcn")).lower(),
                    "direction": str(getattr(result, "direction", "")) if result is not None else "",
                    "confidence": float(getattr(result, "confidence", 0.0)) if result is not None else 0.0,
                    "metrics": metrics_payload,
                    "error_message": error_message,
                },
            )

        if queue_run_id and job_id:
            self._active_queue_job_summaries.append(
                {
                    "run_id": run_id,
                    "job_id": job_id,
                    "queue_run_id": queue_run_id,
                    "queue_group": queue_group,
                    "status": status,
                    "ticker": str(stock).upper(),
                    "prediction_window": str(gui_window),
                    "model": str(selected_model).lower(),
                    "seed": int(self.args.seed),
                    "graph_model": str(getattr(self.args, "graph_model", "gcn")).lower(),
                    "direction": str(getattr(result, "direction", "")) if result is not None else "",
                    "confidence": float(getattr(result, "confidence", 0.0)) if result is not None else 0.0,
                    "metrics": metrics_payload,
                    "error_message": error_message,
                }
            )

        record = RunRecord(
            run_id=run_id,
            job_id=job_id,
            queue_run_id=queue_run_id,
            queue_group=queue_group,
            status=status,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            duration_sec=duration_sec,
            ticker=str(stock).upper(),
            prediction_window=str(gui_window),
            model=str(selected_model).lower(),
            seed=int(self.args.seed),
            graph_model=str(getattr(self.args, "graph_model", "gcn")).lower(),
            universe_id=str(getattr(self.args, "universe_id", "unknown")),
            interval=str(getattr(self.args, "interval", "unknown")),
            k=int(getattr(self.args, "k", 0)),
            graph_mode=str(getattr(self.args, "graph_mode", "unknown")),
            graph_embed=str(getattr(self.args, "graph_embed", "unknown")),
            ablate_feature=str(getattr(self.args, "ablate_feature", "none")),
            threshold_policy=str(getattr(self.args, "decision_threshold_policy", "fixed")),
            direction=str(getattr(result, "direction", "")) if result is not None else "",
            confidence=float(getattr(result, "confidence", 0.0)) if result is not None else 0.0,
            metrics=metrics_payload,
            extras={
                "history_path": history_path,
                "job_payload_path": job_payload_path,
                "queue_run": queue_run_id is not None,
                "queue_run_id": queue_run_id,
                "queue_group": queue_group,
                "error_message": error_message,
            },
        )
        self.experiment_store.append_run(record)

    def startPipeline(self, gui_window: str, stock: str, stop_event: threading.Event):
        if self.pipeline_running:
            self.logger.warning("Pipeline already running.")
            return ("-", 0.0, "-", 0.0, "-", 0.0)

        self.pipeline_running = True
        ts_start = self.experiment_store.utc_now_iso()
        selected_model = self.frontendApp.get_selected_model()
        result = None

        try:
            self.args.model = selected_model

            self.frontendApp.root.after(
                0,
                lambda: self.frontendApp.set_active_model_titles(selected_model)
            )

            self.logger.info("[ModelSelection] %s", selected_model)

            use_headless_evaluator = (
                self._active_queue_job_id is not None
                or threading.current_thread() is not threading.main_thread()
            )
            if use_headless_evaluator:
                evaluator = HeadlessEvaluator()
            else:
                evaluator = self.frontendApp.evaluator
                evaluator.reset_histories()
                self.frontendApp.root.after(0, self.frontendApp._reset_ui)

            self.frontendApp.set_status(f"Starting {selected_model.upper()}...")

            pipeline = Pipeline(self)
            state = pipeline.run(stock, gui_window, stop_event)

            experiment_runner = ExperimentRunner(self)
            result = experiment_runner.run(
                model_name=selected_model,
                stock=stock,
                state=state,
                evaluator=evaluator,
                stop_event=stop_event,
            )

            self._store_run_result(
                stock=stock,
                gui_window=gui_window,
                selected_model=selected_model,
                result=result,
                status="success",
                ts_start=ts_start,
                job_id=self._active_queue_job_id,
                queue_run_id=self._active_queue_run_id,
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
            self._store_run_result(
                stock=stock,
                gui_window=gui_window,
                selected_model=selected_model,
                result=result,
                status="cancelled",
                ts_start=ts_start,
                job_id=self._active_queue_job_id,
                queue_run_id=self._active_queue_run_id,
                error_message="Pipeline interrupted",
            )
            self.frontendApp.root.after(0, self.frontendApp._reset_ui)
            return (self.frontendApp.modelVar.get(), "-", 0.0)

        except Exception as exc:
            self.logger.exception("Pipeline failed")
            self._store_run_result(
                stock=stock,
                gui_window=gui_window,
                selected_model=selected_model,
                result=result,
                status="failed",
                ts_start=ts_start,
                job_id=self._active_queue_job_id,
                queue_run_id=self._active_queue_run_id,
                error_message=str(exc),
            )
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

                metrics_payload = (
                    asdict(result.metrics)
                    if is_dataclass(result.metrics)
                    else dict(result.metrics)
                )

                for key, value in metrics_payload.items():
                    print(f"  {key}: {value}")

            return result

        raise ValueError(
            f"Unknown run_mode '{self.args.run_mode}'. "
            f"Expected one of: gui, headless."
        )

    def run_headless(self, stock: str = None, gui_window: str = None, model_name: str = None):
        try:
            self.frontendApp.root.withdraw()
            self.frontendApp.root.update_idletasks()
        except Exception:
            pass

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

        self.raw_feature_dfs = {t: self.priceHistory[t] for t in valid_tickers}

        if not self.raw_feature_dfs:
            raise RuntimeError("No usable asset data was loaded.")

        if selected_stock not in self.raw_feature_dfs:
            raise ValueError(
                f"Requested stock '{selected_stock}' not available after data load. "
                f"Available: {', '.join(sorted(self.raw_feature_dfs.keys()))}"
            )

        self.frontendApp.bindMainApp(self)
        self.frontendApp.modelVar.set(selected_model.upper())
        self.frontendApp.stockVar.set(selected_stock)
        self.frontendApp.windowVar.set(selected_window)
        self.frontendApp.set_active_model_titles(selected_model)

        evaluator = self.frontendApp.evaluator
        evaluator.reset_histories()

        pipeline = Pipeline(self)
        state = pipeline.run(selected_stock, selected_window, stop_event=None)

        experiment_runner = ExperimentRunner(self)
        result = experiment_runner.run(
            model_name=selected_model,
            stock=selected_stock,
            state=state,
            evaluator=evaluator,
            stop_event=None,
        )

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

    def reload_universe(self, universe_id: str, custom_tickers: list = None):
        """Reload universe and prices - called from frontend universe selector."""
        self.args.universe_id = universe_id
        if custom_tickers:
            self.args.custom_tickers = custom_tickers
        
        self._resolve_universe()
        
        # Load prices for new universe
        self.priceHistory, load_result = self._load_price_history(
            self.args.tickers,
            return_handler=True,
        )
        
        valid_tickers = load_result.listTickers()
        dropped = [t for t in self.args.tickers if t not in valid_tickers]
        if dropped:
            self.logger.warning("Dropped tickers: %s", ", ".join(dropped))
        
        self.args.tickers = valid_tickers
        self.raw_feature_dfs = {t: self.priceHistory[t] for t in valid_tickers}
        
        return valid_tickers

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

        # DataLoader generator expected by model runners.
        # Keeps GUI queue runs deterministic and compatible with headless adapter runs.
        self.dl_gen = torch.Generator()
        self.dl_gen.manual_seed(self.current_seed)

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