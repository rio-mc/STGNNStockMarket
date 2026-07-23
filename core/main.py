import gc
import logging
import os
import signal
import sys
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

from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IS_CLI_ENTRYPOINT = __name__ == "__main__"


def _should_force_headless_cli_exit() -> bool:
    return _IS_CLI_ENTRYPOINT and os.name == "nt"


def _finish_successful_windows_headless_process(exit_code: int = 0) -> None:
    """Exit after explicit cleanup without running fragile native destructors.

    Some Windows CUDA/native-extension combinations terminate with 0xC0000409
    while Python finalises native objects, even though the headless experiment has
    already completed, persisted its result, and released its resources.  This
    helper is deliberately called only after ``MainApp.run()`` returns
    successfully, so training and persistence failures still propagate with a
    non-zero process status.
    """

    try:
        logging.shutdown()
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
    finally:
        os._exit(int(exit_code))


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
        # This logger owns its handler.  Do not send the same record to the
        # root handler configured by graph_builder as well.
        self.logger.propagate = False

        self.device = Utils.resolve_device(
            getattr(self.args, "device", "auto"),
            logger=self.logger,
        )
        self.logger.info("[Device] Using device=%s", self.device)

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
        self.frontendApp = None
        if run_mode == "headless":
            self.loader = None
            self.logger.info("[Init] GUI-free headless mode; FrontEnd not created")
        else:
            # Keep Tk and all frontend dependencies out of the headless import
            # and execution path.
            from ui.front_end import FrontEnd
            from ui.loading_overlay import LoadingOverlay

            self.frontendApp = FrontEnd(self.args.tickers, project_root=PROJECT_ROOT)
            self.frontendApp.setQueueAddCallback(self.enqueue_jobs)
            self.frontendApp.setQueueRunCallback(self.run_queue)
            self.frontendApp.setQueueRemoveCallback(self.remove_job_at)
            self.frontendApp.setQueueClearCallback(self.clear_queue)
            self.frontendApp.setCloseCallback(self.shutdown)

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
        if isinstance(index, (list, tuple, set)):
            removed_jobs = self.queue_controller.remove_indices(index)
            removed = removed_jobs[0] if removed_jobs else None
            removed_count = len(removed_jobs)
        else:
            removed = self.queue_controller.remove_at(index)
            removed_count = 1 if removed is not None else 0

        self.frontendApp.refresh_queue_table(self.queue_controller.snapshot())
        if removed is not None:
            if removed_count > 1:
                self.frontendApp.set_status(f"Removed {removed_count} queued job(s)")
            else:
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

    def shutdown(self) -> None:
        self.logger.info("[Shutdown] Stopping workers and releasing resources")
        self.queue_stop_event.set()
        self.pipeline_running = False

        try:
            if hasattr(self, "frontendApp") and self.frontendApp is not None:
                self.frontendApp.stop_event.set()
        except Exception:
            pass

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_headless_resources(self, result=None) -> None:
        """Deterministically release native GUI/cuDNN resources before exit.

        Letting Windows tear CUDA/native objects down during interpreter
        finalisation can produce 0xC0000409 after a recurrent experiment has
        already saved and printed its result.
        """

        self._release_headless_result(result)

        frontend = getattr(self, "frontendApp", None)
        if frontend is not None:
            try:
                trainers = getattr(frontend, "_trainers", None)
                if isinstance(trainers, dict):
                    trainers.clear()
                frontend._on_close()
            except Exception as exc:
                self.logger.debug("[HeadlessCleanup] Frontend cleanup failed: %s", exc)
                self.shutdown()
        else:
            self.shutdown()

        # On Windows, forcing collection here can enter cuDNN/native destructors
        # before the successful CLI path reaches os._exit(). References have
        # already been detached above; process termination reclaims them.
        if os.name == "nt":
            return

        gc.collect()

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:
            self.logger.debug("[HeadlessCleanup] CUDA cleanup failed: %s", exc)

    def _release_headless_result(self, result=None) -> None:
        """Release one completed model while keeping a headless worker alive."""

        trainer = getattr(result, "trainer", None) if result is not None else None
        model = getattr(result, "model", None) if result is not None else None
        optimiser = None

        try:
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception as exc:
            self.logger.debug("[HeadlessCleanup] Could not move model to CPU: %s", exc)

        try:
            if trainer is not None:
                optimiser = getattr(trainer, "optimiser", None)
                if optimiser is not None and hasattr(optimiser, "state"):
                    optimiser.state.clear()
                trainer.optimiser = None
                trainer.criterion = None
                trainer.edge_index = None
                trainer.model = None
        except Exception as exc:
            self.logger.debug("[HeadlessCleanup] Trainer cleanup failed: %s", exc)

        if result is not None:
            result.trainer = None
            result.model = None

        trainer = None
        model = None
        optimiser = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            self.logger.debug("[HeadlessCleanup] Result CUDA cleanup failed: %s", exc)

    def _request_gui_close(self, *_args) -> None:
        try:
            if hasattr(self, "frontendApp") and self.frontendApp is not None:
                self.frontendApp._on_close()
            else:
                self.shutdown()
        except Exception:
            self.shutdown()

    def _run_single_job(self, job: QueueJob):
        prev_model = self.args.model
        prev_seed = self.args.seed
        prev_graph_model = getattr(self.args, "graph_model", "gcn")
        prev_k = getattr(self.args, "k", 3)
        prev_graph_mode = getattr(self.args, "graph_mode", "knn_mst")
        prev_graph_embed = getattr(self.args, "graph_embed", "pca")
        prev_graph_ablation = getattr(self.args, "graph_ablation", "none")
        prev_ablate_feature = getattr(self.args, "ablate_feature", "none")
        prev_seq_len = getattr(self.args, "seq_len", 10)
        prev_batch_size = getattr(self.args, "batch_size", 256)
        prev_lstm_epochs = getattr(self.args, "lstm_epochs", 200)
        prev_stgnn_epochs = getattr(self.args, "stgnn_epochs", 200)
        prev_job_id = self._active_queue_job_id
        prev_queue_run_id = self._active_queue_run_id

        try:
            self.args.model = str(job.model).strip().lower()
            self.args.seed = int(job.seed)
            self.args.graph_model = str(job.graph_model).strip().lower()
            self.args.k = int(job.k)
            self.args.graph_mode = str(job.graph_mode).strip().lower()
            self.args.graph_embed = str(job.graph_embed).strip().lower()
            self.args.graph_ablation = str(job.graph_ablation).strip().lower()
            self.args.ablate_feature = str(job.ablate_feature).strip().lower()
            self.args.seq_len = int(job.seq_len)
            self.args.graph_window = int(job.seq_len)
            self.args.batch_size = int(job.batch_size)
            self.args.lstm_epochs = int(job.lstm_epochs)
            self.args.stgnn_epochs = int(job.stgnn_epochs)
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
            if hasattr(self.frontendApp, "kVar"):
                self.frontendApp.ui_call(self.frontendApp.kVar.set, str(self.args.k))
                self.frontendApp.ui_call(self.frontendApp.graphModeVar.set, self.args.graph_mode)
                self.frontendApp.ui_call(self.frontendApp.graphEmbedVar.set, self.args.graph_embed)
                self.frontendApp.ui_call(self.frontendApp.graphAblationVar.set, self.args.graph_ablation)
                self.frontendApp.ui_call(self.frontendApp.ablateFeatureVar.set, self.args.ablate_feature)
                self.frontendApp.ui_call(self.frontendApp.seqLenVar.set, str(self.args.seq_len))
                self.frontendApp.ui_call(self.frontendApp.batchSizeVar.set, str(self.args.batch_size))
                self.frontendApp.ui_call(self.frontendApp.lstmEpochsVar.set, str(self.args.lstm_epochs))
                self.frontendApp.ui_call(self.frontendApp.stgnnEpochsVar.set, str(self.args.stgnn_epochs))

            self.startPipeline(job.prediction_window, job.ticker, self.queue_stop_event)
        finally:
            self.args.model = prev_model
            self.args.seed = prev_seed
            self.args.graph_model = prev_graph_model
            self.args.k = prev_k
            self.args.graph_mode = prev_graph_mode
            self.args.graph_embed = prev_graph_embed
            self.args.graph_ablation = prev_graph_ablation
            self.args.ablate_feature = prev_ablate_feature
            self.args.seq_len = prev_seq_len
            self.args.graph_window = prev_seq_len
            self.args.batch_size = prev_batch_size
            self.args.lstm_epochs = prev_lstm_epochs
            self.args.stgnn_epochs = prev_stgnn_epochs
            self._active_queue_job_id = prev_job_id
            self._active_queue_run_id = prev_queue_run_id

    def _apply_frontend_experiment_controls(self) -> None:
        if not hasattr(self.frontendApp, "export_experiment_controls"):
            return

        controls = self.frontendApp.export_experiment_controls()
        if hasattr(self.frontendApp, "graphModelVar"):
            self.args.graph_model = str(self.frontendApp.graphModelVar.get()).strip().lower()
        self.args.k = int(controls["k"])
        self.args.graph_mode = str(controls["graph_mode"])
        self.args.graph_embed = str(controls["graph_embed"])
        self.args.graph_ablation = str(controls["graph_ablation"])
        self.args.ablate_feature = str(controls["ablate_feature"])
        self.args.seq_len = int(controls["seq_len"])
        self.args.graph_window = int(controls["seq_len"])
        self.args.batch_size = int(controls["batch_size"])
        self.args.lstm_epochs = int(controls["lstm_epochs"])
        self.args.stgnn_epochs = int(controls["stgnn_epochs"])

    def _refresh_raw_feature_cols_from_ablation(self) -> None:
        engineered = ["return", "volatility", "momentum"]
        if getattr(self.args, "ablate_feature", "none") != "none":
            engineered = [f for f in engineered if f != self.args.ablate_feature]
        self.raw_feature_cols = ["close"] + engineered

    def _load_data_and_start_gui(self):
        self._refresh_raw_feature_cols_from_ablation()

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


    def _serialise_metadata(self, eval_result):
        if eval_result is None:
            return {}
        return dict(getattr(eval_result, "metadata", {}) or {})

    def _extract_compute_payload(self, metadata_payload):
        keys = [
            "energy_wh",
            "train_seconds",
            "avg_power_w",
            "energy_per_sample_wh",
            "energy_measurement_method",
            "cpu_power_watts",
            "train_samples",
            "training_sample_unit",
            "train_examples_unique",
            "sample_exposures",
            "epochs_completed",
            "gpu_peak_memory_mb",
            "total_params",
            "trainable_params",
        ]
        return {key: metadata_payload.get(key) for key in keys}

    def _extract_capacity_payload(self, metadata_payload):
        capacity = metadata_payload.get("capacity", {}) or {}
        return dict(capacity) if isinstance(capacity, dict) else {}
    

    def _fmt_float(self, value, digits=3, missing="n/a"):
        try:
            if value is None:
                return missing
            value = float(value)
            if value != value:
                return missing
            if value in (float("inf"), float("-inf")):
                return missing
            return f"{value:.{digits}f}"
        except Exception:
            return missing

    def _fmt_int(self, value, missing="n/a"):
        try:
            if value is None:
                return missing
            return str(int(value))
        except Exception:
            return missing

    def _box_line(self, text="", width=96):
        text = str(text)
        usable = width - 4
        if len(text) > usable:
            text = text[: usable - 3] + "..."
        return "| " + text.ljust(usable) + " |"

    def _box_rule(self, kind="middle", width=96):
        if kind == "top":
            return "+" + ("-" * (width - 2)) + "+"
        if kind == "bottom":
            return "+" + ("-" * (width - 2)) + "+"
        return "+" + ("-" * (width - 2)) + "+"

    def _metrics_dict(self, metrics):
        if not metrics:
            return {}
        if is_dataclass(metrics):
            return asdict(metrics)
        if isinstance(metrics, dict):
            return dict(metrics)
        if hasattr(metrics, "__dict__"):
            return dict(metrics.__dict__)
        return {"value": str(metrics)}

    def _format_headless_report(self, result) -> str:
        metrics = self._metrics_dict(getattr(result, "metrics", None))
        metadata = self._serialise_metadata(getattr(result, "eval_result", None))
        compute = self._extract_compute_payload(metadata)
        capacity = self._extract_capacity_payload(metadata)

        model = str(getattr(result, "model_name", metrics.get("model", "MODEL"))).upper()
        ticker = str(metrics.get("ticker") or metadata.get("ticker") or "").upper()
        direction = str(getattr(result, "direction", "UNKNOWN")).upper()
        confidence = self._fmt_float(getattr(result, "confidence", None), 2)

        horizon = metrics.get("horizon", "n/a")
        seed = metadata.get("seed", getattr(self.args, "seed", "n/a"))
        universe_id = metadata.get("universe_id", getattr(self.args, "universe_id", "n/a"))
        interval = metadata.get("interval", getattr(self.args, "interval", "n/a"))

        graph_backend = metadata.get("graph_backend") or metrics.get("graph_backend")
        graph_model = metadata.get("graph_model") or metrics.get("graph_model")

        if graph_backend and model == "STGNN":
            model_label = f"{model}+{str(graph_backend).upper()}"
        else:
            model_label = model

        threshold_policy = (
            metadata.get("decision_threshold_policy")
            or metrics.get("threshold_selection_metric")
            or "n/a"
        )
        width = 96

        operational_threshold = metrics.get(
            "threshold_operational",
            metrics.get("threshold_fixed"),
        )
        validation_threshold = metrics.get("threshold_macro_f1_dense")
        try:
            show_validation_threshold_metrics = (
                abs(float(operational_threshold) - float(validation_threshold)) > 1e-12
            )
        except (TypeError, ValueError):
            show_validation_threshold_metrics = False

        validation_threshold_lines = []
        if show_validation_threshold_metrics:
            validation_threshold_lines = [
                self._box_line("", width),
                self._box_line("VALIDATION MACRO-F1 THRESHOLD ON TEST", width),
                self._box_line(
                    f"  acc {self._fmt_float(metrics.get('accuracy_dense_macro_f1_threshold'))} | "
                    f"f1_pos {self._fmt_float(metrics.get('f1_dense_macro_f1_threshold'))} | "
                    f"f1_macro {self._fmt_float(metrics.get('macro_f1_dense_macro_f1_threshold'))}",
                    width,
                ),
            ]

        lines = [
            self._box_rule("top", width),
            self._box_line("EXPERIMENT RESULT", width),
            self._box_rule("middle", width),
            self._box_line(
                f"RUN        {model_label} | {ticker} | horizon {horizon} bars | "
                f"seed {seed} | {universe_id} | {interval}",
                width,
            ),
            self._box_line(f"SIGNAL     {direction} {confidence}%", width),
            self._box_line(
                "THRESHOLD  "
                f"operational={self._fmt_float(metrics.get('threshold_operational', metrics.get('threshold_fixed')), 3)} | "
                f"validation_macro_f1={self._fmt_float(metrics.get('threshold_macro_f1_dense'), 3)} | "
                f"policy={threshold_policy}",
                width,
            ),
        ]

        if graph_backend or graph_model:
            lines.append(
                self._box_line(
                    f"GRAPH      backend={graph_backend or 'n/a'} | graph_model={graph_model or 'n/a'}",
                    width,
                )
            )

        lines.extend(
            [
                self._box_rule("middle", width),

                self._box_line("DENSE CLASSIFICATION", width),
                self._box_line(
                    f"  acc {self._fmt_float(metrics.get('accuracy_dense'))} | "
                    f"f1_pos {self._fmt_float(metrics.get('f1_dense'))} | "
                    f"f1_macro {self._fmt_float(metrics.get('macro_f1_dense'))} | "
                    f"roc_auc {self._fmt_float(metrics.get('roc_auc_dense'))} | "
                    f"ap {self._fmt_float(metrics.get('ap_dense'))}",
                    width,
                ),
                self._box_line(
                    f"  val_loss {self._fmt_float(metrics.get('val_loss_dense'))} | "
                    f"test_loss {self._fmt_float(metrics.get('test_loss_dense'))} | "
                    f"n {self._fmt_int(metrics.get('n_predictions_dense'))}",
                    width,
                ),

                self._box_line("", width),

                self._box_line("FIXED 0.5 ROBUSTNESS BASELINE", width),
                self._box_line(
                    f"  acc {self._fmt_float(metrics.get('accuracy_dense_fixed_05'))} | "
                    f"f1_pos {self._fmt_float(metrics.get('f1_dense_fixed_05'))} | "
                    f"f1_macro {self._fmt_float(metrics.get('macro_f1_dense_fixed_05'))}",
                    width,
                ),

                *validation_threshold_lines,

                self._box_line("", width),

                self._box_line("TRADE-ALIGNED", width),
                self._box_line(
                    f"  acc {self._fmt_float(metrics.get('accuracy_trade_aligned'))} | "
                    f"f1_pos {self._fmt_float(metrics.get('f1_trade_aligned'))} | "
                    f"f1_macro {self._fmt_float(metrics.get('macro_f1_trade_aligned'))} | "
                    f"roc_auc {self._fmt_float(metrics.get('roc_auc_trade_aligned'))} | "
                    f"ap {self._fmt_float(metrics.get('ap_trade_aligned'))}",
                    width,
                ),
                self._box_line(
                    f"  trades {self._fmt_int(metrics.get('n_trades'))} | "
                    f"hit_rate {self._fmt_float(metrics.get('hit_rate'))} | "
                    f"mean_ret {self._fmt_float(metrics.get('mean_trade_return'), 6)}",
                    width,
                ),

                self._box_line("", width),

                self._box_line("STRATEGY", width),
                self._box_line(
                    f"  sharpe {self._fmt_float(metrics.get('sharpe'))} | "
                    f"final_equity {self._fmt_float(metrics.get('final_equity'))} | "
                    f"max_drawdown {self._fmt_float(metrics.get('max_drawdown'))}",
                    width,
                ),

                self._box_line("", width),

                self._box_line("COMPUTE", width),
                self._box_line(
                    f"  train {self._fmt_float(compute.get('train_seconds'), 2)}s | "
                    f"energy {self._fmt_float(compute.get('energy_wh'), 4)} Wh | "
                    f"avg_power {self._fmt_float(compute.get('avg_power_w'), 2)} W | "
                    f"peak_gpu {self._fmt_float(compute.get('gpu_peak_memory_mb'), 2)} MB",
                    width,
                ),
                self._box_line(
                    f"  unique_examples {self._fmt_int(compute.get('train_examples_unique'))} | "
                    f"sample_exposures {self._fmt_int(compute.get('sample_exposures'))} | "
                    f"epochs {self._fmt_int(compute.get('epochs_completed'))} | "
                    f"unit {compute.get('training_sample_unit') or 'n/a'}",
                    width,
                ),
                self._box_line(
                    f"  energy_per_sample {self._fmt_float(compute.get('energy_per_sample_wh'), 8)} Wh | "
                    f"method {compute.get('energy_measurement_method') or 'n/a'}",
                    width,
                ),

                self._box_line("", width),

                self._box_line("MODEL CAPACITY", width),
                self._box_line(
                    f"  family {capacity.get('family') or 'n/a'} | "
                    f"measure {capacity.get('primary_measure') or 'n/a'} | "
                    f"value {self._fmt_int(capacity.get('primary_value'))} | "
                    f"parameter_storage {self._fmt_int(capacity.get('parameter_storage_bytes'))} bytes",
                    width,
                ),

                self._box_rule("bottom", width),
            ]
        )

        return "\n".join(lines)

    def _format_headless_result_line(self, result) -> str:
        """Compact human-readable companion to runs/run_results.csv."""

        metrics = self._metrics_dict(getattr(result, "metrics", None))
        metadata = self._serialise_metadata(getattr(result, "eval_result", None))
        compute = self._extract_compute_payload(metadata)
        model = str(getattr(result, "model_name", metrics.get("model", "unknown"))).lower()
        graph_backend = metadata.get("graph_backend") or metrics.get("graph_backend")
        model_label = f"{model}+{graph_backend}" if model == "stgnn" and graph_backend else model
        ticker = str(metrics.get("ticker") or metadata.get("ticker") or "").upper()
        seed = metadata.get("seed", getattr(self.args, "seed", "n/a"))
        threshold = metrics.get("threshold_operational", metrics.get("threshold_fixed"))

        return " | ".join(
            [
                "RESULT",
                f"model={model_label}",
                f"ticker={ticker}",
                f"seed={seed}",
                f"signal={str(getattr(result, 'direction', 'unknown')).lower()}",
                f"confidence_pct={self._fmt_float(getattr(result, 'confidence', None), 2)}",
                f"threshold={self._fmt_float(threshold, 3)}",
                f"macro_f1={self._fmt_float(metrics.get('macro_f1_dense'))}",
                f"macro_f1_fixed_05={self._fmt_float(metrics.get('macro_f1_dense_fixed_05'))}",
                f"macro_f1_trade={self._fmt_float(metrics.get('macro_f1_trade_aligned'))}",
                f"sharpe={self._fmt_float(metrics.get('sharpe'))}",
                f"final_equity={self._fmt_float(metrics.get('final_equity'))}",
                f"train_sec={self._fmt_float(compute.get('train_seconds'), 2)}",
            ]
        )


    def _is_graph_aware_model(self, selected_model: str, graph_backend=None) -> bool:
        model = str(selected_model or "").strip().lower()
        return model in {"gcn", "gat", "graphsage", "nnconv", "stgnn"} or graph_backend is not None

    def _resolve_record_graph_backend(self, selected_model: str, metadata_payload: dict):
        graph_backend = metadata_payload.get("graph_backend")
        if graph_backend:
            return str(graph_backend).strip().lower()
        model = str(selected_model or "").strip().lower()
        if model == "stgnn":
            return str(getattr(self.args, "graph_model", "gcn")).strip().lower()
        if model in {"gcn", "gat", "graphsage", "nnconv"}:
            return model
        return None

    def _extract_graph_stats(self, *, selected_model: str, graph_backend, state=None):
        if not self._is_graph_aware_model(selected_model, graph_backend):
            return None
        if isinstance(state, dict) and state.get("graph_stats") is not None:
            return state.get("graph_stats")
        graph_builder = getattr(self, "graphBuilder", None)
        if graph_builder is not None and getattr(graph_builder, "graph_stats", None) is not None:
            return graph_builder.graph_stats
        return None

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
        state=None,
    ) -> None:
        eval_result = getattr(result, "eval_result", None) if result is not None else None

        metadata_payload = self._serialise_metadata(eval_result)
        compute_payload = self._extract_compute_payload(metadata_payload)
        capacity_payload = self._extract_capacity_payload(metadata_payload)

        model_key = str(selected_model or "").strip().lower()
        graph_backend = self._resolve_record_graph_backend(model_key, metadata_payload)

        if model_key == "stgnn":
            record_graph_model = str(
                graph_backend or getattr(self.args, "graph_model", "gcn")
            ).strip().lower()
        elif model_key in {"gcn", "gat", "graphsage", "nnconv"}:
            record_graph_model = model_key
        else:
            record_graph_model = ""
            graph_backend = None

        graph_stats_payload = self._extract_graph_stats(
            selected_model=model_key,
            graph_backend=graph_backend,
            state=state,
        )

        run_id = self.experiment_store.make_run_id(
            model=model_key,
            ticker=stock,
            seed=int(self.args.seed),
            graph_backend=graph_backend,
        )

        ts_end = self.experiment_store.utc_now_iso()

        start_dt = datetime.fromisoformat(ts_start)
        end_dt = datetime.fromisoformat(ts_end)
        duration_sec = (end_dt - start_dt).total_seconds()

        metrics_payload = self._serialise_metrics(
            getattr(result, "metrics", None) if result is not None else None
        )

        common_payload = {
            "run_id": run_id,
            "job_id": job_id,
            "queue_run_id": queue_run_id,
            "status": status,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "duration_sec": duration_sec,
            "ticker": str(stock).upper(),
            "prediction_window": str(gui_window),
            "model": model_key,
            "seed": int(self.args.seed),
            "graph_backend": graph_backend,
            "graph_model": record_graph_model or None,
            "k": int(getattr(self.args, "k", 0)),
            "graph_mode": str(getattr(self.args, "graph_mode", "unknown")),
            "graph_embed": str(getattr(self.args, "graph_embed", "unknown")),
            "graph_ablation": str(getattr(self.args, "graph_ablation", "none")),
            "ablate_feature": str(getattr(self.args, "ablate_feature", "none")),
            "seq_len": int(getattr(self.args, "seq_len", 0)),
            "batch_size": int(getattr(self.args, "batch_size", 0)),
            "lstm_epochs": int(getattr(self.args, "lstm_epochs", 0)),
            "stgnn_epochs": int(getattr(self.args, "stgnn_epochs", 0)),
            "deterministic": bool(getattr(self.args, "deterministic", True)),
            "direction": str(getattr(result, "direction", "")) if result is not None else "",
            "confidence": float(getattr(result, "confidence", 0.0)) if result is not None else 0.0,
            "metrics": metrics_payload,
            "metadata": metadata_payload,
            "compute": compute_payload,
            "capacity": capacity_payload,
            "data_quality": metadata_payload.get("data_quality"),
            "graph_stats": graph_stats_payload,
            "error_message": error_message,
        }

        canonical_result_path = self.experiment_store.save_run_payload(
            run_id=run_id,
            filename="result.json",
            model=model_key,
            graph_backend=graph_backend,
            payload=common_payload,
        )

        config_snapshot_path = self.experiment_store.save_run_payload(
            run_id=run_id,
            filename="config.json",
            model=model_key,
            graph_backend=graph_backend,
            payload=vars(self.args),
        )

        graph_stats_path = None
        if graph_stats_payload is not None:
            graph_stats_path = self.experiment_store.save_run_payload(
                run_id=run_id,
                filename="graph_stats.json",
                model=model_key,
                graph_backend=graph_backend,
                payload=graph_stats_payload,
            )

        history_path = None
        if eval_result is not None and not queue_run_id:
            history_path = self.experiment_store.save_history(
                run_id=run_id,
                hist_train=getattr(eval_result, "hist_train", None),
                hist_val=getattr(eval_result, "hist_val", None),
                model=model_key,
                graph_model=record_graph_model,
            )

        queue_group = None
        job_payload_path = None

        if queue_run_id and job_id:
            queue_group = self.experiment_store.model_group(
                model_key,
                record_graph_model,
            )

            queue_payload = {
                **common_payload,
                "canonical_result_path": canonical_result_path,
                "queue_group": queue_group,
            }

            job_payload_path = self.experiment_store.save_job_payload(
                queue_run_id=queue_run_id,
                job_id=job_id,
                model=model_key,
                graph_model=record_graph_model,
                filename="result.json",
                payload=queue_payload,
            )

            self._active_queue_job_summaries.append(queue_payload)

        run_dir = self.experiment_store.run_dir(
            run_id,
            model=model_key,
            graph_backend=graph_backend,
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
            model=model_key,
            seed=int(self.args.seed),
            graph_model=record_graph_model,
            universe_id=str(getattr(self.args, "universe_id", "unknown")),
            interval=str(getattr(self.args, "interval", "unknown")),
            k=int(getattr(self.args, "k", 0)),
            graph_mode=str(getattr(self.args, "graph_mode", "unknown")),
            graph_embed=str(getattr(self.args, "graph_embed", "unknown")),
            graph_ablation=str(getattr(self.args, "graph_ablation", "none")),
            ablate_feature=str(getattr(self.args, "ablate_feature", "none")),
            threshold_policy=str(
                getattr(self.args, "decision_threshold_policy", "macro_f1_dense")
            ),
            deterministic=bool(getattr(self.args, "deterministic", True)),
            direction=str(getattr(result, "direction", "")) if result is not None else "",
            confidence=float(getattr(result, "confidence", 0.0)) if result is not None else 0.0,
            metrics=metrics_payload,
            extras={
                "run_dir": str(run_dir),
                "canonical_result_path": canonical_result_path,
                "config_snapshot_path": config_snapshot_path,
                "history_path": history_path,
                "graph_stats_path": graph_stats_path,
                "job_payload_path": job_payload_path,
                "queue_run": queue_run_id is not None,
                "queue_run_id": queue_run_id,
                "queue_group": queue_group,
                "metadata": metadata_payload,
                "compute": compute_payload,
                "capacity": capacity_payload,
                "graph_stats": graph_stats_payload,
                "error_message": error_message,
                "graph_backend": graph_backend,
            },
        )

        self.experiment_store.append_run(record)

    def startPipeline(self, gui_window: str, stock: str, stop_event: threading.Event):
        if self.pipeline_running:
            self.logger.warning("Pipeline already running.")
            self.frontendApp.set_status("Pipeline already running.")
            return (self.frontendApp.get_selected_model().upper(), "-", 0.0)

        self.pipeline_running = True
        ts_start = self.experiment_store.utc_now_iso()
        selected_model = self.frontendApp.get_selected_model()
        result = None
        state = None

        try:
            self.args.model = selected_model
            if self._active_queue_job_id is None:
                self._apply_frontend_experiment_controls()
            self._refresh_raw_feature_cols_from_ablation()

            self.frontendApp.root.after(
                0,
                lambda: self.frontendApp.set_active_model_titles(selected_model)
            )

            self.logger.info("[ModelSelection] %s", selected_model)

            use_headless_evaluator = self._active_queue_job_id is not None
            if use_headless_evaluator:
                evaluator = HeadlessEvaluator()
            else:
                evaluator = self.frontendApp.evaluator
                evaluator.reset_histories()
                self.frontendApp.root.after(0, self.frontendApp._reset_ui)

            self.frontendApp.set_status(f"Starting {selected_model.upper()}...")

            pipeline = Pipeline(self)
            state = pipeline.run(stock, gui_window, stop_event)
            graph_models = {"gcn", "gat", "graphsage", "nnconv", "stgnn"}
            if self._active_queue_job_id is None and selected_model in graph_models:
                self.frontendApp.render_graph_state(state)

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
                state=state,
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
                state=state,
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
                state=state,
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
            previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._request_gui_close)

            def _poll_sigint():
                try:
                    if not getattr(self.frontendApp, "_closing", False):
                        self.frontendApp.root.after(200, _poll_sigint)
                except Exception:
                    pass

            self.frontendApp.root.after(200, _poll_sigint)
            try:
                self.frontendApp.root.mainloop()
            except KeyboardInterrupt:
                self._request_gui_close()
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
                self.shutdown()
            return None

        if run_mode == "headless":
            result = None
            try:
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

                report_mode = str(
                    getattr(self.args, "headless_report", "compact")
                ).strip().lower()
                if report_mode == "full":
                    print()
                    print(self._format_headless_report(result))
                elif report_mode == "compact":
                    print(self._format_headless_result_line(result))
            finally:
                self._release_headless_resources(result)

            # Exit before Python unwinds this successful CLI frame. On some
            # Windows CUDA builds, releasing frame-local native objects first
            # can raise 0xC0000409 even though the result is already persisted.
            # Imported/library use and genuine exceptions never take this path.
            if _should_force_headless_cli_exit():
                _finish_successful_windows_headless_process(0)
            return result

        raise ValueError(
            f"Unknown run_mode '{self.args.run_mode}'. "
            f"Expected one of: gui, headless."
        )

    def run_headless(self, stock: str = None, gui_window: str = None, model_name: str = None):
        selected_stock, selected_window, selected_model, state = self.prepare_headless_state(
            stock=stock,
            gui_window=gui_window,
            model_name=model_name,
        )
        return self.run_headless_from_state(
            stock=selected_stock,
            gui_window=selected_window,
            model_name=selected_model,
            state=state,
        )

    def prepare_headless_state(
        self,
        stock: str = None,
        gui_window: str = None,
        model_name: str = None,
    ):
        """Load one universe and build reusable pipeline state for a worker."""

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

        pipeline = Pipeline(self)
        state = pipeline.run(selected_stock, selected_window, stop_event=None)
        return selected_stock, selected_window, selected_model, state

    def run_headless_from_state(
        self,
        *,
        stock: str,
        gui_window: str,
        model_name: str,
        state,
    ):
        """Run and persist one model using already prepared headless state."""

        selected_stock = str(stock).strip().upper()
        selected_window = str(gui_window).strip()
        selected_model = str(model_name).strip().lower()
        self.args.model = selected_model

        evaluator = HeadlessEvaluator()
        ts_start = self.experiment_store.utc_now_iso()
        result = None

        try:
            experiment_runner = ExperimentRunner(self)
            result = experiment_runner.run(
                model_name=selected_model,
                stock=selected_stock,
                state=state,
                evaluator=evaluator,
                stop_event=None,
            )

            self._store_run_result(
                stock=selected_stock,
                gui_window=selected_window,
                selected_model=selected_model,
                result=result,
                status="success",
                ts_start=ts_start,
                state=state,
            )
            return result
        except Exception as exc:
            try:
                self._store_run_result(
                    stock=selected_stock,
                    gui_window=selected_window,
                    selected_model=selected_model,
                    result=result,
                    status="failed",
                    ts_start=ts_start,
                    error_message=str(exc),
                    state=state,
                )
            except Exception:
                self.logger.exception("[Headless] Could not persist failed run")
            raise

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
    # Keep the returned result alive until the process boundary. Dropping it
    # before the next statement can itself trigger native object finalisation.
    _run_result = app.run()
    if (
        os.name == "nt"
        and str(getattr(app.args, "run_mode", "gui")).strip().lower() == "headless"
    ):
        _finish_successful_windows_headless_process(0)
