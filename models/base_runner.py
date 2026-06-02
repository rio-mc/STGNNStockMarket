from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gc
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader as GeoDataLoader

from core.utils.utils import Utils
from data import RecurrentDataset, STGNNDataset
from evaluation.evaluation_types import EvaluationMetrics
from training.trainer import Trainer


@dataclass
class ModelRunResult:
    model_name: str
    direction: str
    confidence: float
    metrics: Optional[EvaluationMetrics]
    eval_result: Any
    trainer: Any = None
    model: Any = None
    extras: Optional[Dict[str, Any]] = None


class BaseModelRunner(ABC):
    """Shared model-runner contract.

    Concrete runners should only build their model and select the appropriate
    live-prediction helper. Dataset construction, loaders, training,
    evaluation, metadata, compute telemetry, thresholding and cleanup all live
    here to keep runner behaviour consistent across architectures.
    """

    model_name: str = "BASE"

    @abstractmethod
    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_recurrent_datasets(self, app, stock):
        train_df_stock = app.train_feats[stock].iloc[-app.min_train_len:]
        val_df_stock = app.val_feats[stock].iloc[-app.min_val_len:]

        train_ds = RecurrentDataset(
            feature_dict={stock: train_df_stock},
            tickers=[stock],
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            prediction_horizon=app.horizon,
        )

        val_ds = RecurrentDataset(
            feature_dict={stock: val_df_stock},
            tickers=[stock],
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            prediction_horizon=app.horizon,
        )

        return train_ds, val_ds, train_df_stock, val_df_stock

    def _build_graph_datasets(self, app, stock):
        train_ds = STGNNDataset(
            graph_builder=app.graphBuilder,
            feature_dict=app.train_feats,
            tickers=app.args.tickers,
            edge_index=app.init_edge_index,
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            horizon=app.horizon,
            include_target_flag=True,
        )

        val_ds = STGNNDataset(
            graph_builder=app.graphBuilder,
            feature_dict=app.val_feats,
            tickers=app.args.tickers,
            edge_index=app.init_edge_index,
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            horizon=app.horizon,
            include_target_flag=True,
        )

        aligned_tickers = train_ds.tickers
        num_nodes = len(aligned_tickers)
        return train_ds, val_ds, aligned_tickers, num_nodes

    def _log_dataset_summary(self, app, train_ds, val_ds, aligned_tickers=None) -> None:
        aligned_tickers = aligned_tickers or []
        app.logger.info(
            "[%sRunner] train_ds size=%d | val_ds size=%d | aligned_tickers=%s",
            self.model_name,
            len(train_ds),
            len(val_ds),
            aligned_tickers,
        )
        if len(train_ds) == 0:
            raise RuntimeError(f"{self.model_name} training dataset is empty.")
        if len(val_ds) == 0:
            raise RuntimeError(f"{self.model_name} validation dataset is empty.")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _make_torch_loader(self, app, train_ds):
        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=app.args.batch_size,
            shuffle=True,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

    def _make_geo_loader(self, app, train_ds):
        return GeoDataLoader(
            train_ds,
            batch_size=app.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

    # Backwards-compatible aliases.
    _build_recurrent_dataloader = _make_torch_loader
    _build_graph_dataloader = _make_geo_loader

    # ------------------------------------------------------------------
    # Trainer lifecycle
    # ------------------------------------------------------------------

    def _make_trainer(
        self,
        *,
        app,
        model,
        stock,
        evaluator,
        lr,
        graph_builder=None,
        features=None,
        tickers=None,
    ):
        return Trainer(
            model,
            Utils.make_adamw(model, lr=lr, weight_decay=app.args.weight_decay),
            BCEWithLogitsLoss(),
            app.device,
            graphBuilder=graph_builder,
            features=features,
            tickers=tickers or getattr(app, "tickers", [stock]),
            targetTicker=stock,
            frontend=app.frontendApp,
            evaluator=evaluator,
            prediction_horizon=app.horizon,
            seq_len=app.seq_len,
            model_name=self.model_name,
        )

    def _set_target_from_dataset(self, trainer, model, dataset) -> None:
        target_idx = getattr(dataset, "target_idx", None)
        trainer.targetIdx = target_idx
        model.target_node_index = target_idx

    def _train_model(self, *, app, trainer, dataloader, epochs, stop_event) -> None:
        trainer.train(
            dataloader,
            epochs,
            stop_event=stop_event,
            patience=app.args.early_stopping_patience,
        )
        self._log_after_memory(self.model_name)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _resolve_graph_backend(self, app) -> str | None:
        model = str(self.model_name or "").strip().lower()
        if model == "stgnn":
            return str(getattr(app.args, "graph_model", "gcn")).strip().lower()
        if model in {"gcn", "gat", "graphsage", "nnconv"}:
            return model
        return None

    def _attach_metadata(self, app, eval_result, stock: str | None = None) -> None:
        if eval_result is None:
            return

        policy = getattr(app.args, "decision_threshold_policy", "fixed")
        graph_backend = self._resolve_graph_backend(app)
        model_family = str(self.model_name or "").strip().lower()

        metadata = {
            "decision_threshold_policy": policy,
            "ticker": stock or eval_result.metadata.get("ticker"),
            "model": self.model_name,
            "model_family": model_family,
            "graph_backend": graph_backend,
            "graph_model": graph_backend if model_family == "stgnn" else None,
            "seed": getattr(app.args, "seed", None),
            "universe_id": getattr(app.args, "universe_id", None),
            "interval": getattr(app.args, "effective_interval", getattr(app.args, "interval", None)),
        }

        # Only genuinely graph-aware models should carry graph construction
        # settings. Panel/recurrent models may use panel-shaped tensors but do
        # not consume graph edges.
        if graph_backend is not None:
            metadata.update({
                "k": getattr(app.args, "k", None),
                "graph_mode": getattr(app.args, "graph_mode", None),
                "graph_embed": getattr(app.args, "graph_embed", None),
            })

        eval_result.metadata.update(metadata)

    def _attach_compute_metadata(self, trainer, eval_result) -> None:
        if eval_result is None:
            return
        eval_result.metadata.update({
            "energy_wh": getattr(trainer, "total_energy_Wh", None),
            "train_seconds": getattr(trainer, "total_train_seconds", None),
            "avg_power_w": getattr(trainer, "avg_power_W", None),
            "energy_per_sample_wh": getattr(trainer, "energy_per_sample_Wh", None),
            "train_samples": getattr(trainer, "total_samples", None),
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if getattr(trainer.device, "type", None) == "cuda"
                else None
            ),
        })

    # ------------------------------------------------------------------
    # Evaluation and prediction
    # ------------------------------------------------------------------

    def _resolve_threshold(self, metrics, policy: str = "fixed") -> float:
        if metrics is None:
            return 0.5
        policy = str(policy or "fixed").strip().lower()
        if policy == "fixed":
            return float(getattr(metrics, "threshold_fixed", 0.5))
        if policy == "macro_f1_dense":
            return float(getattr(metrics, "threshold_macro_f1_dense", 0.5))
        raise ValueError(f"Unknown threshold policy: {policy}")

    def _apply_temperature(self, app, model) -> None:
        if hasattr(model, "classifier") and hasattr(model.classifier, "set_temperature"):
            model.classifier.set_temperature(getattr(app.args, "head_temperature", 1.0))

    def _direction_from_probability(self, prob: float, threshold: float):
        prob = float(prob)
        threshold = float(threshold)
        if prob >= threshold:
            return "Upwards", prob * 100.0
        return "Downwards", (1.0 - prob) * 100.0

    def _evaluate_and_predict(
        self,
        *,
        app,
        stock: str,
        price_df,
        evaluator,
        stop_event,
        model,
        trainer,
        val_ds,
        live_predict_fn: Callable[[], float],
        eval_status: str,
        predict_status: str,
    ) -> ModelRunResult:
        app._check_stop(stop_event)
        app.frontendApp.set_status(eval_status)

        eval_result = trainer.evaluate_rolling(val_ds)
        self._attach_metadata(app, eval_result, stock=stock)
        self._attach_compute_metadata(trainer, eval_result)
        self._apply_temperature(app, model)

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)
        app.frontendApp.set_status(predict_status)

        model.eval()
        with torch.no_grad():
            prob = float(live_predict_fn())

        threshold = self._resolve_threshold(
            metrics,
            policy=getattr(app.args, "decision_threshold_policy", "fixed"),
        )
        direction, confidence = self._direction_from_probability(prob, threshold)

        app.logger.info("[%sRunner] %s (%.1f%%)", self.model_name, direction, confidence)

        return ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=trainer,
            model=model,
        )

    # ------------------------------------------------------------------
    # Live prediction helpers
    # ------------------------------------------------------------------

    def _live_recurrent_probability(self, app, model, val_df_stock) -> float:
        live_x = val_df_stock[app.raw_feature_cols].iloc[-app.seq_len:].values.astype(np.float32)
        x = torch.tensor(live_x, dtype=torch.float32, device=app.device).unsqueeze(0)
        return torch.sigmoid(model(x).view(-1)[0]).item()

    def _live_graph_probability(self, app, model, val_ds) -> float:
        live_graph = val_ds[len(val_ds) - 1]
        x_live = live_graph.x.unsqueeze(0).to(app.device)
        edge_index_live = live_graph.edge_index.to(app.device)
        edge_attr_live = (
            live_graph.edge_attr.to(app.device)
            if getattr(live_graph, "edge_attr", None) is not None
            else None
        )
        logits = model(
            x_live,
            edge_index=edge_index_live,
            edge_attr=edge_attr_live,
            target_node_index=val_ds.target_idx,
        )
        return torch.sigmoid(logits.view(-1)[0]).item()

    def _live_panel_probability(self, app, model, val_ds, *, pass_target_index: bool) -> float:
        live_graph = val_ds[len(val_ds) - 1]
        x_live = live_graph.x.unsqueeze(0).to(app.device)
        if pass_target_index:
            logits = model(x_live, target_node_index=val_ds.target_idx)
        else:
            logits = model(x_live)
        return torch.sigmoid(logits.view(-1)[0]).item()

    # ------------------------------------------------------------------
    # Memory and cleanup
    # ------------------------------------------------------------------

    def _prepare_memory_logging(self, label: str) -> None:
        if getattr(getattr(self, "device", None), "type", None) == "cuda":
            torch.cuda.reset_peak_memory_stats()
        Utils.log_gpu_memory(f"Before {label}")

    def _log_after_memory(self, label: str) -> None:
        Utils.log_gpu_memory(f"After {label}")

    def _cleanup(self, *objects) -> None:
        del objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
