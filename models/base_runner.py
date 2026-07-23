from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gc
import threading
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score
try:
    from torch_geometric.loader import DataLoader as GeoDataLoader
except ImportError:
    GeoDataLoader = None

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
        if not getattr(app, "test_feats", None):
            raise RuntimeError(
                "Pipeline state has no held-out test split. Rebuild the pipeline with "
                "the current train/validation/test splitter."
            )
        train_df_stock = app.train_feats[stock].iloc[-app.min_train_len:]
        val_df_stock = app.val_feats[stock].iloc[-app.min_val_len:]
        test_df_stock = app.test_feats[stock].iloc[-app.min_test_len:]

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

        test_ds = RecurrentDataset(
            feature_dict={stock: test_df_stock},
            tickers=[stock],
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            prediction_horizon=app.horizon,
        )

        return train_ds, val_ds, test_ds, train_df_stock, val_df_stock, test_df_stock

    def _build_graph_datasets(self, app, stock):
        if not getattr(app, "test_feats", None):
            raise RuntimeError(
                "Pipeline state has no held-out test split. Rebuild the pipeline with "
                "the current train/validation/test splitter."
            )
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

        test_ds = STGNNDataset(
            graph_builder=app.graphBuilder,
            feature_dict=app.test_feats,
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
        if val_ds.tickers != aligned_tickers or test_ds.tickers != aligned_tickers:
            raise RuntimeError(
                "Train, validation and test ticker alignment must be identical for graph models."
            )
        return train_ds, val_ds, test_ds, aligned_tickers, num_nodes

    def _log_dataset_summary(self, app, train_ds, val_ds, test_ds, aligned_tickers=None) -> None:
        aligned_tickers = aligned_tickers or []
        app.logger.info(
            "[%sRunner] train=%d | validation=%d | test=%d | aligned_assets=%d",
            self.model_name,
            len(train_ds),
            len(val_ds),
            len(test_ds),
            len(aligned_tickers),
        )
        if len(train_ds) == 0:
            raise RuntimeError(f"{self.model_name} training dataset is empty.")
        if len(val_ds) == 0:
            raise RuntimeError(f"{self.model_name} validation dataset is empty.")
        if len(test_ds) == 0:
            raise RuntimeError(f"{self.model_name} test dataset is empty.")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _make_torch_loader(self, app, dataset, *, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=app.args.batch_size,
            shuffle=shuffle,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

    def _make_geo_loader(self, app, dataset, *, shuffle=True):
        if GeoDataLoader is None:
            raise RuntimeError(
                "Graph models require torch_geometric. Install PyG before running GCN/GAT/GraphSAGE/NNConv/STGNN."
            )
        return GeoDataLoader(
            dataset,
            batch_size=app.args.batch_size,
            shuffle=shuffle,
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
        train_dataset=None,
    ):
        trainer = Trainer(
            model,
            Utils.make_adamw(model, lr=lr, weight_decay=app.args.weight_decay),
            self._make_classification_loss(app, train_dataset),
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
            lr_scheduler=getattr(app.args, "lr_scheduler", "reduce_on_plateau"),
            lr_plateau_factor=getattr(app.args, "lr_plateau_factor", 0.5),
            lr_plateau_patience=getattr(app.args, "lr_plateau_patience", 5),
            lr_plateau_min_lr=getattr(app.args, "lr_plateau_min_lr", 0.0),
            cpu_power_watts=getattr(app.args, "cpu_power_watts", None),
            training_log=getattr(app.args, "training_log", "summary"),
        )
        trainer.train_examples_unique = (
            int(len(train_dataset)) if train_dataset is not None else None
        )
        trainer.training_sample_unit = "supervised_window"
        return trainer

    def _make_classification_loss(self, app, train_dataset=None):
        if str(getattr(app.args, "class_balance", "auto")).strip().lower() != "auto":
            return BCEWithLogitsLoss()

        labels = self._extract_binary_labels(train_dataset)
        if labels is None or labels.numel() == 0:
            return BCEWithLogitsLoss()

        positives = float(labels.sum().item())
        total = float(labels.numel())
        negatives = total - positives

        if positives <= 0.0 or negatives <= 0.0:
            app.logger.warning(
                "[%sRunner] class_balance=auto skipped: one-class training labels "
                "(positives=%d negatives=%d)",
                self.model_name,
                int(positives),
                int(negatives),
            )
            return BCEWithLogitsLoss()

        pos_weight = negatives / positives
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=app.device)
        app.logger.info(
            "[%sRunner] class_balance=auto | train positives=%d negatives=%d pos_weight=%.3f",
            self.model_name,
            int(positives),
            int(negatives),
            pos_weight,
        )
        return BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    @staticmethod
    def _extract_binary_labels(dataset):
        if dataset is None:
            return None

        for attr in ("y", "y_list"):
            labels = getattr(dataset, attr, None)
            if labels is not None:
                return torch.as_tensor(labels, dtype=torch.float32).view(-1)

        values = []
        try:
            for i in range(len(dataset)):
                sample = dataset[i]
                y = getattr(sample, "y", None)
                if y is None and isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    y = sample[1]
                if y is not None:
                    values.append(float(torch.as_tensor(y).view(-1)[0].item()))
        except Exception:
            return None

        if not values:
            return None
        return torch.tensor(values, dtype=torch.float32)

    def _set_target_from_dataset(self, trainer, model, dataset) -> None:
        target_idx = getattr(dataset, "target_idx", None)
        trainer.targetIdx = target_idx
        model.target_node_index = target_idx

    def _train_model(
        self,
        *,
        app,
        trainer,
        dataloader,
        validation_dataloader,
        epochs,
        stop_event,
    ) -> None:
        trainer.train(
            dataloader,
            epochs,
            validation_dataloader=validation_dataloader,
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

        policy = getattr(app.args, "decision_threshold_policy", "macro_f1_dense")
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
            "data_quality": getattr(app, "data_quality", None),
            "architecture": self._architecture_metadata(app),
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
        model = getattr(trainer, "model", None)
        capacity = self._neural_capacity_metadata(model)
        eval_result.metadata.update({
            "energy_wh": getattr(trainer, "total_energy_Wh", None),
            "train_seconds": getattr(trainer, "total_train_seconds", None),
            "avg_power_w": getattr(trainer, "avg_power_W", None),
            "energy_per_sample_wh": getattr(trainer, "energy_per_sample_Wh", None),
            "energy_measurement_method": getattr(trainer, "energy_measurement_method", None),
            "cpu_power_watts": getattr(trainer, "cpu_power_watts", None),
            "train_samples": getattr(trainer, "total_samples", None),
            "training_sample_unit": getattr(
                trainer,
                "training_sample_unit",
                "supervised_window",
            ),
            "train_examples_unique": getattr(
                trainer,
                "train_examples_unique",
                None,
            ),
            "sample_exposures": getattr(trainer, "total_samples", None),
            "epochs_completed": len(getattr(trainer, "train_loss_history", []) or []),
            "total_params": self._count_total_params(model),
            "trainable_params": self._count_trainable_params(model),
            "capacity": capacity,
            "lr_scheduler": getattr(trainer, "lr_scheduler_name", None),
            "lr_history": getattr(trainer, "lr_history", None),
            "epoch_val_loss_history": getattr(trainer, "epoch_val_loss_history", None),
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if getattr(trainer.device, "type", None) == "cuda"
                else None
            ),
        })

    @staticmethod
    def _count_total_params(model) -> int | None:
        if model is None or not hasattr(model, "parameters"):
            return None
        return int(sum(p.numel() for p in model.parameters()))

    @staticmethod
    def _count_trainable_params(model) -> int | None:
        if model is None or not hasattr(model, "parameters"):
            return None
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    @classmethod
    def _neural_capacity_metadata(cls, model) -> Dict[str, Any]:
        total_parameters = cls._count_total_params(model)
        trainable_parameters = cls._count_trainable_params(model)
        parameter_storage_bytes = None

        if model is not None and hasattr(model, "parameters"):
            try:
                parameter_storage_bytes = int(
                    sum(p.numel() * p.element_size() for p in model.parameters())
                )
            except Exception:
                parameter_storage_bytes = None

        return {
            "family": "neural",
            "primary_measure": "trainable_parameters",
            "primary_value": trainable_parameters,
            "parameter_storage_bytes": parameter_storage_bytes,
            "neural_total_parameters": total_parameters,
            "neural_trainable_parameters": trainable_parameters,
        }

    def _architecture_metadata(self, app) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "model_family": str(self.model_name or "").strip().lower(),
            "lstm_hidden": getattr(app.args, "lstm_hidden", None),
            "lstm_layers": getattr(app.args, "lstm_layers", None),
            "bidirectional": getattr(app.args, "bidirectional", None),
            "gcn_hidden": getattr(app.args, "gcn_hidden", None),
            "stgnn_blocks": getattr(app.args, "stgnn_blocks", None),
            "tcn_channels": getattr(app.args, "tcn_channels", None),
            "tcn_kernel_size": getattr(app.args, "tcn_kernel_size", None),
            "rep_dim": getattr(app.args, "rep_dim", None),
            "head_hidden": getattr(app.args, "head_hidden", None),
            "dropout": getattr(app.args, "dropout", None),
            "class_balance": getattr(app.args, "class_balance", None),
            "weight_decay": getattr(app.args, "weight_decay", None),
            "rf_estimators": getattr(app.args, "rf_estimators", None),
            "rf_max_depth": getattr(app.args, "rf_max_depth", None),
            "rf_min_samples_leaf": getattr(app.args, "rf_min_samples_leaf", None),
        }

    def _cpu_energy_metadata(self, app, *, train_seconds: float, train_samples: int | None = None) -> Dict[str, Any]:
        cpu_power_watts = getattr(app.args, "cpu_power_watts", None)
        train_seconds = float(train_seconds or 0.0)

        if cpu_power_watts is None:
            return {
                "energy_wh": None,
                "avg_power_w": None,
                "energy_per_sample_wh": None,
                "energy_measurement_method": "unavailable_cpu_power_not_configured",
                "cpu_power_watts": None,
            }

        cpu_power_watts = float(cpu_power_watts)
        energy_wh = cpu_power_watts * (train_seconds / 3600.0)
        if train_samples and int(train_samples) > 0:
            energy_per_sample_wh = energy_wh / int(train_samples)
        else:
            energy_per_sample_wh = None

        return {
            "energy_wh": float(energy_wh),
            "avg_power_w": float(cpu_power_watts),
            "energy_per_sample_wh": energy_per_sample_wh,
            "energy_measurement_method": "cpu_power_estimate",
            "cpu_power_watts": float(cpu_power_watts),
        }

    # ------------------------------------------------------------------
    # Evaluation and prediction
    # ------------------------------------------------------------------

    def _resolve_threshold(self, metrics, policy: str = "macro_f1_dense") -> float:
        if metrics is None:
            return 0.5
        policy = str(policy or "fixed").strip().lower()
        if policy == "fixed":
            return float(getattr(metrics, "threshold_fixed", 0.5))
        if policy == "macro_f1_dense":
            return float(getattr(metrics, "threshold_macro_f1_dense", 0.5))
        raise ValueError(f"Unknown threshold policy: {policy}")

    @staticmethod
    def _macro_f1_threshold(result) -> tuple[float, float]:
        y_true = np.asarray(getattr(result, "y_true", []) or [], dtype=int)
        probs = np.asarray(getattr(result, "probs", []) or [], dtype=float)
        if y_true.size == 0 or probs.size == 0:
            return 0.5, 0.0

        thresholds = np.unique(np.concatenate(([0.0], probs, [1.0])))
        best_threshold = 0.5
        best_score = -1.0
        for threshold in thresholds:
            predictions = (probs >= threshold).astype(int)
            score = float(f1_score(y_true, predictions, average="macro", zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold, best_score

    def _calibrate_threshold(self, app, validation_result) -> dict:
        policy = str(
            getattr(app.args, "decision_threshold_policy", "macro_f1_dense")
        ).strip().lower()
        macro_threshold, macro_score = self._macro_f1_threshold(validation_result)
        if policy == "fixed":
            selected_threshold = 0.5
        elif policy == "macro_f1_dense":
            selected_threshold = macro_threshold
        else:
            raise ValueError(f"Unknown threshold policy: {policy}")

        y_true = np.asarray(getattr(validation_result, "y_true", []) or [], dtype=int)
        probs = np.asarray(getattr(validation_result, "probs", []) or [], dtype=float)
        if y_true.size and probs.size:
            selected_score = float(
                f1_score(
                    y_true,
                    (probs >= selected_threshold).astype(int),
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            selected_score = 0.0

        return {
            "split": "validation",
            "policy": policy,
            "selected_threshold": float(selected_threshold),
            "macro_f1_threshold": float(macro_threshold),
            "macro_f1_at_selected_threshold": selected_score,
            "macro_f1_at_optimal_threshold": float(macro_score),
            "dense_loss": getattr(validation_result, "dense_val_loss", None),
            "n_predictions": len(getattr(validation_result, "y_true", []) or []),
        }

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
        test_ds,
        live_predict_fn: Callable[[], float],
        eval_status: str,
        predict_status: str,
    ) -> ModelRunResult:
        app._check_stop(stop_event)
        app.frontendApp.set_status(eval_status)

        self._apply_temperature(app, model)
        validation_result = trainer.evaluate_rolling(val_ds, split_name="validation")
        threshold_selection = self._calibrate_threshold(app, validation_result)
        trainer.decision_threshold = float(threshold_selection["selected_threshold"])
        trainer.decision_threshold_policy = str(threshold_selection["policy"])

        eval_result = trainer.evaluate_rolling(test_ds, split_name="test")
        eval_result.metadata.update({
            "evaluation_split": "test",
            "threshold_source": "validation",
            "decision_threshold_policy": threshold_selection["policy"],
            "validation_selection": threshold_selection,
            "validation_loss_dense": threshold_selection["dense_loss"],
            "test_loss_dense": getattr(eval_result, "dense_val_loss", None),
            "threshold_macro_f1_validation": threshold_selection["macro_f1_threshold"],
        })
        self._attach_metadata(app, eval_result, stock=stock)
        self._attach_compute_metadata(trainer, eval_result)

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
            policy=getattr(app.args, "decision_threshold_policy", "macro_f1_dense"),
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
        if threading.current_thread() is threading.main_thread():
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
