from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gc
import torch

from evaluation.evaluation_types import EvaluationMetrics
from core.utils.utils import Utils


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
    """
    Shared runner contract and lifecycle helpers.

    Concrete runners should own model-specific construction:
    - dataset creation
    - model creation
    - trainer creation
    - live prediction tensor preparation

    This base class owns shared bookkeeping:
    - threshold policy
    - metadata attachment
    - compute metadata attachment
    - temperature scaling
    - final evaluation/prediction packaging
    - cleanup
    """

    model_name: str = "BASE"

    @abstractmethod
    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        raise NotImplementedError

    def _resolve_threshold(self, metrics, policy: str = "fixed") -> float:
        if metrics is None:
            return 0.5

        policy = str(policy or "fixed").strip().lower()

        if policy == "fixed":
            thr = getattr(metrics, "threshold_fixed", None)
            return float(thr) if thr is not None else 0.5

        if policy == "macro_f1_dense":
            thr = getattr(metrics, "threshold_macro_f1_dense", None)
            return float(thr) if thr is not None else 0.5

        raise ValueError(f"Unknown threshold policy: {policy}")

    def _prepare_memory_logging(self, label: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        Utils.log_gpu_memory(f"Before {label}")

    def _log_after_memory(self, label: str) -> None:
        Utils.log_gpu_memory(f"After {label}")

    def _attach_metadata(self, app, eval_result, stock: str | None = None) -> None:
        if eval_result is None:
            return

        policy = getattr(app.args, "decision_threshold_policy", "fixed")

        eval_result.metadata.update({
            "decision_threshold_policy": policy,
            "ticker": stock or eval_result.metadata.get("ticker"),
            "model": self.model_name,
            "seed": getattr(app.args, "seed", None),
            "universe_id": getattr(app.args, "universe_id", None),
            "interval": getattr(app.args, "interval", None),
            "k": getattr(app.args, "k", None),
            "graph_mode": getattr(app.args, "graph_mode", None),
            "graph_embed": getattr(app.args, "graph_embed", None),
            "graph_model": getattr(app.args, "graph_model", None),
        })

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
                if torch.cuda.is_available()
                else None
            ),
        })

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

        app.logger.info(
            "[%sRunner] %s (%.1f%%)",
            self.model_name,
            direction,
            confidence,
        )

        return ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=trainer,
            model=model,
        )

    def _cleanup(self, *objects) -> None:
        del objects
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()