from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from evaluation.evaluation_types import EvaluationMetrics


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
    Thin runner contract for extracting per-model execution logic
    out of main.py without changing the underlying Trainer/model code.
    """

    model_name: str = "BASE"

    @abstractmethod
    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        """
        Execute training, evaluation, and final prediction for one model.
        `app` is the MainApp instance so the runner can reuse existing helpers/state.
        """
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
    
    def _attach_metadata(self, app, eval_result):
        if eval_result is None:
            return

        policy = getattr(app.args, "decision_threshold_policy", "fixed")
        eval_result.metadata["decision_threshold_policy"] = policy