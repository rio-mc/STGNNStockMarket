from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelRunResult:
    model_name: str
    direction: str
    confidence: float
    metrics: Dict[str, Any]
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