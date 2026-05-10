from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd


@dataclass
class EvaluationResult:
    """
    Unified single-model evaluation result.

    Required:
    - y_true, y_pred, probs must have the same length
    - prediction_dates should either be empty or match prediction length
    - hist_train is a per-epoch numeric loss series
    - hist_val is typically a list like:
        [{"date": <timestamp>, "loss": <float>}, ...]

    Optional:
    - decision_threshold: threshold used for fixed-threshold evaluation
    - dense_val_loss: mean validation loss across dense rolling evaluation
    - horizon: prediction horizon in bars
    - model_name: active model for this run
    - metadata: free-form auxiliary fields
    """

    y_true: List[int]
    y_pred: List[int]
    probs: List[float]
    prediction_dates: List[pd.Timestamp]

    hist_train: List[float] = field(default_factory=list)
    hist_val: List[Dict[str, Any]] = field(default_factory=list)

    decision_threshold: float = 0.5
    threshold_selection_metric: str = "macro_f1"

    dense_val_loss: Optional[float] = None

    horizon: Optional[int] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        n_true = len(self.y_true)
        n_pred = len(self.y_pred)
        n_probs = len(self.probs)

        if not (n_true == n_pred == n_probs):
            raise ValueError(
                "EvaluationResult length mismatch: "
                f"len(y_true)={n_true}, len(y_pred)={n_pred}, len(probs)={n_probs}"
            )

        if self.prediction_dates and len(self.prediction_dates) != n_true:
            raise ValueError(
                "EvaluationResult prediction_dates length mismatch: "
                f"len(prediction_dates)={len(self.prediction_dates)}, expected={n_true}"
            )

        if self.hist_train:
            bad_train = [x for x in self.hist_train if not isinstance(x, (int, float))]
            if bad_train:
                raise ValueError("hist_train must contain only numeric values")

        if self.hist_val:
            for i, row in enumerate(self.hist_val):
                if not isinstance(row, dict):
                    raise ValueError(f"hist_val[{i}] must be a dict")
                if "loss" not in row:
                    raise ValueError(f"hist_val[{i}] missing required key 'loss'")

        try:
            self.decision_threshold = float(self.decision_threshold)
        except Exception as exc:
            raise ValueError(f"decision_threshold must be numeric, got {self.decision_threshold!r}") from exc
        
        allowed_threshold_metrics = {
            "macro_f1",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "youden_j",
            "sharpe",
        }

        if self.threshold_selection_metric not in allowed_threshold_metrics:
            raise ValueError(
                f"Unsupported threshold_selection_metric="
                f"{self.threshold_selection_metric!r}"
            )
        
        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError(
                f"decision_threshold must be in [0, 1], got {self.decision_threshold}"
            )

        if self.dense_val_loss is not None:
            try:
                self.dense_val_loss = float(self.dense_val_loss)
            except Exception as exc:
                raise ValueError(f"dense_val_loss must be numeric, got {self.dense_val_loss!r}") from exc
            
            
@dataclass
class EvaluationMetrics:
    """
    Typed evaluation summary returned by EvaluationMethods.

    This replaces the loose metrics dict and gives downstream code
    a stable contract.
    """

    model: str

    threshold_fixed: float
    threshold_macro_f1_dense: float
    threshold_selection_metric: str
    
    val_loss_dense: Optional[float]

    accuracy_dense: float
    f1_dense: float
    macro_f1_dense: float
    roc_auc_dense: Optional[float]
    ap_dense: Optional[float]

    accuracy_dense_macro_f1_threshold: float
    f1_dense_macro_f1_threshold: float
    macro_f1_dense_macro_f1_threshold: float

    accuracy_trade_aligned: float
    f1_trade_aligned: float
    macro_f1_trade_aligned: float
    roc_auc_trade_aligned: Optional[float]
    ap_trade_aligned: Optional[float]

    sharpe: Optional[float]
    n_trades: int
    mean_trade_return: Optional[float]
    hit_rate: Optional[float]
    final_equity: Optional[float]
    max_drawdown: Optional[float]

    ticker: Optional[str]
    n_predictions_dense: int
    n_predictions_trade_aligned: int
    horizon: Optional[int]

    energy_wh: Optional[float]
    train_seconds: Optional[float]
    avg_power_w: Optional[float]
    energy_per_sample_wh: Optional[float]
    train_samples: Optional[int]
    gpu_peak_memory_mb: Optional[float]

    graph_backend: Optional[str] = None
    graph_model: Optional[str] = None