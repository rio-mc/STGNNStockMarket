from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd


@dataclass
class EvaluationResult:
    """
    Unified single-model evaluation result.

    This replaces the legacy pattern where one result object carried
    model-specific history fields such as hist_lstm, hist_gru, and
    hist_stgnn.

    Fields:
    - y_true / y_pred / probs: prediction outputs
    - prediction_dates: timestamps aligned to predictions
    - hist_train: per-epoch training loss series
    - hist_val: validation loss entries, typically:
        [{"date": <timestamp>, "loss": <float>}, ...]
    - horizon: prediction horizon in bars
    - model_name: active model for this run
    - metadata: optional extra run metadata
    """

    y_true: List[int]
    y_pred: List[int]
    probs: List[float]
    prediction_dates: List[pd.Timestamp]

    hist_train: List[float] = field(default_factory=list)
    hist_val: List[Dict[str, Any]] = field(default_factory=list)

    horizon: Optional[int] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)