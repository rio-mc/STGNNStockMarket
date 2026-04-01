from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

@dataclass
class EvaluationResult:
    """
    Unified evaluation result object for any model type (LSTM, GRU, STGNN, etc.).
    Tracks both raw prediction outcomes and optional training/validation histories.
    """
    y_true: List[int]
    y_pred: List[int]
    probs: List[float]
    prediction_dates: List[pd.Timestamp]

    # === Training and validation loss histories (generic)
    hist_train: Optional[List[float]] = None
    hist_val: Optional[List[Dict[str, float]]] = None

    # === Model-specific legacy attributes (for backwards compatibility)
    hist_lstm: Optional[List[float]] = None
    hist_stgnn: Optional[List[float]] = None
    hist_gru: Optional[List[float]] = None
    val_lstm: Optional[List[Dict[str, float]]] = None
    val_stgnn: Optional[List[Dict[str, float]]] = None
    val_gru: Optional[List[Dict[str, float]]] = None

    # === Prediction horizon
    horizon: Optional[int] = None

    # === Metadata
    model_name: Optional[str] = None
