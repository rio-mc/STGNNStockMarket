from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

@dataclass
class EvaluationResult:
    """
    Creates a custom object of evaluation results for future analysis and plotting.
    """
    y_true: List[int]
    y_pred: List[int]
    probs: List[float]
    prediction_dates: List[pd.Timestamp]

    # ====================================
    # === Training and validation losses
    hist_lstm: Optional[List[float]] = None
    hist_stgnn: Optional[List[float]] = None
    val_lstm: Optional[List[Dict[str, float]]] = None
    val_stgnn: Optional[List[Dict[str, float]]] = None

    # ====================================
    # === Prediction horizon - necessary for plotting
    horizon: Optional[int] = None
