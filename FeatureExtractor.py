# FeatureExtractor.py
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(
        self,
        rawData: Dict[str, pd.DataFrame],
        rollingVolWindow: int = 5,
        norm_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        fit_normaliser: bool = True,
    ):
        self.rawData = rawData
        self.rollingVolWindow = rollingVolWindow

        self.featureCols = ['open', 'high', 'low', 'close', 'volume',
                            'return', 'volatility', 'momentum']
        self.dfFeats: Dict[str, pd.DataFrame] = {}

        # If provided, we reuse these (no leakage in val/test)
        self.norm_stats: Dict[str, Dict[str, Tuple[float, float]]] = norm_stats or {}
        self.fit_normaliser = fit_normaliser

    def buildFeatureDfs(self) -> None:
        """
        Transforms raw price DataFrames into engineered feature sets.

        Leakage-safe normalisation:
        - Train: fit_normaliser=True -> compute mean/std on TRAIN ONLY
        - Val/Test: fit_normaliser=False -> apply TRAIN mean/std
        """
        featureDfs: Dict[str, pd.DataFrame] = {}

        normalise_cols = ['open', 'high', 'low', 'close', 'volume',
                          'return', 'volatility', 'momentum']

        for tkr, df in self.rawData.items():
            if df.empty or "close" not in df.columns:
                print(f"[Warning] Skipping '{tkr}' — no usable 'close' column.")
                continue

            dfFeat = df.copy().astype(np.float32)

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in dfFeat.columns for col in required_cols):
                print(f"[Warning] Skipping '{tkr}' — missing essential price columns.")
                continue

            # === Feature engineering (causal)
            dfFeat["return"]     = dfFeat["close"].pct_change()
            dfFeat["volatility"] = dfFeat["close"].rolling(self.rollingVolWindow).std().shift(1)
            dfFeat["momentum"]   = dfFeat["close"].pct_change(periods=self.rollingVolWindow)

            dfFeat.dropna(inplace=True)
            dfFeat = dfFeat.replace([np.inf, -np.inf], np.nan).dropna()

            # === Normalisation (NO leakage)
            if self.fit_normaliser:
                tkr_stats: Dict[str, Tuple[float, float]] = {}
                for c in normalise_cols:
                    mu = float(dfFeat[c].mean())
                    sd = float(dfFeat[c].std(ddof=0))
                    if not np.isfinite(sd) or sd <= 0:
                        sd = 1.0
                    tkr_stats[c] = (mu, sd)
                self.norm_stats[tkr] = tkr_stats

            if tkr not in self.norm_stats:
                # Fallback: if val ticker missing stats, fit locally (still causal but less “clean”)
                # Better than crashing.
                print(f"[Warning] No norm_stats for '{tkr}'. Fitting locally as fallback.")
                tkr_stats = {}
                for c in normalise_cols:
                    mu = float(dfFeat[c].mean())
                    sd = float(dfFeat[c].std(ddof=0))
                    if not np.isfinite(sd) or sd <= 0:
                        sd = 1.0
                    tkr_stats[c] = (mu, sd)
                self.norm_stats[tkr] = tkr_stats

            for c in normalise_cols:
                mu, sd = self.norm_stats[tkr][c]
                dfFeat[c] = (dfFeat[c] - mu) / (sd + 1e-8)

            dfFeat = dfFeat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            self.featureCols = normalise_cols
            featureDfs[tkr] = dfFeat

        if not featureDfs:
            raise ValueError("No valid tickers with feature data found.")

        self.dfFeats = featureDfs

    def getFeatures(self) -> Dict[str, pd.DataFrame]:
        if not self.dfFeats:
            raise RuntimeError("Features not yet built. Call buildFeatureDfs() first.")
        return self.dfFeats

    def get_norm_stats(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        return self.norm_stats
