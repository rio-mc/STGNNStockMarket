from typing import Dict
import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(
        self,
        rawData: Dict[str, pd.DataFrame],
        rollingVolWindow: int = 5
    ):
        self.rawData = rawData
        self.rollingVolWindow = rollingVolWindow
        self.featureCols = ['open', 'high', 'low', 'close', 'volume',
                            'return', 'volatility', 'momentum']
        self.dfFeats = {}

    def buildFeatureDfs(self) -> None:
        """
        Transforms raw price DataFrames into engineered feature sets.

        Features produced per ticker:
        - open, high, low, close, volume
        - return, volatility, momentum
        """
		# ====================================
		# === Clear feature Dict
        featureDfs: Dict[str, pd.DataFrame] = {}

        # === STEP 1: Per-stock Feature Creation ===
        # ------------------------------------
        for tkr, df in self.rawData.items():
            # ====================================
		    # === Skip invalid tickers
            if df.empty or "close" not in df.columns:
                print(f"[Warning] Skipping '{tkr}' — no usable 'close' column.")
                continue
            dfFeat = df.copy().astype(np.float32)

            # ====================================
		    # === Ensure essential columns are present before proceeding
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in dfFeat.columns for col in required_cols):
                print(f"[Warning] Skipping '{tkr}' — missing essential price columns.")
                continue

		    # === STEP 2: Feature Engineering ===
            # ------------------------------------
            dfFeat["return"]     = dfFeat["close"].pct_change()
            dfFeat["volatility"] = dfFeat["close"].rolling(self.rollingVolWindow).std().shift(1)
            dfFeat["momentum"]   = dfFeat["close"].pct_change(periods=self.rollingVolWindow)

            #   1. Drop any rows with NaNs before normalisation
            dfFeat.dropna(inplace=True)
            # print("[DEBUG] Raw last row before normalisation:\n", dfFeat.iloc[-1][["return", "volatility", "momentum"]])

            # === STEP 3: Normalisation ===
            # ------------------------------------
            normalise_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'volatility', 'momentum']
            dfFeat[normalise_cols] = (
                dfFeat[normalise_cols] - dfFeat[normalise_cols].mean()
            ) / (dfFeat[normalise_cols].std(ddof=0) + 1e-8)
            dfFeat = dfFeat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # === STEP 3: Ensure Consistent Outputs ===
            # ------------------------------------
            self.featureCols = normalise_cols
            featureDfs[tkr] = dfFeat

        if not featureDfs:
            raise ValueError("No valid tickers with feature data found.")
        
        # === STEP 4: Feature DataFrame Creation ===
        # ------------------------------------
        self.dfFeats = featureDfs

    def getFeatures(self):
        # ====================================
		# === Helper to pass features
        if not self.dfFeats:
            raise RuntimeError("Features not yet built. Call buildFeatureDfs() first.")
        return self.dfFeats