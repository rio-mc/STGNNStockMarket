import random
import re
import numpy as np
import torch
import networkx as nx
from pympler import asizeof
import pandas as pd
from torch.optim import AdamW

class Utils:

    def parse_window(window: str, bars_per_day: float) -> int:
        """
        Convert window string like '3d', '1w', '2mo' into number of trading bars,
        based on estimated bars per day (e.g. ~6.5 for hourly stock data).
        """
		# === STEP 1: Extract values from front-end ===
        # ------------------------------------
        m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]+)\s*", window)
        if not m:
            raise ValueError(f"Invalid window format: {window!r}")
        n, unit = int(m.group(1)), m.group(2).lower()

        # === STEP 2: Convert to hourly interval ===
        # ------------------------------------
        if unit in ("h", "hr", "hour", "hours"):
            return int(n)  # already in trading bars
        if unit in ("d", "day", "days"):
            return int(n * bars_per_day)
        if unit in ("w", "wk", "week", "weeks"):
            return int(n * 5 * bars_per_day)
        if unit in ("mo", "mth", "month", "months"):
            return int(n * 21 * bars_per_day)
        if unit in ("y", "yr", "year", "years"):
            return int(n * 252 * bars_per_day)

        raise ValueError(f"Unknown time unit: {unit}")
    
    def set_seed(seed: int, deterministic: bool = False) -> int:
        """
        Seed Python, NumPy, and PyTorch RNGs. Returns the seed used.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Optional determinism controls
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

        return seed
            
    def compute_pos_weight(dataset):
        # ====================================
		# === Helper to compute class weighting for loss computation

        # === STEP 1: Label Counting ===
        # ------------------------------------
        labels = []
        for idx, data in enumerate(dataset):
            try:
                if hasattr(data, 'y'):  # PyG Data object (STGNN)
                    y = data.y
                elif isinstance(data, (tuple, list)) and len(data) >= 2:
                    y = data[1]
                else:
                    raise ValueError(f"Unsupported data format at index {idx}: {type(data)}")
                
                if isinstance(y, torch.Tensor):
                    labels.append(int(y.item()))
                else:
                    raise TypeError(f"Expected tensor for y at index {idx}, got {type(y)}")
            
            except Exception as e:
                print(f"[compute_pos_weight] Skipping index {idx} due to error: {e}")
                continue
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos

        # === STEP 2: Avoid division by 0 ===
        # ------------------------------------
        if num_pos == 0:
            print("[POS_WEIGHT WARNING] No positive samples. Returning default weight of 1.0")
            return torch.tensor([1.0])
        elif num_neg == 0:
            print("[POS_WEIGHT WARNING] No negative samples. Returning high penalty of 100.0")
            return torch.tensor([100.0])

        pos_weight = num_neg / (num_pos + 1e-6)
        return torch.tensor([pos_weight])
    
    def log_graph_memory(G: nx.Graph, coords: np.ndarray, edge_index: torch.Tensor, tag="Graph"):
        # ====================================
		# === Helper to log graph memory consumption
        G_bytes = asizeof.asizeof(G)
        coords_bytes = coords.nbytes
        ei_bytes = edge_index.element_size() * edge_index.nelement()
        print(f"[Memory] {tag} NetworkX Size: {G_bytes / 1024:.2f} KB")
        print(f"[Memory] {tag} Coords Memory: {coords_bytes / 1024:.2f} KB")
        print(f"[Memory] {tag} Edge Index Memory: {ei_bytes / 1024:.2f} KB")

    def log_gpu_memory(tag: str = ""):
        # ===================================
		# === Helper to log GPU memory consumption
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            peak = torch.cuda.max_memory_allocated()
            print(f"[Memory][{tag}] GPU Allocated: {allocated / 1024**2:.2f} MB")
            print(f"[Memory][{tag}] GPU Reserved: {reserved / 1024**2:.2f} MB")
            print(f"[Memory][{tag}] GPU Peak: {peak / 1024**2:.2f} MB")

    def map_seq_len_from_horizon(horizon: int, bars_per_day: float) -> int:
        # ====================================
		# === Helper to provide history per day of prediction horizon
        horizon_days = horizon / bars_per_day
        seq_days = 3 * horizon_days
        return int(seq_days * bars_per_day)
    
    def sanity_check_features(feats: dict, raw: dict, window: int = 5):
        """
        Ensure that engineered features do not 'peek' into future data.
        Compares rolling stats with raw data over shifted windows.
        """
        for ticker in feats:
            fdf = feats[ticker]
            rdf = raw[ticker]

            if len(fdf) == 0 or len(rdf) == 0:
                print(f"[sanity_check_features] Empty dataframe for {ticker}")
                continue

            # Align both DataFrames to a shared index
            common_idx = fdf.index.intersection(rdf.index)

            if len(common_idx) < window:
                print(f"[sanity_check_features] Not enough overlapping data for {ticker}")
                continue

            fdf_aligned = fdf.loc[common_idx]
            rdf_aligned = rdf.loc[common_idx]

            expected_vol = rdf_aligned["close"].rolling(window=window).std().shift(1).dropna()
            if "_volatility_raw" in fdf.columns:
                actual_vol = fdf["_volatility_raw"].dropna()
            else:
                actual_vol = fdf["volatility"].dropna()

            # Align lengths
            min_len = min(len(expected_vol), len(actual_vol))
            expected_vol = expected_vol[-min_len:]
            actual_vol = actual_vol[-min_len:]

            diff = (expected_vol.values - actual_vol.values)
            max_abs_error = np.max(np.abs(diff))
            mean_abs_error = np.mean(np.abs(diff))

            print(f"[sanity_check_features] {ticker} volatility check: mean abs error = {mean_abs_error:.6f}, max abs error = {max_abs_error:.6f}")
            if max_abs_error > 1e-3:
                raise ValueError(f"[sanity_check_features] Feature leakage suspected in {ticker}: volatility mismatch exceeds tolerance")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def apply_graph_ablation(edge_index: torch.Tensor, num_nodes: int, mode: str) -> torch.Tensor:
        """
        Returns an edge_index shaped [2, E] for a chosen ablation mode.
        - none: keep as-is
        - identity: only self-loops (i,i)
        - empty: no edges
        """
        mode = (mode or "none").lower()
        if mode == "none":
            return edge_index

        if mode == "empty":
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        if mode == "identity":
            idx = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
            return torch.stack([idx, idx], dim=0)

        raise ValueError(f"Unknown graph ablation mode: {mode}")
    
    def load_ticker_to_sector(csv_path: str) -> dict:
        df = pd.read_csv(csv_path)

        assert "ticker" in df.columns and "sector" in df.columns, \
            "CSV must contain 'ticker' and 'sector' columns"

        return {
            str(row["ticker"]).upper(): str(row["sector"])
            for _, row in df.iterrows()
        }
    
    def make_adamw(model, lr, weight_decay):
        """
        Paper-defensible optimiser:
        - AdamW (decoupled weight decay)
        - exclude bias and norm parameters from weight decay (common practice)
        """
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            n = name.lower()
            if n.endswith("bias") or "norm" in n or "layernorm" in n:
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = []
        if decay:
            param_groups.append({"params": decay, "weight_decay": float(weight_decay)})
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0})

        return AdamW(param_groups, lr=float(lr))