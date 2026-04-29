"""Canonical headless experiment runner."""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from config.config_manager import ConfigManager
from core.experiment_runner import ExperimentResult, ExperimentRunner
from core.pipeline import Pipeline
from data.dataset_registry import DatasetRegistry
from data.universe_service import UniverseService


def setup_logging(args):
    logger = logging.getLogger("run_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def set_all_seeds(seed: int, deterministic: bool = False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_universe(args):
    if hasattr(args, "tickers") and args.tickers:
        return list(args.tickers)
    universe_service = UniverseService()
    universe_id = getattr(args, "universe_id", None) or getattr(args, "dataset_name", "sp500")
    universe_def = universe_service.resolve_definition(
        universe_id=universe_id,
        universe_provider=getattr(args, "universe_provider", "static_csv"),
        top_n=getattr(args, "top_n", 100),
        as_of_date=getattr(args, "universe_as_of", None),
        custom_tickers=getattr(args, "custom_tickers", None),
    )
    args.universe_id = universe_def.universe_id
    return list(universe_def.tickers)


def load_data(args, logger):
    args.tickers = resolve_universe(args)
    provider = str(getattr(args, "price_provider", "yahoo")).strip().lower()
    logger.info("[Data] Loading provider=%s tickers=%d", provider, len(args.tickers))
    if provider in {"yahoo", "yfinance"}:
        dataset = DatasetRegistry.load(
            "yahoo",
            tickers=args.tickers,
            period="729d",
            interval=getattr(args, "interval", "1d"),
            start_date=getattr(args, "date_start", None),
            end_date=getattr(args, "date_end", None),
        )
    elif provider == "csv":
        dataset = DatasetRegistry.load("csv", data_dir=getattr(args, "dataset_dir", "./data"), tickers=args.tickers)
    else:
        raise ValueError(f"Unsupported price_provider: {provider}")
    result = dataset.load()
    args.tickers = list(result.tickers)
    logger.info("[Data] Loaded %d tickers, dropped %d", len(result.tickers), len(result.metadata.get("dropped_tickers", [])) if result.metadata else 0)
    return result.data, result.tickers, result.index, result.metadata


def resolve_feature_cols(args):
    engineered = ["return", "volatility", "momentum"]
    ablate = getattr(args, "ablate_feature", "none")
    if ablate != "none":
        engineered = [f for f in engineered if f != ablate]
    return ["close"] + engineered


def run_experiment(args) -> ExperimentResult:
    logger = setup_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_mode = "headless"
    args.graph_window = int(args.seq_len)

    set_all_seeds(seed=int(getattr(args, "seed", 42)), deterministic=bool(getattr(args, "deterministic", False)))

    logger.info("=" * 60)
    logger.info("EXPERIMENT START")
    logger.info("model=%s dataset=%s target=%s device=%s", args.model, args.dataset_name, getattr(args, "target_stock", None), device)
    logger.info("=" * 60)

    raw_data, tickers, index, metadata = load_data(args, logger)
    if not raw_data:
        raise RuntimeError("No usable data loaded.")

    if getattr(args, "target_stock", None):
        target_stock = str(args.target_stock).strip().upper()
        target_source = "cli"
    else:
        target_stock = tickers[0] if tickers else None
        target_source = "default_first_loaded_ticker"
    if not target_stock:
        raise RuntimeError("No target stock available.")

    target_stock = str(target_stock).strip().upper()
    args.target_stock = target_stock
    args.target_source = target_source

    if target_stock not in raw_data:
        raise ValueError(f"Target stock '{target_stock}' not in loaded data. Available sample: {list(raw_data)[:10]}")

    raw_feature_cols = resolve_feature_cols(args)
    logger.info("[Features] %s", raw_feature_cols)
    logger.info("[Experiment] Target stock: %s", target_stock)
    logger.info("[Experiment] seq_len=%d graph_window=%d", int(args.seq_len), int(args.graph_window))

    pipeline = Pipeline(args, raw_feature_dfs=raw_data)
    prediction_window = getattr(args, "prediction_window", "1d")
    state = pipeline.run(target_stock, prediction_window, stop_event=None)

    args.graph_window = int(state.get("graph_window", args.seq_len))
    state["args"] = args
    state["target_ticker"] = target_stock
    state["target_stock"] = target_stock
    state["target_source"] = target_source
    state["raw_feature_cols"] = raw_feature_cols
    state["raw_feature_dfs"] = raw_data
    state["dataset_metadata"] = metadata or {}

    runner = ExperimentRunner(app=None, args=args, device=device)
    result = runner.run(model_name=args.model, stock=target_stock, state=state, evaluator=None, stop_event=None)

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("model=%s direction=%s confidence=%.2f training_time=%.2fs", result.model_name, result.direction, result.confidence, result.training_time_sec)
    logger.info("=" * 60)
    return result


def _metric_items(metrics: Any):
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return dict(metrics)
    if hasattr(metrics, "__dict__"):
        return dict(metrics.__dict__)
    return {"value": str(metrics)}


def save_results(result: ExperimentResult, args) -> str:
    results_dir = Path(getattr(args, "results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{getattr(args, 'model', result.model_name).lower()}_{timestamp}_{getattr(args, 'seed', 42)}"
    csv_path = results_dir / "results.csv"

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model": result.model_name,
        "dataset": getattr(args, "dataset_name", "unknown"),
        "target_stock": getattr(args, "target_stock", None),
        "target_source": getattr(args, "target_source", None),
        "prediction_window": getattr(args, "prediction_window", "1d"),
        "seq_len": int(getattr(args, "seq_len", 0)),
        "graph_window": int(getattr(args, "graph_window", getattr(args, "seq_len", 0))),
        "seed": getattr(args, "seed", 42),
        "direction": result.direction,
        "confidence": result.confidence,
        "training_time_sec": result.training_time_sec,
        "k": getattr(args, "k", 0),
        "graph_mode": getattr(args, "graph_mode", "unknown"),
        "graph_embed": getattr(args, "graph_embed", "unknown"),
        "graph_ablation": getattr(args, "graph_ablation", "none"),
        "ablate_feature": getattr(args, "ablate_feature", "none"),
    }
    for key, value in _metric_items(result.metrics).items():
        row[f"metric_{key}"] = value

    file_exists = csv_path.exists()
    existing_fields = []
    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_fields = next(reader, [])
    fieldnames = list(dict.fromkeys(existing_fields + list(row.keys())))

    if file_exists and set(row.keys()) - set(existing_fields):
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")
    return run_id


def main():
    args = ConfigManager.parseArgs()
    if not getattr(args, "model", None):
        args.model = "lstm"
    result = run_experiment(args)
    run_id = save_results(result, args)
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULT")
    print("=" * 60)
    print(f"Run ID       : {run_id}")
    print(f"Model        : {result.model_name}")
    print(f"Target Stock : {getattr(args, 'target_stock', None)}")
    print(f"Seq Len      : {getattr(args, 'seq_len', None)}")
    print(f"Graph Window : {getattr(args, 'graph_window', None)}")
    print(f"Direction    : {result.direction}")
    print(f"Confidence   : {result.confidence:.2f}%")
    print(f"Training     : {result.training_time_sec:.2f}s")
    print("=" * 60)
    return result


if __name__ == "__main__":
    main()
