"""
Headless Experiment Runner
"""

import logging
import os
import sys
from datetime import datetime

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from config.config_manager import ConfigManager
from core.experiment_runner import ExperimentRunner, ExperimentResult
from core.pipeline import Pipeline
from data.dataset_registry import DatasetRegistry
from data.universe_service import UniverseService


def setup_logging(args):
    logger = logging.getLogger("run_experiment")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_data(args, logger):
    price_provider = getattr(args, "price_provider", None)

    if price_provider is None:
        price_provider = getattr(args, "mode", "yfinance")

    if price_provider == "yfinance":
        price_provider = "yahoo"

    args.price_provider = price_provider

    logger.info("[Data] Loading dataset: provider=%s", args.price_provider)

    if args.price_provider in ("yahoo", "yfinance"):
        dataset = DatasetRegistry.load(
            "yahoo",
            tickers=args.tickers,
            period="729d",
            interval=getattr(args, "interval", "1d"),
        )
    elif args.price_provider == "csv":
        dataset = DatasetRegistry.load(
            "csv",
            data_dir=getattr(args, "dataset_dir", "./data"),
            tickers=args.tickers,
        )
    else:
        raise ValueError(f"Unsupported price_provider: {args.price_provider}")

    result = dataset.load()

    logger.info(
        "[Data] Loaded %d tickers, dropped %d",
        len(result.tickers),
        len(result.metadata.get("dropped_tickers", [])),
    )

    return result.data, result.tickers


def run_experiment(args):
    logger = setup_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("EXPERIMENT START")
    logger.info("Model: %s", args.model)
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Target: %s", args.target_stock)
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- FIX: restore data loading ---
    raw_data, tickers = load_data(args, logger)

    # --- Resolve target stock ---
    target_stock = args.target_stock
    if not target_stock:
        target_stock = tickers[0] if tickers else None

    if not target_stock:
        raise RuntimeError("No target stock available")

    target_stock = str(target_stock).strip().upper()

    # Normalise dataset keys
    raw_data = {str(k).strip().upper(): v for k, v in raw_data.items()}
    tickers = [str(t).strip().upper() for t in tickers]

    if target_stock not in raw_data:
        raise ValueError(
            f"Target stock '{target_stock}' not in loaded data. "
            f"Available: {tickers[:10]}..."
        )

    # --- CRITICAL FIX: persist ---
    args.target_stock = target_stock
    args.tickers = tickers

    logger.info("[Experiment] Target stock: %s", target_stock)

    engineered = ["return", "volatility", "momentum"]
    if getattr(args, "ablate_feature", "none") != "none":
        engineered = [f for f in engineered if f != args.ablate_feature]

    raw_feature_cols = ["close"] + engineered

    pipeline = Pipeline(args, raw_feature_dfs=raw_data)

    prediction_window = getattr(args, "prediction_window", "1d")
    state = pipeline.run(target_stock, prediction_window, stop_event=None)

    state["raw_feature_cols"] = raw_feature_cols

    runner = ExperimentRunner()
    runner.device = device
    runner.args = args

    result = runner.run(
        model_name=args.model,
        stock=target_stock,
        state=state,
        evaluator=None,
        stop_event=None,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("Model: %s", result.model_name)
    logger.info("Direction: %s", result.direction)
    logger.info("Confidence: %.2f%%", result.confidence)
    logger.info("Training time: %.2fs", result.training_time_sec)
    logger.info("=" * 60)

    return result


def metrics_to_dict(metrics):
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics
    if hasattr(metrics, "__dict__"):
        return vars(metrics)
    return {"metrics": str(metrics)}


def save_results(result: ExperimentResult, args):
    from pathlib import Path
    import csv

    results_dir = Path(getattr(args, "results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model}_{timestamp}"

    csv_path = results_dir / "experiments.csv"

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model": result.model_name,
        "dataset": getattr(args, "dataset_name", "unknown"),
        "target_stock": getattr(args, "target_stock", "unknown"),
        "prediction_window": getattr(args, "prediction_window", "1d"),
        "seed": getattr(args, "seed", 42),
        "direction": result.direction,
        "confidence": result.confidence,
        "training_time_sec": result.training_time_sec,
        "k": getattr(args, "k", 0),
        "graph_mode": getattr(args, "graph_mode", "unknown"),
        "graph_embed": getattr(args, "graph_embed", "unknown"),
        "ablate_feature": getattr(args, "ablate_feature", "none"),
    }

    # --- FIX: actually write metrics ---
    metrics = metrics_to_dict(result.metrics)
    for key, value in metrics.items():
        row[f"metric_{key}"] = value

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results saved to: {csv_path}")

    return run_id


def main():
    args = ConfigManager.parseArgs()
    args.run_mode = "headless"

    if not getattr(args, "model", None):
        args.model = "lstm"

    if not getattr(args, "tickers", None):
        universe_service = UniverseService()
        universe_def = universe_service.resolve_definition(
            universe_id=getattr(args, "universe_id", "sp500"),
            universe_provider=getattr(args, "universe_provider", "static_csv"),
            top_n=getattr(args, "top_n", 100),
        )
        args.tickers = list(universe_def.tickers)

    result = run_experiment(args)
    run_id = save_results(result, args)

    # --- your summary block (kept) ---
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULT")
    print("=" * 60)
    print(f"Run ID     : {run_id}")
    print(f"Model      : {result.model_name}")
    print(f"Direction  : {result.direction}")
    print(f"Confidence : {result.confidence:.2f}%")
    print(f"Training   : {result.training_time_sec:.2f}s")

    metrics = metrics_to_dict(result.metrics)
    if metrics:
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    print("=" * 60)

    return result


if __name__ == "__main__":
    main()