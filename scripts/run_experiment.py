"""
Headless Experiment Runner

Runs a single experiment without GUI dependency.
Uses the new Pipeline and ExperimentRunner classes for clean separation.
"""

import logging
import os
import sys
from datetime import datetime

import torch

# Set up environment before imports
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from config.config_manager import ConfigManager
from core.experiment_runner import ExperimentRunner, ExperimentResult
from core.pipeline import Pipeline
from data.dataset_registry import DatasetRegistry
from data.universe_service import UniverseService
from models.registry import ModelRegistry


def setup_logging(args):
    """Configure logging for headless execution."""
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
    """
    Load data using DatasetRegistry.
    
    Returns:
        raw_feature_dfs: Dict[str, pd.DataFrame] - price data per ticker
        tickers: List[str] - valid ticker symbols
    """
    logger.info("[Data] Loading dataset: provider=%s", args.price_provider)
    
    # Use DatasetRegistry to load data
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
    
    return result.data, result.tickers, result.index, result.metadata


def run_experiment(args):
    """
    Run a single experiment with the specified model.
    
    Args:
        args: argparse namespace with configuration
        
    Returns:
        ExperimentResult with predictions and metrics
    """
    logger = setup_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT START")
    logger.info("Model: %s", args.model)
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Target: %s", args.target_stock)
    logger.info("Device: %s", device)
    logger.info("=" * 60)
    
    # Set seeds for reproducibility
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load data
    raw_data, tickers, index, metadata = load_data(args, logger)
    
    # Resolve target stock
    target_stock = args.target_stock
    if not target_stock:
        target_stock = tickers[0] if tickers else None
    
    if not target_stock:
        raise RuntimeError("No target stock available")
    
    if target_stock not in raw_data:
        raise ValueError(
            f"Target stock '{target_stock}' not in loaded data. "
            f"Available: {tickers[:10]}..."
        )
    
    logger.info("[Experiment] Target stock: %s", target_stock)
    
    # Determine feature columns based on ablation settings
    engineered = ["return", "volatility", "momentum"]
    if getattr(args, "ablate_feature", "none") != "none":
        engineered = [f for f in engineered if f != args.ablate_feature]
        logger.info("[Ablation] Removed feature: %s", args.ablate_feature)
    
    raw_feature_cols = ["close"] + engineered
    logger.info("[Features] Using: %s", raw_feature_cols)
    
    # Create pipeline state
    pipeline = Pipeline(args, raw_feature_dfs=raw_data)
    
    prediction_window = getattr(args, "prediction_window", "1d")
    state = pipeline.run(target_stock, prediction_window, stop_event=None)
    
    # Add feature columns to state for model runners
    state["raw_feature_cols"] = raw_feature_cols
    
    # Run the model
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
    if result.metrics:
        logger.info("Metrics: %s", result.metrics)
    logger.info("=" * 60)
    
    return result


def save_results(result: ExperimentResult, args):
    """Save experiment results to CSV."""
    import pandas as pd
    from pathlib import Path
    
    results_dir = Path(getattr(args, "results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model}_{timestamp}"
    
    # Save to experiments.csv
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
        "graph_ablation": getattr(args, "graph_ablation", "none"),
        "ablate_feature": getattr(args, "ablate_feature", "none"),
    }
    
    # Add metrics
    if result.metrics:
        for key, value in result.metrics.items():
            row[f"metric_{key}"] = value
    
    # Append to CSV
    import csv
    file_exists = csv_path.exists()
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"Results saved to: {csv_path}")
    
    return run_id


def main():
    """Main entry point for headless experiment."""
    args = ConfigManager.parseArgs()
    
    # Set defaults for missing args
    args.run_mode = "headless"
    
    # Validate required arguments
    if not hasattr(args, 'model') or not args.model:
        args.model = "lstm"
    
    if not hasattr(args, 'target_stock') or not args.target_stock:
        # Will be set after data loading
        pass
    
    # Resolve universe if needed
    if not hasattr(args, 'tickers') or not args.tickers:
        universe_service = UniverseService()
        universe_def = universe_service.resolve_definition(
            universe_id=getattr(args, 'universe_id', 'sp500'),
            universe_provider=getattr(args, 'universe_provider', 'static_csv'),
            top_n=getattr(args, 'top_n', 100),
        )
        args.tickers = list(universe_def.tickers)
    
    # Run experiment
    result = run_experiment(args)
    
    # Save results
    run_id = save_results(result, args)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULT")
    print("=" * 60)
    print(f"Run ID     : {run_id}")
    print(f"Model      : {result.model_name}")
    print(f"Direction  : {result.direction}")
    print(f"Confidence : {result.confidence:.2f}%")
    print(f"Training   : {result.training_time_sec:.2f}s")
    
    if result.metrics:
        print("\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
    
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()