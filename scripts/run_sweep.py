"""Canonical headless parameter sweep runner."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from config.config_manager import ConfigManager
from run_experiment import run_experiment, save_results, setup_logging


def _parse_sweep_overrides(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--k_values", nargs="+", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--target_stocks", nargs="+", default=None)
    parser.add_argument("--graph_modes", nargs="+", default=None)
    parser.add_argument("--ablate_features", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    sweep_args, remaining = parser.parse_known_args(argv)
    return sweep_args, remaining


def parse_sweep_args(argv=None):
    sweep_args, remaining = _parse_sweep_overrides(argv)

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + remaining
        args = ConfigManager.parseArgs()
    finally:
        sys.argv = original_argv

    args.models = sweep_args.models or [getattr(args, "model", "lstm")]
    args.k_values = sweep_args.k_values or [int(getattr(args, "k", 3))]
    args.num_seeds = int(sweep_args.num_seeds or getattr(args, "num_seeds", 3))
    args.target_stocks = sweep_args.target_stocks or [getattr(args, "target_stock", None)]
    args.graph_modes = sweep_args.graph_modes or [getattr(args, "graph_mode", "knn_mst")]
    args.ablate_features = sweep_args.ablate_features or [getattr(args, "ablate_feature", "none")]
    args.resume = bool(sweep_args.resume)
    args.run_mode = "headless"
    args.graph_window = int(args.seq_len)
    return args


def _completed_keys(results_dir: Path):
    completed = set()
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return completed

    import csv
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed.add((
                row.get("model", "").lower(),
                row.get("target_stock", ""),
                row.get("seed", ""),
                row.get("k", ""),
                row.get("graph_mode", ""),
                row.get("ablate_feature", ""),
            ))
    return completed


def iter_configs(args):
    base_seed = int(getattr(args, "seed", 42))
    seeds = list(range(base_seed, base_seed + int(args.num_seeds)))
    for model, k, seed, target, graph_mode, ablate in itertools.product(
        args.models,
        args.k_values,
        seeds,
        args.target_stocks,
        args.graph_modes,
        args.ablate_features,
    ):
        yield {
            "model": model,
            "k": int(k),
            "seed": int(seed),
            "target_stock": target,
            "graph_mode": graph_mode,
            "ablate_feature": ablate,
        }


def run_sweep(args):
    logger = setup_logging(args)
    results_dir = Path(getattr(args, "results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = list(iter_configs(args))
    completed = _completed_keys(results_dir) if getattr(args, "resume", False) else set()

    logger.info("=" * 60)
    logger.info("SWEEP START")
    logger.info("configs=%d models=%s k_values=%s seeds=%d", len(configs), args.models, args.k_values, args.num_seeds)
    logger.info("graph_window is enforced to seq_len=%s", args.seq_len)
    logger.info("=" * 60)

    results = []
    for idx, config in enumerate(configs, start=1):
        for key, value in config.items():
            setattr(args, key, value)
        args.graph_window = int(args.seq_len)

        resume_key = (
            str(config["model"]).lower(),
            str(config.get("target_stock") or ""),
            str(config["seed"]),
            str(config["k"]),
            str(config["graph_mode"]),
            str(config["ablate_feature"]),
        )
        if resume_key in completed:
            logger.info("[%d/%d] skipping completed %s", idx, len(configs), resume_key)
            continue

        logger.info("[%d/%d] running %s", idx, len(configs), config)
        try:
            result = run_experiment(args)
            save_results(result, args)
            results.append(result)
        except Exception as exc:
            logger.exception("[%d/%d] failed %s: %s", idx, len(configs), config, exc)

    logger.info("=" * 60)
    logger.info("SWEEP COMPLETE completed=%d failed=%d", len(results), len(configs) - len(results))
    logger.info("=" * 60)
    return results


def main(argv=None):
    args = parse_sweep_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    main(sys.argv[1:])
