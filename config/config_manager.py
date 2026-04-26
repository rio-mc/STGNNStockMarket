import argparse
import json
from pathlib import Path


class ConfigManager:
    @staticmethod
    def parseArgs():
        """
        Defines configuration across run mode, dataset selection, model choice,
        training, graph construction, and ablation controls.

        This parser now supports both GUI execution and headless experiment runs.
        """
        parser = argparse.ArgumentParser("Spatio-Temporal Forecasting")

        # ====================================
        # === Run Mode / Execution Control
        parser.add_argument(
            "--run_mode",
            type=str,
            default="gui",
            choices=["gui", "headless"],
            help="Execution mode: launch GUI or run a single headless experiment"
        )
        parser.add_argument(
            "--target_stock",
            type=str,
            default=None,
            help="Ticker to run in headless mode. Defaults to the first available ticker."
        )
        parser.add_argument(
            "--prediction_window",
            type=str,
            default="1d",
            choices=["1d", "2d", "5d", "1w"],
            help="Prediction horizon label used by the pipeline"
        )

        # ====================================
        # === Result Saving
        parser.add_argument(
            "--save_results",
            action="store_true",
            default=False,
            help="Save metrics to CSV"
        )
        parser.add_argument(
            "--results_dir",
            type=str,
            default="./results",
            help="Directory to save experiment results"
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="benchmark_run",
            help="Experiment name for saved outputs"
        )
        parser.add_argument(
            "--config_file",
            type=str,
            default=None,
            help="Path to JSON config file"
        )

        # ====================================
        # === Dataset / Universe Selection
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="sp500",
            choices=["sp500", "nasdaq100", "custom"],
            help="Dataset / asset universe to use"
        )
        parser.add_argument(
            "--top_n",
            type=int,
            default=50,
            help="Top N assets to select where applicable"
        )
        parser.add_argument(
            "--universe_as_of",
            type=str,
            default=None,
            help="As-of date for universe selection metadata"
        )
        parser.add_argument(
            "--custom_tickers",
            nargs="+",
            default=None,
            help="Explicit ticker list when dataset_name=custom"
        )
        parser.add_argument(
            "--dataset_dir",
            type=str,
            default="./data",
            help="Reserved for future local dataset support"
        )
        parser.add_argument(
            "--date_start",
            type=str,
            default=None,
            help="Inclusive start date for price history"
        )
        parser.add_argument(
            "--date_end",
            type=str,
            default=None,
            help="Inclusive end date for price history"
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            help="Yahoo Finance sampling interval, e.g. 1h, 30m, 1d"
        )

        # ====================================
        # === Legacy compatibility
        parser.add_argument(
            "--mode",
            choices=["csv", "yfinance", "av"],
            default="yfinance",
            help="Legacy argument. Retained temporarily for compatibility."
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="./data",
            help="Legacy argument. Retained temporarily for compatibility."
        )
        parser.add_argument(
            "--av_key",
            type=str,
            default=None,
            help="Legacy Alpha Vantage key. Not used in Yahoo-only path."
        )

        # ====================================
        # === Model Selection
        parser.add_argument(
            "--model",
            type=str,
            choices=[
                "lstm",
                "gru",
                "panel_gru",
                "panel_lstm",
                "gcn",
                "nnconv",
                "graphsage",
                "stgnn",
            ],
            default="lstm",
            help="Model to run",
        )

        parser.add_argument(
            "--graph_model",
            type=str,
            default="gcn",
            choices=["gcn", "graphsage", "gat", "nnconv"],
            help="Graph operator/backend used by STGNN-style graph models."
        )
        parser.add_argument(
            "--decision-threshold-policy",
            type=str,
            default="fixed",
            choices=["fixed", "macro_f1_dense"],
            help="Threshold policy for final decision"
        )
        
        # ====================================
        # === Experiment Controls
        parser.add_argument(
            "--num_seeds",
            type=int,
            default=3,
            help="Number of random seeds to run per configuration"
        )
        parser.add_argument(
            "--base_seed",
            type=int,
            default=42,
            help="Base RNG seed for sweeps"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Single-run RNG seed"
        )
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="Use deterministic CUDA kernels and raise on non-deterministic ops."
        )
        parser.add_argument(
            "--no_deterministic",
            dest="deterministic",
            action="store_false"
        )
        parser.set_defaults(deterministic=False)

        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="Number of worker processes for DataLoaders"
        )

        # ====================================
        # === Sequence Length
        parser.add_argument(
            "--seq_len",
            type=int,
            default=10,
            help="Temporal lookback window"
        )

        # ====================================
        # === Graph Construction
        parser.add_argument(
            "--k",
            type=int,
            default=3,
            help="Number of edges to retain per node (KNN)"
        )
        parser.add_argument(
            "--graph_mode",
            type=str,
            default="knn_mst",
            choices=["knn", "mst", "knn_mst"],
            help="Graph construction strategy"
        )
        parser.add_argument(
            "--graph_window",
            type=int,
            default=10,
            help="Rolling window size used when aggregating graph features"
        )

        # ====================================
        # === Graph Rewiring
        parser.add_argument(
            "--rewiring",
            action="store_true",
            help="Enable post-training graph rewiring after STGNN training."
        )
        parser.add_argument(
            "--no_rewiring",
            dest="rewiring",
            action="store_false",
            help="Disable post-training graph rewiring."
        )
        parser.set_defaults(rewiring=False)

        # ====================================
        # === Generic Training
        parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
        parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")

        epoch_count = 50
        lr = 1e-4

        parser.add_argument(
            "--head_temperature",
            type=float,
            default=2.0,
            help="Post-hoc logit temperature. Keep 1.0 for raw probabilities."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-5,
            help="Global L2 weight decay"
        )

        # ====================================
        # === LSTM / GRU Training
        parser.add_argument("--lstm_lr", type=float, default=lr, help="LSTM/GRU learning rate")
        parser.add_argument("--lstm_epochs", type=int, default=epoch_count, help="LSTM/GRU epochs")
        parser.add_argument("--lstm_save", type=str, default="misc/lstm_best.pth", help="LSTM save path")

        # ====================================
        # === LSTM / GRU Architecture
        parser.add_argument("--lstm_hidden", type=int, default=16, help="LSTM/GRU hidden dimension")
        parser.add_argument("--lstm_layers", type=int, default=2, help="Number of recurrent layers")
        parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM/GRU")

        # ====================================
        # === STGNN Training
        parser.add_argument("--stgnn_lr", type=float, default=lr, help="STGNN learning rate")
        parser.add_argument("--stgnn_epochs", type=int, default=epoch_count, help="STGNN epochs")
        parser.add_argument("--stgnn_save", type=str, default="misc/stgnn_best.pth", help="STGNN save path")

        # ====================================
        # === STGNN Architecture
        parser.add_argument("--stgnn_blocks", type=int, default=2, help="Number of ST blocks")
        parser.add_argument("--tcn_channels", type=int, default=16, help="TCN channels")
        parser.add_argument("--tcn_kernel_size", type=int, default=2, help="TCN kernel size")
        parser.add_argument("--gcn_hidden", type=int, default=16, help="GCN hidden dimension")

        # ====================================
        # === Ablations
        parser.add_argument(
            "--graph_ablation",
            type=str,
            default="none",
            choices=["none", "identity", "empty"],
            help="Graph ablation mode"
        )
        parser.add_argument(
            "--ablate_feature",
            type=str,
            default="none",
            choices=["none", "return", "volatility", "momentum"],
            help="Remove one engineered feature everywhere"
        )
        parser.add_argument(
            "--graph_embed",
            type=str,
            default="pca",
            choices=["pca", "raw"],
            help="Graph scalar embedding mode"
        )

        # ====================================
        # === Shared Representation + Head
        parser.add_argument(
            "--rep_dim",
            type=int,
            default=128,
            help="Shared representation dimension"
        )
        parser.add_argument(
            "--head_hidden",
            type=int,
            default=128,
            help="Shared classifier head hidden width"
        )

        args = parser.parse_args()

        # ====================================
        # === Config file override
        if args.config_file is not None:
            config_path = Path(args.config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with config_path.open("r", encoding="utf-8") as f:
                config_dict = json.load(f)

            for key, value in config_dict.items():
                if not hasattr(args, key):
                    raise ValueError(f"Unknown config key in {config_path}: {key}")
                setattr(args, key, value)

        # ====================================
        # === Validation
        if args.dataset_name == "custom" and not args.custom_tickers:
            raise ValueError("dataset_name='custom' requires --custom_tickers")

        if args.run_mode == "headless" and args.target_stock is not None:
            args.target_stock = args.target_stock.strip().upper()

        if args.graph_window < 1:
            raise ValueError("--graph_window must be >= 1")

        if args.seq_len < 1:
            raise ValueError("--seq_len must be >= 1")

        if args.k < 0:
            raise ValueError("--k must be >= 0")

        return args