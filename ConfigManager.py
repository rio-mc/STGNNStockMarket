import argparse
from pathlib import Path
import pandas as pd

class ConfigManager:
    @staticmethod
    def parseArgs():
        """
        Defines configuration across raw data, models, and graph construction
        """
        parser = argparse.ArgumentParser("Spatio-Temporal Forecasting")

        # ====================================
        # === Result Saving === Toggle to False for main interactive build
        parser.add_argument("--save_results", action="store_true", default=True, help="Save metrics to CSV")
        
        # ====================================
        # === Experiment Running
        parser.add_argument("--results_dir", default="./results", help="Directory to save experiment results")
        parser.add_argument("--experiment_name", default="benchmark_run", help="Experiment name for CSV output")
        parser.add_argument("--num-seeds", type=int, default=3,
                    help="Number of random seeds to run per (ticker, config).")
        
        # ====================================
        # === Raw Data
        parser.add_argument("--mode", choices=["csv", "yfinance", "av"], default="yfinance")
        parser.add_argument("--data_dir", default="./data")
        parser.add_argument("--av_key", default=None)

        # ====================================
        # === Stock Tickers (from sector CSV)
        sector_map_path = "tickers.csv"

        sector_df = pd.read_csv(sector_map_path)

        # canonical ticker universe
        ALL_TICKERS = (
            sector_df["ticker"]
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        # optional deterministic ordering
        ALL_TICKERS = sorted(ALL_TICKERS)

        # test subset
        # test_limit = 20
        test_limit = len(ALL_TICKERS)
        tickers_for_test = ALL_TICKERS[:min(test_limit, len(ALL_TICKERS))]
        
        parser.add_argument(
            "--tickers", nargs="+", default=tickers_for_test,
            help="List of stock tickers to include in analysis."
        )

        # ====================================
        # === Determinism & seeds
        parser.add_argument("--deterministic", action="store_true",
                            help="Use deterministic CUDA kernels and raise on non-deterministic ops.")
        parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
        parser.set_defaults(deterministic=False)
        parser.add_argument("--base-seed", type=int, default=42,
                            help="Base RNG seed. Experiments increment this per repetition.")

        parser.add_argument("--num-workers", type=int, default=0,
                            help="Number of worker processes for DataLoaders.")
        
        # ====================================
        # === Sampling interval
        parser.add_argument("--interval", default="1h", help="Data sampling interval (e.g., 1h, 30m, 1d)")

        # ====================================
        # === Sequence Length
        parser.add_argument(
            "--seq-len", type=int, default=10,
            help="Number of past timesteps to use as temporal lookback window."
        )

        # ====================================
        # === Graph Construction
        parser.add_argument("--max_k", type=int, default=5, help="Number of edges to retain per node (KNN)")
        
        # ====================================
        # === Graph Rewiring
        parser.add_argument("--rewiring", action="store_true",
                            help="Enable post-training graph rewiring after STGNN training.")
        parser.add_argument("--no-rewiring", dest="rewiring", action="store_false",
                            help="Disable post-training graph rewiring.")
        parser.set_defaults(rewiring=False)

        # ====================================
        # === Generic Training
        parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training")
        parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
        epoch_count = 50
        lr = 1e-4
        parser.add_argument("--head_temperature", type=float, default=2.0,
                            help="Post-hoc logit temperature (T>1 softens probabilities). Keep 1.0 for raw.")

        # === Regularisation
        parser.add_argument("--weight_decay", type=float, default=1e-5,
                            help="Global L2 weight decay")
        
        # ====================================
        # === LSTM Training
        parser.add_argument("--lstm_lr", type=float, default=lr, help="Learning rate for LSTM")
        parser.add_argument("--lstm_epochs", type=int, default=epoch_count, help="Number of LSTM training epochs")
        parser.add_argument("--lstm_save", default="misc/lstm_best.pth",  # use forward slashes (portable)
                            help="Path to save LSTM model")

        # ====================================
        # === LSTM Architecture
        parser.add_argument("--lstm_hidden", type=int, default=16, help="Hidden dimension of LSTM layers")
        parser.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers")
        parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")

        # ====================================
        # === STGNN Training
        parser.add_argument("--stgnn_lr", type=float, default=lr, help="Learning rate for STGNN")
        parser.add_argument("--stgnn_epochs", type=int, default=epoch_count, help="Number of STGNN training epochs")
        parser.add_argument("--stgnn_save", default="misc/stgnn_best.pth",
                            help="Path to save STGNN model")

        # ====================================
        # === STGNN Architecture
        parser.add_argument("--stgnn_blocks", type=int, default=2, help="Number of ST Blocks")
        parser.add_argument("--tcn_channels", type=int, default=16, help="Channels in TCN layer")
        parser.add_argument("--tcn_kernel_size", type=int, default=2, help="Kernel size in TCN layer")
        parser.add_argument("--gcn_hidden", type=int, default=16, help="Hidden dimension in GCN")
        
        # ====================================
        # === Feature Ablations
        parser.add_argument(
            "--graph_ablation",
            type=str,
            default="none",
            choices=["none", "identity"],
            help="Graph ablation for STGNN: none=use learned sparse graph; "
                "identity=self-loops only."
        )

        parser.add_argument(
            "--ablate_feature",
            type=str,
            default="none",
            choices=["none", "return", "volatility", "momentum"],
            help="Ablation: remove exactly one engineered feature everywhere (node inputs + graph scalars)."
        )

        # === PCA ablation for graph embedding ===
        parser.add_argument(
            "--graph_embed",
            type=str,
            default="pca",
            choices=["pca", "raw"],
            help="Graph scalar embedding: pca=StandardScaler+PCA; raw=StandardScaler only (no PCA)."
        )

        # ====================================
        # === Shared Representation + Head (ALL MODELS)
        parser.add_argument("--rep_dim", type=int, default=128,
                            help="Common representation dim fed into the shared classifier head")
        parser.add_argument("--head_hidden", type=int, default=128,
                            help="Hidden width inside the shared classifier head (same for all models)")

        return parser.parse_args([])
