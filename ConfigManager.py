import argparse
from pathlib import Path  # optional but handy

class ConfigManager:
    @staticmethod
    def parseArgs():
        """
        Defines configuration across raw data, models, and graph construction
        """
        parser = argparse.ArgumentParser("Spatio-Temporal Forecasting")

        # ====================================
        # === Raw Data
        parser.add_argument("--mode", choices=["csv", "yfinance", "av"], default="yfinance")
        parser.add_argument("--data_dir", default="./data")
        parser.add_argument("--av_key", default=None)

        # ====================================
        # === Stock Tickers
        ALL_TICKERS = [
            "NVDA", "MSFT", "AAPL", "AMZN", "META",
            "AVGO", "GOOGL", "GOOG", "BRK-B", "TSLA",
            "JPM", "WMT", "ORCL", "LLY", "V",
            "MA", "NFLX", "XOM", "COST", "JNJ",
            "PLTR", "ABBV", "HD", "BAC", "AMD",
            "PG", "UNH", "GE", "CVX", "KO",
            "CSCO", "IBM", "WFC", "TMUS", "MS",
            "CRM", "AXP", "PM", "CAT", "RTX",
            "GS", "ABT", "MCD", "MRK", "MU",
            "TMO", "LIN", "PEP", "DIS", "NOW"
        ]
        test_limit = 20
        tickers_for_test = [ALL_TICKERS[i] for i in range(min(test_limit, len(ALL_TICKERS)))]

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
        lr = 1e-3

        # === Regularisation
        parser.add_argument("--weight_decay", type=float, default=1e-4,
                            help="Global L2 weight decay")
        
        # ====================================
        # === LSTM Training
        parser.add_argument("--lstm_lr", type=float, default=lr, help="Learning rate for LSTM")
        parser.add_argument("--lstm_epochs", type=int, default=epoch_count, help="Number of LSTM training epochs")
        parser.add_argument("--lstm_save", default="misc/lstm_best.pth",  # use forward slashes (portable)
                            help="Path to save LSTM model")

        # ====================================
        # === LSTM Architecture
        parser.add_argument("--lstm_hidden", type=int, default=64, help="Hidden dimension of LSTM layers")
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
        parser.add_argument("--tcn_channels", type=int, default=32, help="Channels in TCN layer")
        parser.add_argument("--tcn_kernel_size", type=int, default=2, help="Kernel size in TCN layer")
        parser.add_argument("--gcn_hidden", type=int, default=32, help="Hidden dimension in GCN")
        
        parser.add_argument(
            "--graph_ablation",
            type=str,
            default="none",
            choices=["none", "identity", "empty"],
            help="Graph ablation for STGNN: none=use learned sparse graph; "
                "identity=self-loops only; empty=no edges."
        )


        # ====================================
        # === Experiment Running
        parser.add_argument("--results_dir", default="./results", help="Directory to save experiment results")
        parser.add_argument("--experiment_name", default="benchmark_run", help="Experiment name for CSV output")

        # ====================================
        # === Result Saving === Toggle to False for main interactive build
        parser.add_argument("--save_results", action="store_true", default=False, help="Save metrics to CSV")

        return parser.parse_args([])
