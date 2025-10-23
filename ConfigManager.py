import argparse

class ConfigManager:
    def parseArgs():
        """
        Defines configuration across raw data, models, and graph construction
        """
        parser = argparse.ArgumentParser("Spatio-Temporal Forecasting")

        # ====================================
		# === Raw Data : https://www.slickcharts.com/sp500 ===
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
            "--tickers",
            nargs="+",
            default=tickers_for_test,
            help="List of stock tickers to include in analysis."
        )

        # ====================================
        # === Sequence Length
        parser.add_argument(
            "--seq-len", type=int, default=3,
            help="Number of past timesteps to use as temporal lookback window."
        )

        # ====================================
		# === Generic Training
        parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size for training")
        parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
        epoch_count = 50
        lr=1e-4

        # ====================================
		# === LSTM Training
        parser.add_argument("--lstm_lr", type=float, default=lr, help="Learning rate for LSTM")
        parser.add_argument("--lstm_epochs", type=int, default=epoch_count, help="Number of LSTM training epochs")
        parser.add_argument("--lstm_save", default="misc\lstm_best.pth", help="Path to save LSTM model")

        # ====================================
		# === LSTM Architecture
        parser.add_argument("--lstm_hidden", type=int, default=24, help="Hidden dimension of LSTM layers")
        parser.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers")
        parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")

        # ====================================
		# === STGNN Training
        parser.add_argument("--stgnn_lr", type=float, default=lr, help="Learning rate for STGNN")
        parser.add_argument("--stgnn_epochs", type=int, default=epoch_count, help="Number of STGNN training epochs")
        parser.add_argument("--stgnn_save", default="misc\stgnn_best.pth", help="Path to save STGNN model")

        # ====================================
		# === STGNN Architecture        
        parser.add_argument("--stgnn_blocks", type=int, default=3, help="Number of ST Blocks")
        parser.add_argument("--tcn_channels", type=int, default=24, help="Channels in TCN layer")
        parser.add_argument("--tcn_kernel_size", type=int, default=1, help="Kernel size in TCN layer")
        parser.add_argument("--gcn_hidden", type=int, default=16, help="Hidden dimension in GCN")

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
		# === Sequence Length Horizon
        parser.add_argument("--lookback", type=int, default=20,
            help="Number of time steps to use as temporal lookback (sequence length).")

        return parser.parse_args()