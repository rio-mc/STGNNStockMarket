import copy
from threading import Thread
import threading
import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import logging
import time
import gc    
from torch_geometric.loader import DataLoader as GeoDataLoader
import os
import itertools

# cuBLAS determinism (PyTorch will raise without this when deterministic=True on CUDA >= 10.2)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Optional but recommended for stricter numerical reproducibility
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # stable device ordering

from FrontEnd           import FrontEnd
from RawDataHandler     import RawDataHandler
from FeatureExtractor   import FeatureExtractor
from TensorFactory      import TensorFactory
from GraphBuilder       import GraphBuilder
from Trainer            import Trainer
from LSTMClassifier     import LSTMClassifier
from GRUClassifier      import GRUClassifier
from STGNNClassifier    import STGNNClassifier
from ModelDataset       import LSTMDataset, STGNNDataset
from EvaluationMethods  import EvaluationMethods
from LoadingOverlay     import LoadingOverlay
from Utils              import Utils
from ConfigManager      import ConfigManager

class MainApp:
    """
    Orchestrates loading, preprocessing, graph-building, config, and model training.
    """
    def __init__(self):
        # === STEP 1: Set up configuration ===
        # ------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = ConfigManager.parseArgs()
        self.args.interval = '1h'

        self.args.base_seed = getattr(self.args, "base_seed", 42)
        self.args.deterministic = getattr(self.args, "deterministic", False)
        Utils.set_seed(self.args.base_seed, deterministic=self.args.deterministic)

        self.pipeline_running = False
        self.data_ready = threading.Event()

        # === STEP 2: Set up logging ===
        # ------------------------------------

        #   1. Logger initialisation
        self.logger = logging.getLogger("MainApp")
        self.logger.setLevel(logging.INFO)        
        
        #   2. Logger formatting
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # === STEP 3: Front-end initialisation ===
        # ------------------------------------

        #   1. Set and sort tickers (from config)
        self.args.tickers = sorted(self.args.tickers)
        self.frontendApp = FrontEnd(self.args.tickers)

        #   2. Show loading overlay
        avi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loading.avi")
        self.loader = LoadingOverlay(self.frontendApp.root, avi_path, delay=24)
    
    def _load_data_and_start_gui(self):
        # === STEP 1: Establish feature columns ===
        # ------------------------------------
        self.raw_feature_cols = ["close", "volatility", "momentum", "return"]

        def background_task():
            # === STEP 2: Gather all valid stock price histories ===
            # ------------------------------------

            #   1. Load valid and remove invalid histories
            self.priceHistory, raw_handler = self._load_price_history(self.args.tickers, return_handler=True)
            valid_tickers = raw_handler.listTickers()
            dropped_tickers = [t for t in self.args.tickers if t not in valid_tickers]
            if dropped_tickers:
                self.logger.warning(f"Dropped tickers (no valid data): {', '.join(dropped_tickers)}")
            self.args.tickers = valid_tickers

            # === STEP 3: Gather feature DataFrames ===
            # ------------------------------------
            self.raw_feature_dfs = {
                t: self.priceHistory[t] for t in valid_tickers
            }
            if not self.raw_feature_dfs:
                raise RuntimeError("No usable stock data was loaded.")

            # === STEP 4: Front-end binding ===
            # ------------------------------------

            #   1. Bind front-end and destroy loader
            self.frontendApp.bindMainApp(self)
            self.loader.trigger_fade_and_destroy()

            #   2. Create compute hook for front-end functionality
            self.frontendApp.bindTickerClick(self.frontendApp.on_ticker_click)
            self.frontendApp.setComputeCallback(self.startPipeline)

            self.data_ready.set()

        # === STEP 5: Run loading on new thread for performance ===
        # ------------------------------------
        threading.Thread(target=background_task).start()

    def startPipeline(self, gui_window: str, stock: str, stop_event: threading.Event):
		# === STEP 1: Ensure stop_event initialisation is correct ===
        # ------------------------------------
        if self.pipeline_running:
            self.logger.warning("Pipeline already running.")
            return
        self.pipeline_running = True

		# ====================================
		# ===   Place core pipeline in try/except for stop_event termination
        try:
            torch.use_deterministic_algorithms(self.args.deterministic)  # True => strict, False => normal
            torch.backends.cudnn.deterministic = self.args.deterministic
            torch.backends.cudnn.benchmark = not self.args.deterministic
            # NOTE: self._set_all_seeds(run_seed) was already called by run_experiments()

            # === STEP 2: Clear memory states ===
            # ------------------------------------
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # === STEP 3: Front-end updates for chosen stock and prediction horizon ===
            # ------------------------------------

            #   1. Clear UI state
            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.frontendApp.set_status("Starting...")

            #   2. Initialise trainer and evaluator
            evaluator = EvaluationMethods(
                self.frontendApp
            )
            evaluator.reset_histories()

            #   3. Establish data from chosen stock
            price_df = self.raw_feature_dfs[stock]

            #   4. Ensure sampling interval matches requested interval
            deltas = price_df.index.to_series().diff().dropna()
            if deltas.empty:
                raise RuntimeError("Not enough data to infer sampling interval")
            mode = deltas.mode()
            sample_interval = mode.iloc[0] if not mode.empty else deltas.median()
            bars_per_day = max(1, int(pd.Timedelta('1D') / sample_interval))

            #   5. Set chosen stock and prediction horizon from front-end
            horizon = Utils.parse_window(gui_window, bars_per_day)
            self.horizon = horizon
            self.seq_len = self.args.seq_len

            #   6. Check stop_event
            self._check_stop(stop_event)

            # === STEP 4: Raw Data Split ===
            # ----------------------------------------------------
            self.frontendApp.set_status("Preparing raw data...")

            #   1. Align calendar across all tickers
            shared_index = set.intersection(*(set(df.index) for df in self.raw_feature_dfs.values()))
            shared_index = sorted(shared_index)

            cutoff_idx  = int(0.7 * len(shared_index))
            cutoff_date = shared_index[cutoff_idx]

            embargo_bars   = max(1, int(self.horizon))  # at least 1 bar embargo
            train_end_idx  = max(0, cutoff_idx - embargo_bars)
            train_end_date = shared_index[train_end_idx]

            # split
            train_raw_map = {t: df[df.index < train_end_date].copy()  for t, df in self.raw_feature_dfs.items()}
            val_raw_map   = {t: df[df.index >= cutoff_date].copy()    for t, df in self.raw_feature_dfs.items()}

            #   3. Check stop_event
            self._check_stop(stop_event)

            # === STEP 5: Feature Engineering ===
            # -----------------------------------------------------
            self.frontendApp.set_status("Extracting engineered features...")

            #   1. Build training features
            feat_ext_train = FeatureExtractor(train_raw_map, rollingVolWindow=self.seq_len)
            feat_ext_train.buildFeatureDfs()
            train_feats = feat_ext_train.dfFeats

            #   2. Build validation features
            feat_ext_val = FeatureExtractor(val_raw_map, rollingVolWindow=self.seq_len)
            feat_ext_val.buildFeatureDfs()
            val_feats = feat_ext_val.dfFeats

            #   3. Set minimum training and validation feature set lengths
            min_train_len = min(len(df) for df in train_feats.values())
            min_val_len   = min(len(df) for df in val_feats.values())

            #   4. Set single stock features for LSTM
            train_df_stock = train_feats[stock].iloc[-min_train_len:]
            val_df_stock   = val_feats[stock].iloc[-min_val_len:]
            
            #   5. Check stop_event
            self._check_stop(stop_event)

            # === STEP 6: Tensor Factory ===
            # -----------------------------------------------------
            self.frontendApp.set_status("Creating tensors...")

            #   1. Build tensors for train and validation sets
            tf_train = self.build_tensor_factory(horizon=self.horizon)
            tf_val   = self.build_tensor_factory(horizon=self.horizon)

            #   2. Check stop_event
            self._check_stop(stop_event)

            # === STEP 7: Graph Building ===
            # -----------------------------------------------------
            self.frontendApp.set_status("Build the graph...")

            #   1. Set up graph building
            self.graph_builder = None
            graph_builder = GraphBuilder(
                dfFeats=train_feats,
                max_k=self.get_max_k(),
                n_pca=3
            )
            self.graphBuilder = graph_builder
 
            #   2. Build graph (for STGNN) using all train features
            tickers, coords3d, pruned, mst = graph_builder.getLightGraph()
            G = graph_builder.buildNetworkX(tickers, coords3d, pruned)

            #   3. Send graph to front-end
            self.frontendApp.root.after(0, lambda: self.frontendApp.plot3d_on_ax(
                tickers=tickers,
                coords=coords3d,
                pruned_edges=pruned,
                mst_edges=mst
            ))

            #   4. Extract (src, dst) pairs and format as 2×E tensor for PyG.
            edge_index = None
            edge_index = torch.tensor(
                [(i, j) for i, j, _ in pruned],
                dtype=torch.long
            ).t().contiguous()

            #   5. Keep CPU copy for later, make GOU copy for training
            self.init_edge_index = edge_index.clone()      # keep a CPU copy
            self.graphBuilder.edge_index = self.init_edge_index
            self.graphBuilder.edge_index = self.init_edge_index

            #   6. Track memory of graph components
            Utils.log_graph_memory(G, coords3d, edge_index, tag="Initial")

            #   7. Construct DataFrame with current features (per stock)
            latest_feats = []
            for t in self.args.tickers:
                df = train_feats.get(t)
                if df is not None and len(df) > 0:
                    latest_row = df.iloc[-1]
                    latest_feats.append({
                        "return": latest_row.get("return", np.nan),
                        "volatility": latest_row.get("volatility", np.nan),
                        "volume": latest_row.get("volume", np.nan),
                        "momentum": latest_row.get("momentum", np.nan)
                    })
            node_df = pd.DataFrame(latest_feats, index=self.args.tickers)
            node_df.index = node_df.index.astype(str)

            #   9. Update front-end
            self.frontendApp.root.after(0, lambda: self.frontendApp.updateTable(node_df))

            #   10. Check stop_event
            self._check_stop(stop_event)

            wd = self.args.weight_decay

            # === MISC: per-run seeded DataLoader ===
            dl_gen = torch.Generator(device="cpu")
            # self.current_seed was set by self._set_all_seeds(run_seed) in run_experiments()
            # Fallback to base_seed if not set (defensive)
            seed_for_loaders = getattr(self, "current_seed", int(self.args.base_seed)) % (2**32)
            dl_gen.manual_seed(seed_for_loaders)

            # === STEP 8: LSTM Training Phase ===
            # ------------------------------------
            self.frontendApp.set_status("Training LSTM...")

            #   1. Clear and log memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            Utils.log_gpu_memory("Before LSTM")

            #   2. Create LSTM trainer dataset and dataloader
            lstm_train_ds = LSTMDataset(
                tf_train,
                {stock:train_df_stock},
                stock,
                self.horizon
            )

            dl_lstm_train = torch.utils.data.DataLoader(
                lstm_train_ds,
                batch_size=self.args.batch_size,
                shuffle=True,
                generator=dl_gen,
                worker_init_fn=self._seed_worker
            )            
            dl_gru_train = dl_lstm_train

            #   3. Create LSTM validation dataset
            lstm_val_ds = LSTMDataset(
                tf_val,
                {stock: val_df_stock},
                stock,
                self.horizon
            )

            #   4. Initialise LSTM model
            lstm_model = LSTMClassifier(
                feature_dim=len(self.raw_feature_cols),
                hidden_dim=self.args.lstm_hidden,
                num_layers=self.args.lstm_layers,
                out_channels=1,
                bidirectional=self.args.bidirectional,
                dropout=self.args.dropout
            ).to(self.device)

            #   5. Build LSTM trainer
            trainer_lstm = Trainer(
                lstm_model,
                Adam(lstm_model.parameters(), lr=self.args.lstm_lr),
                BCEWithLogitsLoss(),
                self.device,
                graphBuilder=None,
                features=None,
                tickers=[stock],
                targetTicker=stock,
                frontend=self.frontendApp,
                evaluator=evaluator,
                prediction_horizon=self.horizon,
                seq_len=self.seq_len,
                model_name="LSTM",
                weight_decay=wd
            )

            #   6. Train LSTM model (with efficiency tracking)
            l_start = time.time()
            trainer_lstm.train(
                dl_lstm_train,
                self.args.lstm_epochs,
                stop_event=stop_event,
            )
            lstm_energy_Wh = getattr(trainer_lstm, "total_energy_Wh", None)
            lstm_train_secs = getattr(trainer_lstm, "total_train_seconds", None)
            lstm_avg_power_W = getattr(trainer_lstm, "avg_power_W", None)
            self.logger.info(f"[startPipeline] LSTM training completed in {time.time() - l_start:.2f}s")
            Utils.log_gpu_memory("After LSTM")

            #   7. Check stop_event
            self._check_stop(stop_event)

            # === STEP 9: LSTM Evaluation Phase ===
            # --------------------------------------
            self.frontendApp.set_status("Evaluating LSTM...")

            #   1. Perform post-training evaluation
            trainer_lstm.prediction_horizon = self.horizon
            eval_result_lstm = trainer_lstm.evaluate_rolling(lstm_val_ds)

            trainer_lstm.calibrate_threshold_from_eval(eval_result_lstm, method="f1")

            #   2. Update front-end
            metrics_lstm = evaluator.evaluate(
                model_name="LSTM",
                result=eval_result_lstm,
                price_df=price_df
            )
            
            #   3. Check stop_event
            self._check_stop(stop_event)

            # === STEP 10: LSTM Final Prediction Phase ===
            # ------------------------------------
            self.frontendApp.set_status("Predicting with LSTM...")

            #   1. Prepare data in accordance to training and validation for future prediction
            live_l = val_df_stock[self.raw_feature_cols].iloc[-self.seq_len:].values.astype(np.float32)
            arr_l = torch.tensor(live_l, dtype=torch.float32, device=self.device).unsqueeze(0)

            #   2. Predict directional confidence
            prob_l = torch.sigmoid(lstm_model(arr_l)[0]).item()
            thr_l = getattr(trainer_lstm, "decision_threshold", 0.5)
            dir_l, conf_l = (
                ("Upwards", (prob_l - thr_l) / max(1 - thr_l, 1e-12) * 100.0)
                if prob_l >= thr_l else
                ("Downwards", (thr_l - prob_l) / max(thr_l, 1e-12) * 100.0)
            )

            #   3. Update front-end
            self.logger.info(f"[startPipeline] LSTM={dir_l} ({conf_l:.1f}%)")
            self.frontendApp.root.after(0, lambda: self.frontendApp.updateResults(
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l, 
                "-", 0.0, 
                "-", 0.0
            ))
            
            #   4.  Check stop_event
            self._check_stop(stop_event)

            # === STEP 11: Clear memory ===
            # ------------------------------------
            del lstm_model
            del dl_lstm_train
            torch.cuda.empty_cache()  # Clears unused cached memory
            gc.collect()              # Force Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            Utils.log_gpu_memory("Before STGNN")

            # === STEP 12: GRU Training Phase ===
            self.frontendApp.set_status("Training GRU...")

            gru_model = GRUClassifier(
                feature_dim=len(self.raw_feature_cols),
                hidden_dim=self.args.lstm_hidden,
                num_layers=self.args.lstm_layers,
                out_channels=1,
                bidirectional=self.args.bidirectional,
                dropout=self.args.dropout
            ).to(self.device)

            g_start = time.time()
            trainer_gru = Trainer(
                gru_model,
                Adam(gru_model.parameters(), lr=self.args.lstm_lr),
                BCEWithLogitsLoss(),
                self.device,
                graphBuilder=None,
                features=None,
                tickers=[stock],
                targetTicker=stock,
                frontend=self.frontendApp,
                evaluator=evaluator,
                prediction_horizon=self.horizon,
                seq_len=self.seq_len,
                model_name="GRU",
                weight_decay=wd
            )

            trainer_gru.train(dl_gru_train, self.args.lstm_epochs, stop_event=stop_event)
            gru_energy_Wh = getattr(trainer_gru, "total_energy_Wh", None)
            gru_train_secs = getattr(trainer_gru, "total_train_seconds", None)
            gru_avg_power_W = getattr(trainer_gru, "avg_power_W", None)

            self.logger.info(f"[startPipeline] GRU training completed in {time.time() - g_start:.2f}s")
            Utils.log_gpu_memory("After GRU")

            self._check_stop(stop_event)
            self.frontendApp.set_status("Evaluating GRU...")

            eval_result_gru = trainer_gru.evaluate_rolling(lstm_val_ds)
            trainer_gru.calibrate_threshold_from_eval(eval_result_gru, method="f1")

            metrics_gru = evaluator.evaluate(
                "GRU", 
                eval_result_gru, 
                price_df=price_df
            )

            # === STEP 12.5: GRU Final Prediction Phase ===
            # ----------------------------------------------
            self.frontendApp.set_status("Predicting with GRU...")

            #   1. Prepare data for prediction
            live_g = val_df_stock[self.raw_feature_cols].iloc[-self.seq_len:].values.astype(np.float32)
            arr_g = torch.tensor(live_g, dtype=torch.float32, device=self.device).unsqueeze(0)

            #   2. Predict directional confidence
            with torch.no_grad():
                prob_g = torch.sigmoid(gru_model(arr_g)[0]).item()
            thr_g = getattr(trainer_gru, "decision_threshold", 0.5)  # NEW
            dir_g, conf_g = (
                ("Upwards", (prob_g - thr_g) / max(1 - thr_g, 1e-12) * 100.0)
                if prob_g >= thr_g else
                ("Downwards", (thr_g - prob_g) / max(thr_g, 1e-12) * 100.0)
            )

            #   3. Log and optionally update the GUI immediately
            self.logger.info(f"[startPipeline] GRU={dir_g} ({conf_g:.1f}%)")
            self.frontendApp.root.after(0, lambda: self.frontendApp.updateResults(
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l,
                f"{dir_g} (Next {horizon//bars_per_day}d)", conf_g,
                "-", 0.0
            ))

            # === STEP 13: Clear memory ===
            # ------------------------------------
            del gru_model
            del dl_gru_train
            torch.cuda.empty_cache()  # Clears unused cached memory
            gc.collect()              # Force Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            Utils.log_gpu_memory("Before STGNN")

            # === STEP 14: STGNN — Graph-Based Model Pipeline ===
            # ------------------------------------

            self.frontendApp.set_status("Training STGNN...")

            #   1. Use full train set of all tickers and the fixed graph from earlier.
            stgnn_model = STGNNClassifier(
                edge_index     = edge_index,
                num_nodes      = len(self.args.tickers),
                feature_dim    = len(self.raw_feature_cols) + 1,  # +1 for the target‐node flag
                tcn_channels   = self.args.tcn_channels,
                tcn_kernel     = self.args.tcn_kernel_size,
                gcn_hidden     = self.args.gcn_hidden,
                stgnn_blocks     = self.args.stgnn_blocks,
                out_dim        = 1,
                dropout        = self.args.dropout
            ).to(self.device)

            #   2. Compute edge weights
            edge_attr = self.graphBuilder.build_edge_weight_tensor(self.init_edge_index).to(self.device)
            stgnn_model.edge_attr = edge_attr

            #   3. Create STGNN trainer dataset and dataloader
            stgnn_train_ds = STGNNDataset(
                tf_train,
                graph_builder,
                train_feats,
                self.args.tickers,
                self.init_edge_index,
                stock,
                self.horizon
            )

            dl_stgnn_train = GeoDataLoader(
                stgnn_train_ds,
                batch_size=self.args.batch_size,
                shuffle=True,
                generator=dl_gen,
                worker_init_fn=self._seed_worker
            )           

            #   4. Create STGNN validation dataset
            stgnn_val_ds = STGNNDataset(
                tf_val,
                graph_builder,
                val_feats,
                self.args.tickers,
                self.init_edge_index,
                stock,
                self.horizon
            )

            #   5. Initialise STGNN model
            trainer_stgnn = Trainer(
                stgnn_model,
                Adam(stgnn_model.parameters(), lr=self.args.stgnn_lr),
                BCEWithLogitsLoss(),
                self.device,
                self.graphBuilder,
                {"feature": None},
                self.args.tickers,
                stock,
                self.frontendApp,
                evaluator,
                prediction_horizon=self.horizon,
                seq_len=self.seq_len,
                model_name="STGNN",
                weight_decay=wd
            )
            
            #   6. Train STGNN model (with efficiency tracking)
            s_start = time.time()

            trainer_stgnn.train(
                dl_stgnn_train,
                num_epochs=self.args.stgnn_epochs,
                stop_event=stop_event
            )
            stgnn_energy_Wh = getattr(trainer_stgnn, "total_energy_Wh", None)
            stgnn_train_secs = getattr(trainer_stgnn, "total_train_seconds", None)
            stgnn_avg_power_W = getattr(trainer_stgnn, "avg_power_W", None)
            self.logger.info(f"[startPipeline] STGNN training completed in {time.time() - s_start:.2f}s")
            Utils.log_gpu_memory("After STGNN")

            #   7. Check stop_event
            self._check_stop(stop_event)

            # === STEP 15: Extract latent embeddings from STGNN ===
            # ------------------------------------
            with torch.no_grad():
                #   1. Prepare input data for embedding — use TRAINING features only
                valid_feats = {
                    t: df for t, df in train_feats.items()
                    if len(df) >= self.seq_len + self.horizon
                }
                X_all = np.stack([
                    valid_feats[t][self.raw_feature_cols].iloc[-self.seq_len:].values.astype(np.float32)
                    for t in self.args.tickers
                ], axis=1)  # [T, N, F]
                X_all = torch.tensor(X_all, dtype=torch.float32, device=self.device).permute(1, 0, 2).unsqueeze(0)  # [1, N, T, F]

                #   2. Add target node indicator (not used for embedding but keeps input consistent)
                N, T = X_all.shape[1], X_all.shape[2]
                label_channel = torch.zeros((1, N, T, 1), dtype=torch.float32, device=self.device)
                stock_idx = self.args.tickers.index(stock)
                label_channel[0, stock_idx, :, 0] = 1.0
                arr_input = torch.cat([X_all, label_channel], dim=-1)  # [1, N, T, F+1]

                #   3. Get embeddings from model (shape: [1, N, 3])
                edge_attr = self.graphBuilder.build_edge_weight_tensor(self.init_edge_index).to(self.device)
                latent_embeddings = stgnn_model.embed(arr_input, self.init_edge_index.to(self.device), edge_attr).squeeze(0).cpu().numpy()
            
            #   4. Check stop_event
            self._check_stop(stop_event)
            
            # === STEP 16 (Optional): Rebuild the graph  ===
            # ------------------------------------
            rebuild_graph = bool(getattr(self.args, "rewiring", False))

            if rebuild_graph:
                self.frontendApp.set_status("Rebuilding graph...")

                #   1. Use the same features used to generate the latent node embeddings.
                graph_builder_refreshed = GraphBuilder(
                    dfFeats=train_feats,
                    max_k=self.get_max_k(),
                    n_pca=3
                )

                #   2. Update the back-end graph
                graph_builder_refreshed.set_node_embeddings(latent_embeddings)
                tickers_new, coords_new, pruned_new, mst_new = graph_builder_refreshed.getLightGraph()
                Utils.log_graph_memory(G, coords3d, edge_index, tag="Post-training")

                #   3. Update the front-end graph
                self.frontendApp.root.after(0, lambda: self.frontendApp.plot3d_on_ax(
                    tickers=tickers_new,
                    coords=coords_new,
                    pruned_edges=pruned_new,
                    mst_edges=mst_new
                ))

                #   4. Update model and trainer with new graph
                edge_index_new = torch.tensor(
                    [(i, j) for i, j, _ in pruned],
                    dtype=torch.long
                ).t().contiguous()
                trainer_stgnn.graphBuilder = graph_builder_refreshed
                trainer_stgnn.edge_index = edge_index_new.clone()
                trainer_stgnn.model.edge_index = edge_index_new

            # === STEP 17: STGNN Evaluation Phase ===
            # --------------------------------------
            self.frontendApp.set_status("Evaluating STGNN...")

            #   1. Perform post-training evaluation
            trainer_stgnn.prediction_horizon = self.horizon
            eval_result_stgnn = trainer_stgnn.evaluate_rolling(stgnn_val_ds)
            trainer_stgnn.calibrate_threshold_from_eval(eval_result_stgnn, method="f1")

            #   2. Update front-end
            metrics_stgnn = evaluator.evaluate(
                model_name="STGNN",
                result=eval_result_stgnn,
                price_df=price_df
            )
            
            #   3. Check stop_event
            self._check_stop(stop_event)

            # === STEP 18: STGNN Final Prediction Phase ===
            # ------------------------------------
            self.frontendApp.set_status("Predicting with STGNN...")

            #   1. Filter by required columns
            valid_feats = {
                t: df for t, df in val_feats.items()
                if len(df) >= self.seq_len + self.horizon
            }

            #   2. Build feature tensor: shape [T, N, F]
            X_all = np.stack([
                valid_feats[t][self.raw_feature_cols].iloc[-self.seq_len:].values.astype(np.float32)
                for t in self.args.tickers
            ], axis=1)  # → [T, N, F]

            #   3. Permute to [1, N, T, F]
            X_all = torch.tensor(X_all, dtype=torch.float32, device=self.device).permute(1, 0, 2).unsqueeze(0)

            #   4. Add label channel: [1, N, T, 1]
            N, T = X_all.shape[1], X_all.shape[2]
            label_channel = torch.zeros((1, N, T, 1), dtype=torch.float32, device=self.device)
            stock_idx = self.args.tickers.index(stock)
            label_channel[0, stock_idx, :, 0] = 1.0  # mark target node

            #   5. Concatenate features and label mask: [1, N, T, F+1]
            arr_s = torch.cat([X_all, label_channel], dim=-1)

            #   6. Predict direction with STGNN
            logits = stgnn_model(arr_s, self.init_edge_index.to(self.device), edge_attr=edge_attr, target_node_index=stock_idx)
            prob_s = torch.sigmoid(logits[0]).item()
            thr_s = getattr(trainer_stgnn, "decision_threshold", 0.5)
            dir_s_str, conf_s = (
                ("Upwards", (prob_s - thr_s) / max(1 - thr_s, 1e-12) * 100.0)
                if prob_s >= thr_s else
                ("Downwards", (thr_s - prob_s) / max(thr_s, 1e-12) * 100.0)
            )

		    # === STEP 19: Final front-end update ===
            # ------------------------------------
            self.frontendApp.set_status("Predictions completed.")

            #   1. Update frontend with both models' outputs
            if self.frontendApp:
                self.frontendApp.root.after(0, lambda: self.frontendApp.updateResults(
                    f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l,
                    f"{dir_g} (Next {horizon//bars_per_day}d)", conf_g,
                    f"{dir_s_str} (Next {horizon//bars_per_day}d)", conf_s
                ))

            #   2. Refresh tabs for visibility
            self.frontendApp.root.after(0, lambda: self.frontendApp.refresh_selected_tabs())

            #   3. Establish pipeline completion
            self.pipeline_running = False
            
            # === STEP 20: Log results for auto-test ===
            # ------------------------------------
            if getattr(app.args, "save_results", True):
                self.results_log = getattr(self, "results_log", [])
                self.results_log.extend([
                    {
                        **metrics_lstm,
                        "ticker": stock,
                        "horizon": gui_window,
                        "seed": self.current_seed,
                        "energy_Wh": lstm_energy_Wh,
                        "train_seconds": lstm_train_secs,
                        "avg_power_W": lstm_avg_power_W,
                        "model": "LSTM",
                    },
                    {
                        **metrics_gru,
                        "ticker": stock,
                        "horizon": gui_window,
                        "seed": self.current_seed,
                        "energy_Wh": gru_energy_Wh,
                        "train_seconds": gru_train_secs,
                        "avg_power_W": gru_avg_power_W,
                        "model": "GRU",
                    },
                    {
                        **metrics_stgnn,
                        "ticker": stock,
                        "horizon": gui_window,
                        "seed": self.current_seed,
                        "energy_Wh": stgnn_energy_Wh,
                        "train_seconds": stgnn_train_secs,
                        "avg_power_W": stgnn_avg_power_W,
                        "model": "STGNN",
                    },
                ])

            return (
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l,
                f"{dir_g} (Next {horizon//bars_per_day}d)", conf_g,
                f"{dir_s_str} (Next {horizon//bars_per_day}d)", conf_s
            ) if all(v is not None for v in [dir_l, dir_g, dir_s_str]) else ("—", 0.0, "—", 0.0, "—", 0.0)
            
        except InterruptedError:
            # ====================================
		    # ===   Raises if stop_event triggered termination
            self.logger.info("[startPipeline] Pipeline interrupted by stop_event")
            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.pipeline_running = False
            return ("-", 0.0, "-", 0.0, "-", 0.0)
        
        finally:
            self.pipeline_running = False

    def run(self):
		# ====================================
		# ===   Helper to load data and run main loop
        threading.Thread(target=self._load_data_and_start_gui).start()
        self.frontendApp.root.mainloop()

    def _load_price_history(self, tickers, period="729d", return_handler=False):
        # ====================================
		# ===   Helper to pass raw data from API
        rd = RawDataHandler(
            source=self.args.mode,
            tickers=tickers,
            dataDir=self.args.data_dir,
            avApiKey=self.args.av_key,
            yfPeriod=period,
            yfInterval=self.args.interval
        )
        data = {t: rd.getDataframe(t) for t in tickers}
        return (data, rd) if return_handler else data
    
    def _check_stop(self, stop_event: threading.Event):
        # ====================================
		# ===   Helper to terminate all processes
        if stop_event and stop_event.is_set():
            self.logger.info("[MainApp] Pipeline stopped via stop_event.")
            raise InterruptedError("Pipeline interrupted")
    
    def get_max_k(self):
        # ====================================
        # ===   Helper to control graph efficiency
        return self.args.max_k
    
    def build_edge_index(self, coords, edges, *_):
        # ====================================
		# ===   Helper to build edges indices from graph
        if edges:
            rows, cols = zip(*[(i, j) for i, j, _ in edges])
            return torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        return torch.zeros((2, 0), dtype=torch.long, device=self.device)
    
    def build_tensor_factory(self, horizon):
        # ====================================
		# ===   Helper to build tensor factories
        return TensorFactory(
            tickers=self.args.tickers,
            featureCols=self.raw_feature_cols,
            seq_len=self.seq_len,
            prediction_horizon=horizon
        )
    
    def _set_all_seeds(self, seed: int):
        """Set all RNGs and remember the current run seed."""
        used = Utils.set_seed(seed, deterministic=self.args.deterministic)
        self.current_seed = int(used)  # keep a 32-bit-ish int around for logs
        self.logger.info(f"[AutoTest] Using seed={self.current_seed} (deterministic={self.args.deterministic})")
    
    def _seed_worker(self, worker_id: int):
        """Ensure each DataLoader worker has a distinct, reproducible seed."""
        worker_seed = (self.current_seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        import random as _random
        _random.seed(worker_seed)

    def run_experiments(self):
        # Wait until background loader built raw_feature_dfs and bound the UI
        self.data_ready.wait()

        rewiring_options = [True, False]
        max_k_values = [2]
        seq_len_values = [2]
        repetitions = 2

        param_grid = list(itertools.product(rewiring_options, max_k_values, seq_len_values))

        os.makedirs(self.args.results_dir, exist_ok=True)
        all_results = []

        for (rewire, k, seq_len) in param_grid:
            exp_name = f"{self.args.experiment_name}_rewire-{int(rewire)}_k-{k}_seq-{seq_len}"
            csv_path = os.path.join(self.args.results_dir, f"{exp_name}.csv")
            self.logger.info(f"[AutoTest] Starting config: rewiring={rewire}, max_k={k}, seq_len={seq_len}")

            for stock in self.args.tickers:
                stock_results = []
                self.logger.info(f"[AutoTest] Running stock: {stock}")

                # Reset to base for each stock
                stock_base_seed = int(self.args.base_seed) % (2**32)

                for rep in range(repetitions):
                    # 1) Apply config for this run
                    cfg = copy.deepcopy(self.args)
                    cfg.rewiring = rewire
                    cfg.max_k = k
                    cfg.seq_len = seq_len
                    self.args = cfg

                    # 2) Increment seed per repetition and set it everywhere
                    run_seed = (stock_base_seed + rep) % (2**32)
                    self._set_all_seeds(run_seed)

                    stop_event = threading.Event()
                    start_time = time.time()

                    try:
                        self.logger.info(f"[AutoTest] Run {rep+1}/{repetitions} for {stock}")

                        # Avoid double-counting from earlier runs
                        self.results_log = []

                        # Inline pipeline run (deterministic knobs already set in startPipeline)
                        self.startPipeline(gui_window="1d", stock=stock, stop_event=stop_event)

                        # Stamp metadata including the seed used
                        for entry in self.results_log:
                            entry.update({
                                "rewiring": rewire,
                                "max_k": k,
                                "seq_len": seq_len,
                                "ticker": stock,
                                "runtime_sec": round(time.time() - start_time, 2),
                                "rep": rep + 1,
                                "seed": self.current_seed,
                            })
                            stock_results.append(entry)

                    except Exception as e:
                        self.logger.error(f"[AutoTest] {stock} rep {rep+1} failed: {e}")
                        stock_results.append({
                            "ticker": stock,
                            "rewiring": rewire,
                            "max_k": k,
                            "seq_len": seq_len,
                            "rep": rep + 1,
                            "seed": getattr(self, "current_seed", None),
                            "error": str(e)
                        })

                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()

                # ---- append per-config results to <exp_name>.csv ----
                if stock_results:
                    df = pd.DataFrame(stock_results)

                    # keep raw rows
                    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
                    df.to_csv(csv_path, mode="a", header=write_header, index=False)
                    self.logger.info(f"[AutoTest] Appended {len(df)} entries for {stock} → {csv_path}")

                    # ---- compute mean and std summary rows across repetitions ----
                    numeric_cols = df.select_dtypes(include='number').columns

                    # mean across repetitions
                    mean_row = df[numeric_cols].mean(numeric_only=True)
                    mean_row["ticker"] = stock
                    mean_row["rewiring"] = rewire
                    mean_row["max_k"] = k
                    mean_row["seq_len"] = seq_len
                    mean_row["rep"] = "mean"    # summary marker
                    all_results.append(mean_row.to_dict())

                    # std across repetitions (sample std, ddof=1)
                    std_series = df[numeric_cols].std(numeric_only=True, ddof=1)
                    std_row = std_series.copy()
                    std_row["ticker"] = stock
                    std_row["rewiring"] = rewire
                    std_row["max_k"] = k
                    std_row["seq_len"] = seq_len
                    std_row["rep"] = "std"      # summary marker
                    all_results.append(std_row.to_dict())

        # ---- append (or create) summary CSV ----
        summary_path = os.path.join(self.args.results_dir, f"{self.args.experiment_name}_summary.csv")
        df_summary = pd.DataFrame(all_results)
        write_header = (not os.path.exists(summary_path)) or os.path.getsize(summary_path) == 0
        df_summary.to_csv(summary_path, mode="a", header=write_header, index=False)
        self.logger.info(f"[AutoTest] Summary appended: {len(df_summary)} rows → {summary_path}")

if __name__ == "__main__":
    app = MainApp()
    if getattr(app.args, "save_results", False):
        # Start the loader; it will set data_ready when finished
        threading.Thread(target=app._load_data_and_start_gui, daemon=True).start()

        # Launch experiments from a background thread once Tk is up
        def start_tests():
            threading.Thread(target=app.run_experiments, daemon=True).start()
        app.frontendApp.root.after(0, start_tests)

        # Enter Tk mainloop on the main thread
        app.frontendApp.root.mainloop()
    else:
        app.run()
        