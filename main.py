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

from FrontEnd           import FrontEnd
from RawDataHandler     import RawDataHandler
from FeatureExtractor   import FeatureExtractor
from TensorFactory      import TensorFactory
from GraphBuilder       import GraphBuilder
from Trainer            import Trainer
from LSTMClassifier     import LSTMClassifier
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
        Utils.set_seed()
        self.args = ConfigManager.parseArgs()
        self.args.interval = '1h'
        self.pipeline_running = False
        
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
        self.loader = LoadingOverlay(self.frontendApp.root, avi_path="misc\loading.avi", delay=24)
    
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

        # === STEP 5: Run loading on new thread for performance ===
        # ------------------------------------
        threading.Thread(target=background_task).start()

    def startPipeline(self, gui_window: str, stock: str, stop_event: threading.Event):
		# === STEP 1: Ensure stop_event initialisation is correct ===
        # ------------------------------------
        self.pipeline_running = False
        if self.pipeline_running:
            self.logger.warning("Pipeline already running.")
            return
        self.pipeline_running = True

		# ====================================
		# ===   Place core pipeline in try/except for stop_event termination
        try:
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
            self.seq_len = Utils.map_seq_len_from_horizon(horizon, bars_per_day)

            #   6. Check stop_event
            self._check_stop(stop_event)

            # === STEP 4: Raw Data Split ===
            # ----------------------------------------------------
            self.frontendApp.set_status("Preparing raw data...")

            #   1. Align calendar across all tickers
            shared_index = set.intersection(*(set(df.index) for df in self.raw_feature_dfs.values()))
            shared_index = sorted(shared_index)

            cutoff_idx = int(0.7 * len(shared_index))
            cutoff_date = shared_index[cutoff_idx]

            #   2. Split raw data on calendar
            train_raw_map = {
                t: df[df.index < cutoff_date].copy()
                for t, df in self.raw_feature_dfs.items()
            }
            val_raw_map = {
                t: df[df.index >= cutoff_date].copy()
                for t, df in self.raw_feature_dfs.items()
            }

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
            edge_index = torch.tensor([(i, j) for i, j, _ in pruned]).T  # or mst

            #   5. Keep CPU copy for later, make GOU copy for training
            self.init_edge_index = edge_index
            edge_index = edge_index.to(self.device)
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
            dl_lstm_train = torch.utils.data.DataLoader(lstm_train_ds, self.args.batch_size, shuffle=True)

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
                seq_len=self.seq_len
            )

            #   6. Train LSTM model (with efficiency tracking)
            l_start = time.time()
            trainer_lstm.train(
                dl_lstm_train,
                self.args.lstm_epochs,
                self.args.lstm_save,
                stop_event=stop_event,
            )
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

            #   2. Update front-end
            evaluator.evaluate(
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
            dir_l, conf_l = ("Upwards", prob_l * 100) if prob_l >= 0.5 else ("Downwards", (1 - prob_l) * 100)

            #   3. Update front-end
            self.logger.info(f"[startPipeline] LSTM={dir_l} ({conf_l:.1f}%)")
            self.frontendApp.root.after(0, lambda: self.frontendApp.updateResults(
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l, "-", 0.0
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

            # === STEP 12: STGNN — Graph-Based Model Pipeline ===
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
            dl_stgnn_train = GeoDataLoader(stgnn_train_ds, self.args.batch_size, shuffle=True)
            
            #   4. Create LSTM validation dataset
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
                seq_len=self.seq_len
            )
            
            #   6. Train STGNN model (with efficiency tracking)
            s_start = time.time()

            trainer_stgnn.train(
                dl_stgnn_train,
                num_epochs=self.args.stgnn_epochs,
                save_model_path=self.args.stgnn_save,
                stop_event=stop_event
            )
            self.logger.info(f"[startPipeline] STGNN training completed in {time.time() - s_start:.2f}s")
            Utils.log_gpu_memory("After STGNN")

            #   7. Check stop_event
            self._check_stop(stop_event)

		    # === STEP 13: Extract latent embeddings from STGNN ===
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
            
            # === STEP 14: Rebuild the graph  ===
            # ------------------------------------
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
            edge_index_new = torch.tensor([(i, j) for i, j, _ in pruned_new], dtype=torch.long).T.to(self.device)
            trainer_stgnn.graphBuilder = graph_builder_refreshed
            trainer_stgnn.edge_index = edge_index_new
            trainer_stgnn.model.edge_index = edge_index_new

            # === STEP 15: STGNN Evaluation Phase ===
            # --------------------------------------
            self.frontendApp.set_status("Evaluating STGNN...")

            #   1. Perform post-training evaluation
            trainer_stgnn.prediction_horizon = self.horizon
            eval_result_stgnn = trainer_stgnn.evaluate_rolling(stgnn_val_ds)

            #   2. Update front-end
            evaluator.evaluate(
                model_name="STGNN",
                result=eval_result_stgnn,
                price_df=price_df
            )
            
            #   3. Check stop_event
            self._check_stop(stop_event)

		    # === STEP 16: STGNN Final Prediction Phase ===
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
            dir_s_str = "Upwards" if prob_s >= 0.5 else "Downwards"
            conf_s = prob_s * 100 if dir_s_str == "Upwards" else (1 - prob_s) * 100

		    # === STEP 17: Final front-end update ===
            # ------------------------------------
            self.frontendApp.set_status("Predictions completed.")

            #   1. Update frontend with both models' outputs
            self.frontendApp.root.after(0, lambda: self.frontendApp.updateResults(
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l,
                f"{dir_s_str} (Next {horizon//bars_per_day}d)", conf_s
            ))

            #   2. Refresh tabs for visibility
            self.frontendApp.root.after(0, lambda: self.frontendApp.refresh_selected_tabs())

            #   3. Establish pipeline completion
            self.pipeline_running = False

            return (
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l,
                f"{dir_s_str} (Next {horizon//bars_per_day}d)", conf_s
            ) if dir_l and dir_s_str else ("-", 0.0, "-", 0.0)
        
        except InterruptedError:
            # ====================================
		    # ===   Raises if stop_event triggered termination
            self.logger.info("[startPipeline] Pipeline interrupted by stop_event")
            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.pipeline_running = False
            return ("-", 0.0, "-", 0.0)

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
    
if __name__ == "__main__":
    app = MainApp()
    app.run()
