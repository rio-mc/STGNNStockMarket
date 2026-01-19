import copy
from threading import Thread
import threading
import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
import logging
import time
import gc    
from torch_geometric.loader import DataLoader as GeoDataLoader
import os
import itertools
import platform
from datetime import datetime, timezone

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

        self.graph_homophily = float("nan")

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
        # Engineered feature pool
        engineered = ["return", "volatility", "momentum"]

        # Remove exactly one if ablation is enabled
        if self.args.ablate_feature != "none":
            engineered = [f for f in engineered if f != self.args.ablate_feature]

        # Final node input cols always include close
        self.raw_feature_cols = ["close"] + engineered

        self.logger.info(
            f"[Ablation] ablate_feature={self.args.ablate_feature} | "
            f"node_raw_feature_cols={self.raw_feature_cols}"
        )

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
        self.logger.info(f"[Env] Python={platform.python_version()}")
        self.logger.info(f"[Env] PyTorch={torch.__version__}")
        self.logger.info(f"[Env] CUDA available={torch.cuda.is_available()}")
        self.logger.info(f"[Env] CUDA version={torch.version.cuda}")
        self.logger.info(f"[Env] cuDNN={torch.backends.cudnn.version()}")
        self.logger.info(f"[Env] NumPy={np.__version__}")
        
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
            sector_map_path = "tickers.csv"
            ticker_to_sector = Utils.load_ticker_to_sector(sector_map_path)

            self.frontendApp.set_sector_map(ticker_to_sector)

            # Optional sanity check
            missing = [t for t in self.args.tickers if t not in ticker_to_sector]
            if missing:
                self.logger.warning(
                    "[SectorMap] %d tickers missing sector labels: %s",
                    len(missing), missing[:5]
                )

            #   1. Clear UI state
            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.frontendApp.set_status("Starting...")

            #   2. Initialise trainer and evaluator
            evaluator = self.frontendApp.evaluator
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
            feat_ext_train = FeatureExtractor(
                train_raw_map,
                rollingVolWindow=self.seq_len,
                norm_stats=None,
                fit_normaliser=True,
                ablate_feature=self.args.ablate_feature,
            )

            feat_ext_train.buildFeatureDfs()
            train_feats = feat_ext_train.dfFeats
            train_norm_stats = feat_ext_train.get_norm_stats()

            #   2. Build validation features
            feat_ext_val = FeatureExtractor(
                val_raw_map,
                rollingVolWindow=self.seq_len,
                norm_stats=train_norm_stats,
                fit_normaliser=False,
                ablate_feature=self.args.ablate_feature,
            )

            feat_ext_val.buildFeatureDfs()
            val_feats = feat_ext_val.dfFeats

            #   3. Set minimum training and validation feature set lengths
            min_train_len = min(len(df) for df in train_feats.values())
            min_val_len   = min(len(df) for df in val_feats.values())

            # === Align time windows across ALL models ===
            # Truncate every ticker to the same contiguous window within each split.
            # Use the tail so the window ends at the split boundary (most recent info).
            train_feats = {t: df.iloc[-min_train_len:].copy() for t, df in train_feats.items()}
            val_feats   = {t: df.iloc[-min_val_len:].copy()   for t, df in val_feats.items()}

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

            graph_builder = GraphBuilder(
                dfFeats=train_feats,
                max_k=self.get_max_k(),
                n_pca=3,
                ticker_to_sector=ticker_to_sector,
                graph_embed=self.args.graph_embed,
                ablate_feature=self.args.ablate_feature,
            )

            self.logger.info(
                f"[Ablation] graph_embed={self.args.graph_embed} | "
                f"ablate_feature={self.args.ablate_feature}"
            )

            self.graphBuilder = graph_builder

            # Build graph (for STGNN) using all train features
            tickers, coords3d, pruned, mst = graph_builder.getLightGraph()
            G = graph_builder.buildNetworkX(tickers, coords3d, pruned)

            mode = getattr(self.args, "graph_ablation", "none")
            num_nodes = len(tickers)

            # --- Plot graph: hide cross-asset edges for identity/empty so UI matches ablation ---
            pruned_for_plot = pruned
            mst_for_plot = mst
            if mode in ("identity", "empty"):
                pruned_for_plot = []
                mst_for_plot = []

            self.frontendApp.root.after(0, lambda: self.frontendApp.plot3d_on_ax(
                tickers=tickers,
                coords=coords3d,
                pruned_edges=pruned_for_plot,
                mst_edges=mst_for_plot
            ))
            
            # --- Build edge_index for PyG from pruned edges ---
            edge_index = torch.tensor(
                [(i, j) for i, j, _ in pruned],
                dtype=torch.long
            ).t().contiguous()

            # IMPORTANT: use the SAME node count as the graph builder output
            mode = getattr(self.args, "graph_ablation", "none")
            num_nodes = len(tickers)

            # If max_k=0 (or pruning) yields no edges, PyG convs (e.g., NNConv) can break on empty edge_index.
            # Represent "no relational edges" as identity/self-loops only (no cross-asset mixing).
            # This is paper-defensible: k=0 => remove cross-asset edges, keep self loops to keep MP well-defined.
            if edge_index.numel() == 0:
                idx = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
                edge_index = torch.stack([idx, idx], dim=0)
                # Optional: track this for logging/debugging if you have a logger
                if hasattr(self, "logger"):
                    self.logger.info("[Graph] Empty edge_index after pruning (likely max_k=0). Using identity self-loops.")

            # Make edge_index undirected for PyG message passing (cosine similarity is symmetric)
            # NOTE: self-loops remain self-loops after reversal/dedupe, so this is safe.
            rev = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, rev], dim=1)

            # Dedupe + stable sort by (src, dst)
            key = edge_index[0] * num_nodes + edge_index[1]
            uniq = torch.unique(key, sorted=True)
            edge_index = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0)

            # Apply optional graph ablation (none / identity / empty) ONCE, after finalisation
            # NOTE: If you ever set mode="empty", it will override the self-loop fallback above.
            edge_index = Utils.apply_graph_ablation(edge_index, num_nodes=num_nodes, mode=mode)

            # Optional: sanity log (now truthful for identity/empty)
            self.logger.info(
                f"[AblationCheck] mode={mode} | V={num_nodes} | E={edge_index.size(1)} | "
                f"all_self_loops={(edge_index.size(1) > 0 and (edge_index[0] == edge_index[1]).all().item())}"
            )

            requested_k = self.get_max_k()
            effective_k = getattr(graph_builder, "effective_k", requested_k)

            # Freeze this graph for datasets/training
            self.init_edge_index = edge_index.clone()
            self.graphBuilder.edge_index = self.init_edge_index

            # --- Post-hoc graph interpretability metric (NOT used in training) ---
            try:
                self.graph_homophily = graph_builder.sector_homophily_from_edge_index(
                    tickers=tickers,
                    edge_index=self.init_edge_index,
                    ignore_unknown=True,
                    ignore_self_loops=True,
                )
                self.logger.info(
                    f"[Graph] sector_homophily={self.graph_homophily:.4f} "
                    "(ignore_unknown=True, ignore_self_loops=True)"
                )
            except Exception as e:
                self.logger.warning(f"[Graph] homophily computation failed: {e}")
                self.graph_homophily = float("nan")

            # Log AFTER ablation
            num_edges = edge_index.size(1)
            possible_directed_no_self = num_nodes * (num_nodes - 1)
            graph_density = (num_edges / possible_directed_no_self) if possible_directed_no_self > 0 else 0.0

            self.logger.info(
                f"[Graph] ablation={mode} requested_k={requested_k} effective_k={effective_k} "
                f"|V|={num_nodes} |E|={num_edges} density={graph_density:.6f}"
            )

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
            gru_train_ds = lstm_train_ds

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

            gru_val_ds = lstm_val_ds

            #   4. Initialise LSTM model
            lstm_model = LSTMClassifier(
                feature_dim=len(self.raw_feature_cols),
                hidden_dim=self.args.lstm_hidden,
                num_layers=self.args.lstm_layers,
                out_channels=1,
                bidirectional=self.args.bidirectional,
                dropout=self.args.dropout,
                rep_dim=self.args.rep_dim,
                head_hidden=self.args.head_hidden
            ).to(self.device)
            print(f"dropout={self.args.dropout}, rep_dim={self.args.rep_dim}, head_hidden={self.args.head_hidden}")
            lstm_params = Utils.count_parameters(lstm_model)
            self.logger.info(f"LSTM parameters: {lstm_params:,}")

            #   5. Build LSTM trainer
            trainer_lstm = Trainer(
                lstm_model,
                Utils.make_adamw(lstm_model, lr=self.args.lstm_lr, weight_decay=self.args.weight_decay),
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
                model_name="LSTM"
            )

            #   6. Train LSTM model (with efficiency tracking)
            l_start = time.time()
            trainer_lstm.train(
                dl_lstm_train,
                self.args.lstm_epochs,
                stop_event=stop_event,
            )
            lstm_energy_Wh = getattr(trainer_lstm, "total_energy_Wh", None)
            lstm_energy_per_sample_Wh = getattr(trainer_lstm, "energy_per_sample_Wh", None)
            lstm_energy_epochs_Wh = getattr(trainer_lstm, "energy_epochs_Wh", None)
            lstm_energy_per_sample_epochs_Wh = getattr(trainer_lstm, "energy_per_sample_epochs_Wh", None)
            lstm_samples_per_epoch = getattr(trainer_lstm, "samples_per_epoch", None)

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

            # Post-hoc calibration for fair comparison (doesn't change predicted class at 0.5)
            if hasattr(lstm_model, "classifier") and hasattr(lstm_model.classifier, "set_temperature"):
                lstm_model.classifier.set_temperature(self.args.head_temperature)

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
            with torch.no_grad():
                prob_l = torch.sigmoid(lstm_model(arr_l)[0]).item()
                thr_l = metrics_lstm.get("best_threshold", 0.5)
                if prob_l >= thr_l:
                    dir_l = "Upwards"
                    conf_l = prob_l * 100.0          # chosen-class probability
                else:
                    dir_l = "Downwards"
                    conf_l = (1.0 - prob_l) * 100.0  # chosen-class probability

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
            del lstm_train_ds, lstm_val_ds
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
                dropout=self.args.dropout,
                rep_dim=self.args.rep_dim,
                head_hidden=self.args.head_hidden
            ).to(self.device)
            gru_params  = Utils.count_parameters(gru_model)
            self.logger.info(f"GRU parameters: {gru_params:,}")

            g_start = time.time()
            trainer_gru = Trainer(
                gru_model,
                Utils.make_adamw(gru_model, lr=self.args.lstm_lr, weight_decay=self.args.weight_decay),
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
                model_name="GRU"
            )

            trainer_gru.train(dl_gru_train, self.args.lstm_epochs, stop_event=stop_event)
            gru_energy_Wh = getattr(trainer_gru, "total_energy_Wh", None)
            gru_train_secs = getattr(trainer_gru, "total_train_seconds", None)
            gru_avg_power_W = getattr(trainer_gru, "avg_power_W", None)
            gru_energy_per_sample_Wh = getattr(trainer_gru, "energy_per_sample_Wh", None)
            gru_energy_epochs_Wh = getattr(trainer_gru, "energy_epochs_Wh", None)
            gru_energy_per_sample_epochs_Wh = getattr(trainer_gru, "energy_per_sample_epochs_Wh", None)
            gru_samples_per_epoch = getattr(trainer_gru, "samples_per_epoch", None)

            self.logger.info(f"[startPipeline] GRU training completed in {time.time() - g_start:.2f}s")
            Utils.log_gpu_memory("After GRU")

            self._check_stop(stop_event)
            self.frontendApp.set_status("Evaluating GRU...")

            eval_result_gru = trainer_gru.evaluate_rolling(gru_val_ds)

            if hasattr(gru_model, "classifier") and hasattr(gru_model.classifier, "set_temperature"):
                gru_model.classifier.set_temperature(self.args.head_temperature)

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
                thr_g = metrics_gru.get("best_threshold", 0.5)
                if prob_g >= thr_g:
                    dir_g = "Upwards"
                    conf_g = prob_g * 100.0
                else:
                    dir_g = "Downwards"
                    conf_g = (1.0 - prob_g) * 100.0


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
            del gru_train_ds, gru_val_ds
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
                edge_index      = self.init_edge_index,
                num_nodes       = len(self.args.tickers),
                feature_dim     = len(self.raw_feature_cols) + 1,  # +1 for the target‐node flag
                tcn_channels    = self.args.tcn_channels,
                tcn_kernel      = self.args.tcn_kernel_size,
                gcn_hidden      = self.args.gcn_hidden,
                stgnn_blocks    = self.args.stgnn_blocks,
                out_dim         = 1,
                dropout         = self.args.dropout,
                rep_dim         = self.args.rep_dim,
                head_hidden     = self.args.head_hidden
            ).to(self.device)

            stgnn_params = Utils.count_parameters(stgnn_model)
            self.logger.info(f"STGNN parameters: {stgnn_params:,}")

            #   2. Compute edge weights
            mode = getattr(self.args, "graph_ablation", "none")
            num_nodes = len(self.args.tickers)

            if mode == "empty":
                edge_attr = None
            elif mode == "identity":
                # one scalar feature per self-loop edge
                edge_attr = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)
            else:
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
                Utils.make_adamw(stgnn_model, lr=self.args.stgnn_lr, weight_decay=self.args.weight_decay),
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
                model_name="STGNN"
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
            stgnn_energy_per_sample_Wh = getattr(trainer_stgnn, "energy_per_sample_Wh", None)
            stgnn_energy_epochs_Wh = getattr(trainer_stgnn, "energy_epochs_Wh", None)
            stgnn_energy_per_sample_epochs_Wh = getattr(trainer_stgnn, "energy_per_sample_epochs_Wh", None)
            stgnn_samples_per_epoch = getattr(trainer_stgnn, "samples_per_epoch", None)

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

                mode = getattr(self.args, "graph_ablation", "none")

                # 1. Use the same features used to generate the latent node embeddings.
                graph_builder_refreshed = GraphBuilder(
                    dfFeats=train_feats,
                    max_k=self.get_max_k(),
                    n_pca=3,
                    ticker_to_sector=ticker_to_sector  # keep sector metadata consistent
                )

                # 2. Update the back-end graph
                graph_builder_refreshed.set_node_embeddings(latent_embeddings)
                tickers_new, coords_new, pruned_new, mst_new = graph_builder_refreshed.getLightGraph()

                # 3. Update UI graph (hide cross-asset edges if identity/empty)
                pruned_new_for_plot = pruned_new
                mst_new_for_plot = mst_new
                if mode in ("identity", "empty"):
                    pruned_new_for_plot = []
                    mst_new_for_plot = []

                self.frontendApp.root.after(0, lambda: self.frontendApp.plot3d_on_ax(
                    tickers=tickers_new,
                    coords=coords_new,
                    pruned_edges=pruned_new_for_plot,
                    mst_edges=mst_new_for_plot
                ))

                # 4. Update model and trainer with new graph, THEN re-apply ablation
                edge_index_new = torch.tensor([(i, j) for i, j, _ in pruned_new], dtype=torch.long).t().contiguous()
                num_nodes_new = len(tickers_new)
                mode = getattr(self.args, "graph_ablation", "none")

                # Make undirected + dedupe
                if edge_index_new.numel() > 0:
                    rev = edge_index_new[[1, 0], :]
                    edge_index_new = torch.cat([edge_index_new, rev], dim=1)

                    key = edge_index_new[0] * num_nodes_new + edge_index_new[1]
                    uniq = torch.unique(key, sorted=True)
                    edge_index_new = torch.stack([uniq // num_nodes_new, uniq % num_nodes_new], dim=0)

                # Apply ablation ONCE
                edge_index_new = Utils.apply_graph_ablation(edge_index_new, num_nodes=num_nodes_new, mode=mode)

                # Sync
                self.init_edge_index = edge_index_new.clone()
                trainer_stgnn.edge_index = edge_index_new.clone()
                trainer_stgnn.model.edge_index = edge_index_new

                # Optional: sanity log
                self.logger.info(
                    f"[AblationCheck-Rewire] mode={mode} | V={num_nodes_new} | E={edge_index_new.size(1)} | "
                    f"self_loops={(edge_index_new.size(1) > 0 and (edge_index_new[0] == edge_index_new[1]).all().item())}"
                )


            # === STEP 17: STGNN Evaluation Phase ===
            # --------------------------------------
            self.frontendApp.set_status("Evaluating STGNN...")

            #   1. Perform post-training evaluation
            trainer_stgnn.prediction_horizon = self.horizon
            eval_result_stgnn = trainer_stgnn.evaluate_rolling(stgnn_val_ds)

            if hasattr(stgnn_model, "classifier") and hasattr(stgnn_model.classifier, "set_temperature"):
                stgnn_model.classifier.set_temperature(self.args.head_temperature)

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
            with torch.no_grad():
                logits = stgnn_model(arr_s, self.init_edge_index.to(self.device), edge_attr=edge_attr, target_node_index=stock_idx)
                prob_s = torch.sigmoid(logits[0]).item()
                thr_s = metrics_stgnn.get("best_threshold", 0.5)
                if prob_s >= thr_s:
                    dir_s_str = "Upwards"
                    conf_s = prob_s * 100.0
                else:
                    dir_s_str = "Downwards"
                    conf_s = (1.0 - prob_s) * 100.0

            # === STEP 19: Final front-end update ===
            # ------------------------------------
            self.frontendApp.set_status("Predictions completed.")

            #   1. Update frontend with all models' outputs
            self.frontendApp.root.after(
                0,
                self.frontendApp.updateResults,
                f"{dir_l} (Next {horizon//bars_per_day}d)", conf_l,
                f"{dir_g} (Next {horizon//bars_per_day}d)", conf_g,
                f"{dir_s_str} (Next {horizon//bars_per_day}d)", conf_s,
            )

            #   2. Refresh tabs for visibility
            self.frontendApp.root.after(0, lambda: self.frontendApp.refresh_selected_tabs())

            #   3. Establish pipeline completion
            self.pipeline_running = False

            # Garbage collection before logging
            del stgnn_model
            del dl_stgnn_train
            del stgnn_train_ds, stgnn_val_ds
            torch.cuda.empty_cache()  # Clears unused cached memory
            gc.collect()              # Force Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # === STEP 20: Log results for auto-test ===
            # ------------------------------------
            if getattr(self.args, "save_results", True):
                self.results_log = getattr(self, "results_log", [])

            # Timestamp metadata for later validation
            pred_made_at_utc = datetime.now(timezone.utc).isoformat()
            bar_interval = str(getattr(self.args, "interval", "1h"))
            horizon_bars = int(self.horizon) if getattr(self, "horizon", None) is not None else None

            # Use the last timestamp that fed the prediction features.
            # This should exist if val_df_stock is built; keep defensive fallback anyway.
            try:
                features_end_ts = pd.Timestamp(val_df_stock.index[-1]).isoformat()
            except Exception:
                features_end_ts = None

            # Defensive defaults (avoid None leaking into CSV)
            dir_l_safe = dir_l if dir_l is not None else "-"
            dir_g_safe = dir_g if dir_g is not None else "-"
            dir_s_safe = dir_s_str if dir_s_str is not None else "-"

            def _mk_row(
                *,
                metrics: dict | None,
                model: str,
                direction: str,
                confidence_pct: float | None,
                raw_score: float | None,
                decision_threshold: float | None,
                energy_Wh: float | None,
                energy_per_sample_Wh: float | None,
                train_seconds: float | None,
                avg_power_W: float | None,
                graph_ablation: str | None = None,
                num_edges: int | None = None,
                graph_homophily: float | None = None,
            ) -> dict:
                row = {
                    **(metrics or {}),

                    "ticker": stock,
                    "model": model,
                    "horizon": gui_window,
                    "seed": int(self.current_seed) if self.current_seed is not None else None,

                    "direction": direction,
                    "confidence_pct": float(confidence_pct) if confidence_pct is not None else None,

                    "raw_score": float(raw_score) if raw_score is not None else None,
                    "decision_threshold": float(decision_threshold) if decision_threshold is not None else None,

                    "pred_made_at_utc": pred_made_at_utc,
                    "features_end_ts": features_end_ts,
                    "horizon_bars": horizon_bars,
                    "bar_interval": bar_interval,

                    "energy_Wh": float(energy_Wh) if energy_Wh is not None else None,
                    "energy_per_sample_Wh": float(energy_per_sample_Wh) if energy_per_sample_Wh is not None else None,
                    "train_seconds": float(train_seconds) if train_seconds is not None else None,
                    "avg_power_W": float(avg_power_W) if avg_power_W is not None else None,
                }

                if graph_ablation is not None:
                    row["graph_ablation"] = graph_ablation
                if num_edges is not None:
                    row["num_edges"] = int(num_edges)
                if graph_homophily is not None:
                    row["graph_homophily"] = float(graph_homophily)

                return row

            self.results_log.extend([
                _mk_row(
                    metrics=metrics_lstm,
                    model="LSTM",
                    direction=dir_l_safe,
                    confidence_pct=conf_l,
                    raw_score=prob_l,
                    decision_threshold=thr_l,
                    energy_Wh=lstm_energy_Wh,
                    energy_per_sample_Wh=lstm_energy_per_sample_Wh,
                    train_seconds=lstm_train_secs,
                    avg_power_W=lstm_avg_power_W,
                ),
                _mk_row(
                    metrics=metrics_gru,
                    model="GRU",
                    direction=dir_g_safe,
                    confidence_pct=conf_g,
                    raw_score=prob_g,
                    decision_threshold=thr_g,
                    energy_Wh=gru_energy_Wh,
                    energy_per_sample_Wh=gru_energy_per_sample_Wh,
                    train_seconds=gru_train_secs,
                    avg_power_W=gru_avg_power_W,
                ),
                _mk_row(
                    metrics=metrics_stgnn,
                    model="STGNN",
                    direction=dir_s_safe,
                    confidence_pct=conf_s,
                    raw_score=prob_s,
                    decision_threshold=thr_s,
                    energy_Wh=stgnn_energy_Wh,
                    energy_per_sample_Wh=stgnn_energy_per_sample_Wh,
                    train_seconds=stgnn_train_secs,
                    avg_power_W=stgnn_avg_power_W,
                    graph_ablation=getattr(self.args, "graph_ablation", "none"),
                    num_edges=int(self.init_edge_index.size(1)) if getattr(self, "init_edge_index", None) is not None else None,
                    graph_homophily=float(getattr(self, "graph_homophily", float("nan"))),
                ),
            ])


            if hasattr(self, "graphBuilder"):
                del self.graphBuilder
                self.graphBuilder = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        engineered = ["return", "volatility", "momentum"]
        if getattr(self.args, "ablate_feature", "none") in engineered:
            engineered = [f for f in engineered if f != self.args.ablate_feature]
        feature_cols = ["close"] + engineered

        return TensorFactory(
            tickers=self.args.tickers,
            featureCols=feature_cols,
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

        rewiring_options = [False]  # comparative paper: keep fixed by default

        # Main suite: identity is the key self-loop ablation; keep "empty" optional
        graph_ablation_options = ["none", "identity"]
        if bool(getattr(self.args, "include_empty_ablation", False)):
            graph_ablation_options.append("empty")

        # Ablated condition (none | pca | one engineered feature)
        # "pca" means: remove PCA -> graph_embed="raw"
        # feature means: remove exactly that engineered feature everywhere -> ablate_feature=<feature>
        ablated_options = ["none", "pca", "return", "volatility", "momentum"]

        max_k_values = [1,2,3,5,8]  # sparsity sweep; adjust as needed
        seq_len_values = [32]  # approximately 1 week of hourly stock market data

        # Multi-seed controls (repetitions == number of seeds)
        num_seeds = int(getattr(self.args, "num_seeds", 1))
        explicit_seeds = getattr(self.args, "seeds", None)

        param_grid = list(itertools.product(
            rewiring_options, graph_ablation_options, ablated_options, max_k_values, seq_len_values
        ))

        os.makedirs(self.args.results_dir, exist_ok=True)
        all_results = []
        config_id = 0

        for (rewire, graph_abl, ablated, k, seq_len) in param_grid:
            config_id += 1
            exp_name = (
                f"{self.args.experiment_name}"
                f"_rewire-{int(rewire)}"
                f"_graphabl-{graph_abl}"
                f"_ablated-{ablated}"
                f"_k-{k}"
                f"_seq-{seq_len}"
            )
            results_csv = os.path.join(
                self.args.results_dir,
                f"{self.args.experiment_name}.csv"
            )


            self.logger.info(
                f"[AutoTest] Starting config: rewiring={rewire}, graph_ablation={graph_abl}, "
                f"ablated={ablated}, max_k={k}, seq_len={seq_len}"
            )

            for stock in self.args.tickers:
                stock_results = []
                self.logger.info(f"[AutoTest] Running stock: {stock}")

                stock_base_seed = int(self.args.base_seed) % (2**32)

                # Seeds for this stock/config (aligned across models/ablations)
                if explicit_seeds is not None and len(explicit_seeds) > 0:
                    seeds = [int(s) % (2**32) for s in explicit_seeds]
                else:
                    seeds = [(stock_base_seed + i) % (2**32) for i in range(num_seeds)]

                num_reps = len(seeds)

                for rep, run_seed in enumerate(seeds):
                    # 1) Apply config for this run
                    cfg = copy.deepcopy(self.args)
                    cfg.rewiring = rewire
                    cfg.graph_ablation = graph_abl
                    cfg.max_k = k
                    cfg.seq_len = seq_len

                    # Map "ablated" -> graph_embed + ablate_feature
                    # Defaults
                    cfg.graph_embed = getattr(cfg, "graph_embed", "pca")
                    cfg.ablate_feature = getattr(cfg, "ablate_feature", "none")

                    if ablated == "pca":
                        # PCA ablation: remove PCA by switching embedding to "raw"
                        cfg.graph_embed = "raw"
                        cfg.ablate_feature = "none"
                    elif ablated in ("return", "volatility", "momentum"):
                        # Feature ablation: remove one engineered feature everywhere
                        cfg.graph_embed = "pca"  # keep PCA in place unless explicitly ablated
                        cfg.ablate_feature = ablated
                    else:
                        # "none"
                        cfg.graph_embed = "pca"
                        cfg.ablate_feature = "none"

                    self.args = cfg

                    # 2) Set seed and stamp it
                    self._set_all_seeds(run_seed)
                    self.current_seed = run_seed

                    stop_event = threading.Event()
                    start_time = time.time()

                    try:
                        self.logger.info(
                            f"[AutoTest] Run {rep+1}/{num_reps} for {stock} (seed={run_seed}) | "
                            f"graph_embed={self.args.graph_embed} | ablate_feature={self.args.ablate_feature}"
                        )

                        # Avoid double-counting from earlier runs
                        self.results_log = []

                        # Inline pipeline run
                        self.startPipeline(gui_window="1d", stock=stock, stop_event=stop_event)

                        # Stamp metadata including the seed used
                        for entry in self.results_log:
                            entry.update({
                                "config_id": config_id,
                                "ticker": stock,
                                "runtime_sec": round(time.time() - start_time, 2),
                                "rep": rep + 1,
                                "seed": self.current_seed,
                                "rewiring": rewire,
                                "graph_ablation": graph_abl,
                                "ablated": ablated,
                                "graph_embed": self.args.graph_embed,  # optional but helpful
                                "ablate_feature": self.args.ablate_feature,  # optional but helpful
                                "max_k": k,
                                "seq_len": seq_len,
                            })
                            stock_results.append(entry)

                    except Exception as e:
                        self.logger.error(f"[AutoTest] {stock} rep {rep+1} failed: {e}")
                        stock_results.append({
                            "ticker": stock,
                            "rewiring": rewire,
                            "graph_ablation": graph_abl,
                            "ablated": ablated,
                            "graph_embed": getattr(self.args, "graph_embed", None),
                            "ablate_feature": getattr(self.args, "ablate_feature", None),
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

                    write_header = not os.path.exists(results_csv)
                    df.to_csv(
                        results_csv,
                        mode="a",
                        header=write_header,
                        index=False
                    )

                    self.logger.info(
                        f"[AutoTest] Appended {len(df)} rows → {results_csv}"
                    )

                    # ---- compute mean and std summary rows across seeds (PER MODEL) ----
                    numeric_cols = df.select_dtypes(include="number").columns

                    if "model" in df.columns:
                        grouped = df.groupby("model")
                    elif "model_name" in df.columns:
                        grouped = df.groupby("model_name")
                    else:
                        grouped = [(None, df)]  # fallback (not ideal but won't crash)

                    for model_key, gdf in grouped:
                        # mean row
                        mean_row = gdf[numeric_cols].mean(numeric_only=True)
                        mean_row["ticker"] = stock
                        mean_row["rewiring"] = rewire
                        mean_row["graph_ablation"] = graph_abl
                        mean_row["ablated"] = ablated
                        mean_row["graph_embed"] = getattr(self.args, "graph_embed", None)
                        mean_row["ablate_feature"] = getattr(self.args, "ablate_feature", None)
                        mean_row["max_k"] = k
                        mean_row["seq_len"] = seq_len
                        mean_row["rep"] = "mean"

                        if model_key is not None:
                            if "model" in df.columns:
                                mean_row["model"] = model_key
                            else:
                                mean_row["model_name"] = model_key

                        # carry identifiers that should not be averaged
                        for col in ["model_name", "model", "params"]:
                            if col in gdf.columns:
                                first = gdf[col].dropna()
                                mean_row[col] = first.iloc[0] if len(first) else None

                        # num_edges: keep first non-null if present
                        if "num_edges" in gdf.columns:
                            ne = gdf["num_edges"].dropna()
                            mean_row["num_edges"] = int(ne.iloc[0]) if len(ne) else np.nan

                        # graph_homophily: keep first non-null if present
                        if "graph_homophily" in gdf.columns:
                            gh = gdf["graph_homophily"].dropna()
                            mean_row["graph_homophily"] = float(gh.iloc[0]) if len(gh) else np.nan

                        all_results.append(mean_row.to_dict())

                        # std row (sample std)
                        std_series = gdf[numeric_cols].std(numeric_only=True, ddof=1)
                        std_row = std_series.copy()
                        std_row["ticker"] = stock
                        std_row["rewiring"] = rewire
                        std_row["graph_ablation"] = graph_abl
                        std_row["ablated"] = ablated
                        std_row["graph_embed"] = getattr(self.args, "graph_embed", None)
                        std_row["ablate_feature"] = getattr(self.args, "ablate_feature", None)
                        std_row["max_k"] = k
                        std_row["seq_len"] = seq_len
                        std_row["rep"] = "std"

                        if model_key is not None:
                            if "model" in df.columns:
                                std_row["model"] = model_key
                            else:
                                std_row["model_name"] = model_key

                        for col in ["model_name", "model", "params"]:
                            if col in gdf.columns:
                                first = gdf[col].dropna()
                                std_row[col] = first.iloc[0] if len(first) else None

                        # num_edges for std row: same safe logic
                        if "num_edges" in gdf.columns:
                            ne = gdf["num_edges"].dropna()
                            std_row["num_edges"] = int(ne.iloc[0]) if len(ne) else np.nan

                        # graph_homophily for std row: same safe logic (optional but consistent)
                        if "graph_homophily" in gdf.columns:
                            gh = gdf["graph_homophily"].dropna()
                            std_row["graph_homophily"] = float(gh.iloc[0]) if len(gh) else np.nan

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
        