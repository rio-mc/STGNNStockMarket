import gc
import logging
import os
import platform
import threading

import numpy as np
import pandas as pd
import torch

from front_end import FrontEnd
from raw_data_handler import RawDataHandler
from feature_extractor import FeatureExtractor
from data.tensor_factory import TensorFactory
from graph_builder import GraphBuilder
from loading_overlay import LoadingOverlay
from Utils import Utils
from config_manager import ConfigManager

from models import ModelRegistry

# cuBLAS determinism
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


class MainApp:
    """
    Orchestrates loading, preprocessing, graph-building, config,
    and delegates per-model execution to runners in models/.
    """

    def __init__(self):
        # ====================================
        # === STEP 1: Configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = ConfigManager.parseArgs()
        self.args.interval = "1h"

        self.args.base_seed = getattr(self.args, "base_seed", 42)
        self.args.deterministic = getattr(self.args, "deterministic", False)

        self.graph_homophily = float("nan")
        self.pipeline_running = False
        self.data_ready = threading.Event()

        # ====================================
        # === STEP 2: Logging
        self.logger = logging.getLogger("MainApp")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self._set_all_seeds()
        self.results_log = []

        # ====================================
        # === STEP 3: Front-end
        self.args.tickers = sorted(self.args.tickers)
        self.frontendApp = FrontEnd(self.args.tickers)

        avi_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "loading.avi"
        )
        self.loader = LoadingOverlay(self.frontendApp.root, avi_path, delay=24)

    def _load_data_and_start_gui(self):
        # ====================================
        # === STEP 1: Establish feature columns
        engineered = ["return", "volatility", "momentum"]

        if self.args.ablate_feature != "none":
            engineered = [f for f in engineered if f != self.args.ablate_feature]

        self.raw_feature_cols = ["close"] + engineered

        self.logger.info(
            f"[Ablation] ablate_feature={self.args.ablate_feature} | "
            f"node_raw_feature_cols={self.raw_feature_cols}"
        )

        def background_task():
            # ====================================
            # === STEP 2: Load raw price history
            self.priceHistory, raw_handler = self._load_price_history(
                self.args.tickers,
                return_handler=True
            )

            valid_tickers = raw_handler.listTickers()
            dropped_tickers = [t for t in self.args.tickers if t not in valid_tickers]
            if dropped_tickers:
                self.logger.warning(
                    f"Dropped tickers (no valid data): {', '.join(dropped_tickers)}"
                )

            self.args.tickers = valid_tickers

            # ====================================
            # === STEP 3: Build feature DF map
            self.raw_feature_dfs = {
                t: self.priceHistory[t] for t in valid_tickers
            }

            if not self.raw_feature_dfs:
                raise RuntimeError("No usable stock data was loaded.")

            # ====================================
            # === STEP 4: Bind front-end
            self.frontendApp.bindMainApp(self)
            self.loader.trigger_fade_and_destroy()

            self.frontendApp.bindTickerClick(self.frontendApp.on_ticker_click)
            self.frontendApp.setComputeCallback(self.startPipeline)

            self.data_ready.set()

        threading.Thread(target=background_task, daemon=True).start()

    def startPipeline(self, gui_window: str, stock: str, stop_event: threading.Event):
        self.logger.info(f"[Env] Python={platform.python_version()}")
        self.logger.info(f"[Env] PyTorch={torch.__version__}")
        self.logger.info(f"[Env] CUDA available={torch.cuda.is_available()}")
        self.logger.info(f"[Env] CUDA version={torch.version.cuda}")
        self.logger.info(f"[Env] cuDNN={torch.backends.cudnn.version()}")
        self.logger.info(f"[Env] NumPy={np.__version__}")

        if self.pipeline_running:
            self.logger.warning("Pipeline already running.")
            return ("-", 0.0, "-", 0.0, "-", 0.0)

        self.pipeline_running = True

        try:
            torch.use_deterministic_algorithms(self.args.deterministic)
            torch.backends.cudnn.deterministic = self.args.deterministic
            torch.backends.cudnn.benchmark = not self.args.deterministic

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # ====================================
            # === STEP 1: Sector map and UI reset
            sector_map_path = "tickers.csv"
            ticker_to_sector = Utils.load_ticker_to_sector(sector_map_path)
            self.frontendApp.set_sector_map(ticker_to_sector)

            missing = [t for t in self.args.tickers if t not in ticker_to_sector]
            if missing:
                self.logger.warning(
                    "[SectorMap] %d tickers missing sector labels: %s",
                    len(missing),
                    missing[:5]
                )

            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.frontendApp.set_status("Starting...")

            evaluator = self.frontendApp.evaluator
            evaluator.reset_histories()

            price_df = self.raw_feature_dfs[stock]

            # ====================================
            # === STEP 2: Determine horizon
            deltas = price_df.index.to_series().diff().dropna()
            if deltas.empty:
                raise RuntimeError("Not enough data to infer sampling interval")

            mode = deltas.mode()
            sample_interval = mode.iloc[0] if not mode.empty else deltas.median()
            bars_per_day = max(1, int(pd.Timedelta("1D") / sample_interval))

            horizon = Utils.parse_window(gui_window, bars_per_day)
            self.horizon = horizon
            self.seq_len = self.args.seq_len

            self._check_stop(stop_event)

            # ====================================
            # === STEP 3: Raw train/validation split
            self.frontendApp.set_status("Preparing raw data...")

            shared_index = set.intersection(
                *(set(df.index) for df in self.raw_feature_dfs.values())
            )
            shared_index = sorted(shared_index)

            cutoff_idx = int(0.7 * len(shared_index))
            cutoff_date = shared_index[cutoff_idx]

            embargo_bars = max(1, int(self.horizon))
            train_end_idx = max(0, cutoff_idx - embargo_bars)
            train_end_date = shared_index[train_end_idx]

            train_raw_map = {
                t: df[df.index < train_end_date].copy()
                for t, df in self.raw_feature_dfs.items()
            }
            val_raw_map = {
                t: df[df.index >= cutoff_date].copy()
                for t, df in self.raw_feature_dfs.items()
            }

            self._check_stop(stop_event)

            # ====================================
            # === STEP 4: Feature engineering
            self.frontendApp.set_status("Extracting engineered features...")

            feat_ext_train = FeatureExtractor(
                train_raw_map,
                rollingVolWindow=self.seq_len,
                norm_stats=None,
                fit_normaliser=True,
                ablate_feature=self.args.ablate_feature,
            )
            feat_ext_train.buildFeatureDfs()
            self.train_feats = feat_ext_train.dfFeats
            train_norm_stats = feat_ext_train.get_norm_stats()

            feat_ext_val = FeatureExtractor(
                val_raw_map,
                rollingVolWindow=self.seq_len,
                norm_stats=train_norm_stats,
                fit_normaliser=False,
                ablate_feature=self.args.ablate_feature,
            )
            feat_ext_val.buildFeatureDfs()
            self.val_feats = feat_ext_val.dfFeats

            self.min_train_len = min(len(df) for df in self.train_feats.values())
            self.min_val_len = min(len(df) for df in self.val_feats.values())

            self.train_feats = {
                t: df.iloc[-self.min_train_len:].copy()
                for t, df in self.train_feats.items()
            }
            self.val_feats = {
                t: df.iloc[-self.min_val_len:].copy()
                for t, df in self.val_feats.items()
            }

            self._check_stop(stop_event)

            # ====================================
            # === STEP 5: Tensor factories
            self.frontendApp.set_status("Creating tensors...")
            self.tf_train = self.build_tensor_factory(horizon=self.horizon)
            self.tf_val = self.build_tensor_factory(horizon=self.horizon)

            self._check_stop(stop_event)

            # ====================================
            # === STEP 6: Graph building
            self.frontendApp.set_status("Build the graph...")

            self.graphBuilder = GraphBuilder(
                dfFeats=self.train_feats,
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

            tickers, coords3d, pruned, mst = self.graphBuilder.getLightGraph()
            graph_nx = self.graphBuilder.buildNetworkX(tickers, coords3d, pruned)

            graph_mode = getattr(self.args, "graph_ablation", "none")
            num_nodes = len(tickers)

            pruned_for_plot = pruned
            mst_for_plot = mst
            if graph_mode in ("identity", "empty"):
                pruned_for_plot = []
                mst_for_plot = []

            self.frontendApp.root.after(
                0,
                lambda: self.frontendApp.plot3d_on_ax(
                    tickers=tickers,
                    coords=coords3d,
                    pruned_edges=pruned_for_plot,
                    mst_edges=mst_for_plot,
                )
            )

            edge_pairs = [(i, j) for i, j, _ in pruned]
            if edge_pairs:
                edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            if edge_index.numel() == 0:
                idx = torch.arange(num_nodes, dtype=torch.long)
                edge_index = torch.stack([idx, idx], dim=0)
                self.logger.info(
                    "[Graph] Empty edge_index after pruning. Using identity self-loops."
                )

            rev = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, rev], dim=1)

            key = edge_index[0] * num_nodes + edge_index[1]
            uniq = torch.unique(key, sorted=True)
            edge_index = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0)

            edge_index = Utils.apply_graph_ablation(
                edge_index,
                num_nodes=num_nodes,
                mode=graph_mode
            )

            self.logger.info(
                f"[AblationCheck] mode={graph_mode} | V={num_nodes} | "
                f"E={edge_index.size(1)} | "
                f"all_self_loops={(edge_index.size(1) > 0 and (edge_index[0] == edge_index[1]).all().item())}"
            )

            requested_k = self.get_max_k()
            effective_k = getattr(self.graphBuilder, "effective_k", requested_k)

            self.init_edge_index = edge_index.clone()
            self.graphBuilder.edge_index = self.init_edge_index

            try:
                self.graph_homophily = self.graphBuilder.sector_homophily_from_edge_index(
                    tickers=tickers,
                    edge_index=self.init_edge_index,
                    ignore_unknown=True,
                    ignore_self_loops=True,
                )
                self.logger.info(
                    f"[Graph] sector_homophily={self.graph_homophily:.4f} "
                    "(ignore_unknown=True, ignore_self_loops=True)"
                )
            except Exception as exc:
                self.logger.warning(f"[Graph] homophily computation failed: {exc}")
                self.graph_homophily = float("nan")

            num_edges = edge_index.size(1)
            possible_directed_no_self = num_nodes * (num_nodes - 1)
            graph_density = (
                num_edges / possible_directed_no_self
                if possible_directed_no_self > 0 else 0.0
            )

            self.logger.info(
                f"[Graph] ablation={graph_mode} requested_k={requested_k} "
                f"effective_k={effective_k} |V|={num_nodes} |E|={num_edges} "
                f"density={graph_density:.6f}"
            )

            Utils.log_graph_memory(graph_nx, coords3d, edge_index, tag="Initial")

            latest_feats = []
            for t in self.args.tickers:
                df = self.train_feats.get(t)
                if df is not None and len(df) > 0:
                    latest_row = df.iloc[-1]
                    latest_feats.append({
                        "return": latest_row.get("return", np.nan),
                        "volatility": latest_row.get("volatility", np.nan),
                        "volume": latest_row.get("volume", np.nan),
                        "momentum": latest_row.get("momentum", np.nan),
                    })

            node_df = pd.DataFrame(latest_feats, index=self.args.tickers)
            node_df.index = node_df.index.astype(str)
            self.frontendApp.root.after(0, lambda: self.frontendApp.updateTable(node_df))

            self._check_stop(stop_event)

            # ====================================
            # === STEP 7: Seeded DataLoader generator
            self.dl_gen = torch.Generator(device="cpu")
            seed_for_loaders = getattr(
                self,
                "current_seed",
                int(self.args.base_seed)
            ) % (2 ** 32)
            self.dl_gen.manual_seed(seed_for_loaders)

            # ====================================
            # === STEP 8: Run model runners
            selected_model = self.frontendApp.get_selected_model()
            runner = ModelRegistry.get_runner(selected_model)

            result = runner.run(
                self,
                stock=stock,
                price_df=price_df,
                evaluator=evaluator,
                stop_event=stop_event,
            )

            self._check_stop(stop_event)

            self.frontendApp.root.after(
                0,
                lambda: self.frontendApp.updateResults(
                    result.model_name,
                    result.direction,
                    result.confidence,
                )
            )

            return (
                result.model_name,
                result.direction,
                result.confidence,
            )

        except InterruptedError:
            self.logger.info("[startPipeline] Pipeline interrupted by stop_event")
            self.frontendApp.root.after_idle(self.frontendApp._reset_ui)
            self.pipeline_running = False
            return (self.frontendApp.modelVar.get(), "-", 0.0)
    
        except Exception:
            self.logger.exception("Pipeline failed")
            raise

        finally:
            self.pipeline_running = False
            self.frontendApp.btnCompute.config(state="normal", text="Compute ▶")
            self.frontendApp.btnStop.config(state="disabled")
            self.frontendApp.clear_status()

    def run(self):
        # ====================================
        # === Helper to load data and run main loop
        threading.Thread(target=self._load_data_and_start_gui, daemon=True).start()
        self.frontendApp.root.mainloop()

    def _load_price_history(self, tickers, period="729d", return_handler=False):
        # ====================================
        # === Helper to pass raw data from API
        rd = RawDataHandler(
            source=self.args.mode,
            tickers=tickers,
            dataDir=self.args.data_dir,
            avApiKey=self.args.av_key,
            yfPeriod=period,
            yfInterval=self.args.interval,
        )
        data = {t: rd.getDataframe(t) for t in tickers}
        return (data, rd) if return_handler else data

    def _check_stop(self, stop_event: threading.Event):
        # ====================================
        # === Helper to terminate all processes
        if stop_event and stop_event.is_set():
            self.logger.info("[MainApp] Pipeline stopped via stop_event.")
            raise InterruptedError("Pipeline interrupted")

    def get_max_k(self):
        # ====================================
        # === Helper to control graph efficiency
        return self.args.max_k

    def build_edge_index(self, coords, edges, *_):
        # ====================================
        # === Helper to build edge indices from graph
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
            prediction_horizon=horizon,
            device="cpu",
        )

    def _set_all_seeds(self, run_seed: int | None = None):
        """
        Set all RNGs and remember the current run seed.
        """
        seed = int(run_seed) if run_seed is not None else int(self.args.base_seed)
        used = Utils.set_seed(seed, deterministic=self.args.deterministic)
        self.current_seed = int(used)
        self.logger.info(
            f"[Seed] Using seed={self.current_seed} "
            f"(deterministic={self.args.deterministic})"
        )

    def _seed_worker(self, worker_id: int):
        """
        Ensure each DataLoader worker has a distinct, reproducible seed.
        """
        worker_seed = (self.current_seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)

        import random as _random
        _random.seed(worker_seed)

    def run_experiments(self):
        """
        Placeholder for experiment sweeps.
        Keep existing implementation if you still need it.
        """
        self.data_ready.wait()
        self.logger.info("run_experiments() not yet refactored into the new model-runner flow.")

if __name__ == "__main__":
    app = MainApp()
    app.run()