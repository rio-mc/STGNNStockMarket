import math
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Tuple, Optional

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from core.job_queue import (
    QueueJob,
    JobQueueController,
    parse_seed_spec,
    parse_ticker_spec,
)
from core.utils.utils import Utils
from evaluation.evaluation_methods import EvaluationMethods


class Cancelled(Exception):
    """Raised when training is aborted by user."""
    pass


class FrontEnd:
    """
    Handles front-end logic for a single selected-model run.
    """

    def __init__(self, availableTickers, project_root=None):
        self.root = tk.Tk()
        self.root.title("Stock Trend Explorer")
        self.root.geometry("1650x1050")
        self.root.minsize(1300, 900)

        # Store project root for custom universe imports
        from pathlib import Path
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent

        # ------------------------------------------------------------------
        # Global state used by callbacks and async loading
        # ------------------------------------------------------------------
        self.stop_event = threading.Event()
        self.statusVar = tk.StringVar(value="Idle")
        self._background_busy = False
        self._busy_controls = []

        self._compute_callback = None
        self._trainers = {}
        self._main_app = None

        self.priceHistory = {}
        self.currentWindowIdx = 0

        self._last_drawn_tickers = []
        self._last_drawn_pos = {}
        self._last_pruned_edges = []
        self._last_mst_edges = []
        self._last_edge_labels = []
        self._sector_to_colour = {}
        self._highlight_markers = []
        self.ticker_to_sector = {}
        self._selected_ticker = None

        self._queue_add_callback = None
        self._queue_run_callback = None
        self._queue_remove_callback = None
        self._queue_clear_callback = None
        self._close_callback = None
        self._closing = False

        self.queue_popout = None
        self.queue_popout_table = None
        self.queueGraphModelMenu = None

        # ------------------------------------------------------------------
        # Notebook
        # ------------------------------------------------------------------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.mainTab = tk.Frame(self.notebook)
        self.notebook.add(self.mainTab, text="Main")

        # ------------------------------------------------------------------
        # Top control area: Configuration / Run / Queue
        # ------------------------------------------------------------------
        self.toolbar = tk.Frame(self.mainTab, bd=1, relief=tk.RAISED)
        self.toolbar.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.configGroup = tk.LabelFrame(self.toolbar, text="Configuration", padx=8, pady=6)
        self.configGroup.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.runGroup = tk.LabelFrame(self.toolbar, text="Run", padx=8, pady=6)
        self.runGroup.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))

        self.queueGroup = tk.LabelFrame(self.toolbar, text="Queue", padx=8, pady=6)
        self.queueGroup.pack(side=tk.RIGHT, fill=tk.Y)

        # Configuration group: compact grid instead of one long toolbar.
        for col in range(12):
            self.configGroup.columnconfigure(col, weight=0)
        self.configGroup.columnconfigure(1, weight=1)
        self.configGroup.columnconfigure(3, weight=1)
        self.configGroup.columnconfigure(5, weight=1)
        self.configGroup.columnconfigure(7, weight=1)

        tk.Label(self.configGroup, text="Prediction range").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.windowValues = ["1d", "2d", "5d", "1w"]
        self.windowOptions = []
        self.windowVar = tk.StringVar(value=self.windowValues[0])
        self.windowMenu = ttk.Combobox(
            self.configGroup,
            textvariable=self.windowVar,
            values=self.windowValues,
            width=7,
            state="readonly",
        )
        self.windowMenu.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.windowMenu.bind("<<ComboboxSelected>>", self._onSelectionChange)

        tk.Label(self.configGroup, text="Stock(s)").grid(row=0, column=1, sticky="w", padx=(0, 4))
        self.stockVar = tk.StringVar(value=availableTickers[0] if availableTickers else "")
        self.stockMenu = ttk.Combobox(
            self.configGroup,
            textvariable=self.stockVar,
            values=availableTickers,
            width=26,
            state="normal",
        )
        self.stockMenu.grid(row=1, column=1, sticky="ew", padx=(0, 10))
        self.stockMenu.bind("<<ComboboxSelected>>", self._onSelectionChange)

        tk.Label(self.configGroup, text="Universe").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.universeValues = ["S&P 500", "NASDAQ 100", "Custom"]
        self.universeVar = tk.StringVar(value=self.universeValues[0])
        self.universeMenu = ttk.Combobox(
            self.configGroup,
            textvariable=self.universeVar,
            values=self.universeValues,
            width=13,
            state="readonly",
        )
        self.universeMenu.grid(row=1, column=2, sticky="ew", padx=(0, 4))
        self.universeMenu.bind("<<ComboboxSelected>>", self._on_universe_change)

        self.btnImport = tk.Button(self.configGroup, text="Import CSV", command=self._on_import_csv)
        self.btnImport.grid(row=1, column=3, sticky="w", padx=(0, 10))
        self.btnImport.configure(state=tk.DISABLED)

        tk.Label(self.configGroup, text="Model").grid(row=0, column=4, sticky="w", padx=(0, 4))
        self.modelValues = [
            "LSTM",
            "GRU",
            "PANEL_GRU",
            "PANEL_LSTM",
            "GCN",
            "GAT",
            "NNCONV",
            "GRAPHSAGE",
            "STGNN",
        ]
        self.modelVar = tk.StringVar(value=self.modelValues[0])
        self.modelMenu = ttk.Combobox(
            self.configGroup,
            textvariable=self.modelVar,
            values=self.modelValues,
            width=13,
            state="readonly",
        )
        self.modelMenu.grid(row=1, column=4, sticky="ew", padx=(0, 10))
        self.modelMenu.bind("<<ComboboxSelected>>", self._on_model_selection_change)

        tk.Label(self.configGroup, text="Seed(s)").grid(row=0, column=5, sticky="w", padx=(0, 4))
        self.seedVar = tk.StringVar(value="42")
        self.seedEntry = ttk.Entry(self.configGroup, textvariable=self.seedVar, width=14)
        self.seedEntry.grid(row=1, column=5, sticky="ew", padx=(0, 10))

        tk.Label(self.configGroup, text="Graph backend").grid(row=0, column=6, sticky="w", padx=(0, 4))
        self.graphModelValues = ["GCN", "GRAPHSAGE", "GAT", "NNCONV"]
        self.graphModelVar = tk.StringVar(value=self.graphModelValues[0])
        self.graphModelMenu = ttk.Combobox(
            self.configGroup,
            textvariable=self.graphModelVar,
            values=self.graphModelValues,
            width=13,
            state="readonly",
        )
        self.graphModelMenu.grid(row=1, column=6, sticky="ew", padx=(0, 10))

        # Advanced experiment controls are shown in the queue popout but are
        # kept as shared variables so queued and direct GUI runs use one source.
        self.kVar = tk.StringVar(value="3")
        self.graphModeValues = ["knn_mst", "knn", "mst"]
        self.graphModeVar = tk.StringVar(value="knn_mst")
        self.graphEmbedValues = ["pca", "raw"]
        self.graphEmbedVar = tk.StringVar(value="pca")
        self.graphAblationValues = ["none", "identity", "empty"]
        self.graphAblationVar = tk.StringVar(value="none")
        self.ablateFeatureValues = ["none", "return", "volatility", "momentum"]
        self.ablateFeatureVar = tk.StringVar(value="none")
        self.seqLenVar = tk.StringVar(value="10")
        self.batchSizeVar = tk.StringVar(value="256")
        self.lstmEpochsVar = tk.StringVar(value="200")
        self.stgnnEpochsVar = tk.StringVar(value="200")

        # Run group
        self.btnCompute = tk.Button(self.runGroup, text="Compute ▶", command=self._onCompute, width=14)
        self.btnCompute.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self.btnStop = tk.Button(self.runGroup, text="Stop ■", command=self._onStop, state=tk.DISABLED, width=12)
        self.btnStop.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        self.progressVar = tk.DoubleVar(value=0.0)
        self.progressBar = ttk.Progressbar(
            self.runGroup,
            variable=self.progressVar,
            maximum=1.0,
            length=210,
        )
        self.progressBar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self.statusLabel = tk.Label(
            self.runGroup,
            textvariable=self.statusVar,
            anchor="w",
            fg="gray",
            font=("Arial", 9),
        )
        self.statusLabel.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Queue group
        self.btnQueueAdd = tk.Button(self.queueGroup, text="Add to queue +", command=self._onAddToQueue, width=14)
        self.btnQueueAdd.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self.btnRunQueue = tk.Button(self.queueGroup, text="Run queue ▷", command=self._onRunQueue, width=14)
        self.btnRunQueue.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        self.btnQueuePopout = tk.Button(self.queueGroup, text="Queue popout", command=self._open_queue_popout, width=14)
        self.btnQueuePopout.grid(row=1, column=0, columnspan=2, sticky="ew")

        # ------------------------------------------------------------------
        # Queue strip: secondary, compact, still always available
        # ------------------------------------------------------------------
        self.queueFrame = tk.LabelFrame(self.mainTab, text="Prediction Queue", padx=6, pady=6)
        self.queueFrame.pack(fill=tk.X, padx=10, pady=(4, 4))

        queue_body = tk.Frame(self.queueFrame)
        queue_body.pack(fill=tk.X, expand=True)

        queue_cols = ("Job ID", "Window", "Ticker", "Model", "Seed", "Graph", "k", "Mode", "Embed", "Ablation", "Seq")
        self.queueTable = ttk.Treeview(self.queueFrame, columns=queue_cols, show="headings", height=4)
        for c in queue_cols:
            self.queueTable.heading(c, text=c)
            self.queueTable.column(c, width=90 if c in {"k", "Seq"} else 110, anchor=tk.CENTER)
        self.queueTable.pack(side=tk.LEFT, fill=tk.X, expand=True)

        queue_scroll = ttk.Scrollbar(self.queueFrame, orient=tk.VERTICAL, command=self.queueTable.yview)
        queue_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.queueTable.configure(yscrollcommand=queue_scroll.set)

        self.queueButtons = tk.Frame(self.queueFrame)
        self.queueButtons.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        tk.Button(self.queueButtons, text="Remove", command=self._onRemoveQueueItem).pack(fill=tk.X, pady=2)
        tk.Button(self.queueButtons, text="Clear", command=self._onClearQueue).pack(fill=tk.X, pady=2)

        # ------------------------------------------------------------------
        # Main workspace: top = price + prediction, bottom = graph + table
        # ------------------------------------------------------------------
        self.workspace = tk.PanedWindow(self.mainTab, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        self.workspace.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.topWorkspace = tk.PanedWindow(self.workspace, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.workspace.add(self.topWorkspace, minsize=260)

        self.priceFrame = tk.LabelFrame(self.topWorkspace, text="Price history", padx=4, pady=4)
        self.topWorkspace.add(self.priceFrame, stretch="always", minsize=650)

        # Dedicated layout container so the chart cannot consume the slider's space
        self.priceFrame.rowconfigure(0, weight=1)   # chart expands
        self.priceFrame.rowconfigure(1, weight=0)   # slider fixed
        self.priceFrame.columnconfigure(0, weight=1)

        self.priceChartFrame = tk.Frame(self.priceFrame)
        self.priceChartFrame.grid(row=0, column=0, sticky="nsew")

        self.priceChartFrame.rowconfigure(0, weight=1)
        self.priceChartFrame.columnconfigure(0, weight=1)

        self.priceFig, self.priceAx = plt.subplots(figsize=(8, 2.8))
        self.priceCanvas = FigureCanvasTkAgg(self.priceFig, master=self.priceChartFrame)
        self.priceCanvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.priceCanvas.mpl_connect("scroll_event", self._onPriceScroll)

        self.zoomSliderFrame = tk.Frame(self.priceFrame)
        self.zoomSliderFrame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.zoomSliderFrame.columnconfigure(0, weight=1)

        self.zoomSlider = tk.Scale(
            self.zoomSliderFrame,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            length=600,
            showvalue=False,
            command=self._onSliderChange,
        )
        self.zoomSlider.grid(row=0, column=0, sticky="ew", padx=10)

        self.resultFrame = tk.Frame(self.topWorkspace)
        self.topWorkspace.add(self.resultFrame, minsize=280)

        self.modelRes = tk.LabelFrame(self.resultFrame, text="Prediction Summary", padx=14, pady=14)
        self.modelRes.pack(fill=tk.BOTH, expand=True)

        self.modelNameLabel = tk.Label(
            self.modelRes,
            text=f"Model: {self.modelVar.get()}",
            font=("Helvetica", 10),
            anchor="w",
        )
        self.modelNameLabel.pack(fill=tk.X, anchor="w")

        self.modelStatus = tk.Label(
            self.modelRes,
            text="Trending: —",
            font=("Helvetica", 18, "bold"),
            fg="black",
            anchor="w",
        )
        self.modelStatus.pack(fill=tk.X, anchor="w", pady=(12, 4))

        self.modelConf = tk.Label(
            self.modelRes,
            text="Confidence: —",
            font=("Helvetica", 11),
            anchor="w",
        )
        self.modelConf.pack(fill=tk.X, anchor="w")

        self.summaryHelp = tk.Label(
            self.modelRes,
            text="Run a single-stock compute job or use the queue for multi-stock runs.",
            fg="gray",
            justify=tk.LEFT,
            wraplength=260,
            anchor="w",
        )
        self.summaryHelp.pack(fill=tk.X, anchor="w", pady=(18, 0))

        self.bottomWorkspace = tk.PanedWindow(self.workspace, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.workspace.add(self.bottomWorkspace, minsize=420)

        self.graphFrame = tk.LabelFrame(self.bottomWorkspace, text="Graph Output")
        self.bottomWorkspace.add(self.graphFrame, stretch="always", minsize=650)

        self.graph_container = tk.Frame(self.graphFrame)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        self.graph_canvas_frame = tk.Frame(self.graph_container)
        self.graph_canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.graphFig = Figure(figsize=(6, 4), dpi=100)
        self.graphAx = self.graphFig.add_subplot(111, projection="3d")
        self.graphCanvas = FigureCanvasTkAgg(self.graphFig, master=self.graph_canvas_frame)
        self.graphCanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.graphCanvas.mpl_connect("key_press_event", self._on_graph_keypress)

        self.legend_frame = tk.LabelFrame(self.graph_container, text="Sector Legend", padx=8, pady=8)
        self.legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(self.legend_frame, text="(sector map not loaded)").pack(anchor="w")

        self.table_pane = tk.LabelFrame(self.bottomWorkspace, text="Feature Table", padx=4, pady=4)
        self.bottomWorkspace.add(self.table_pane, minsize=360)

        cols = ("Stock", "Return", "Volatility", "Volume", "Momentum")
        self.table = ttk.Treeview(self.table_pane, columns=cols, show="headings")
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=100, anchor=tk.CENTER)

        self.table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.table.bind("<<TreeviewSelect>>", self._on_table_select)

        scroll = ttk.Scrollbar(self.table_pane, orient=tk.VERTICAL, command=self.table.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scroll.set)

        # ------------------------------------------------------------------
        # Evaluation tab
        # ------------------------------------------------------------------
        self.evalTab = tk.Frame(self.notebook)
        self.notebook.add(self.evalTab, text="Evaluation")

        self.eval_vertical_pane = tk.PanedWindow(self.evalTab, orient=tk.VERTICAL)
        self.eval_vertical_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.eval_summary_frame = tk.LabelFrame(
            self.eval_vertical_pane,
            text="Model Evaluation",
            padx=5,
            pady=5,
        )
        self.eval_vertical_pane.add(self.eval_summary_frame, stretch="always")

        self.eval_pane = tk.PanedWindow(self.eval_summary_frame, orient=tk.VERTICAL)
        self.eval_pane.pack(fill=tk.BOTH, expand=True)

        self.loss_pane_frame = tk.LabelFrame(
            self.eval_vertical_pane,
            text="Training Loss",
            padx=5,
            pady=5,
        )
        self.eval_vertical_pane.add(self.loss_pane_frame, stretch="always")

        self.loss_pane = tk.Frame(self.loss_pane_frame)
        self.loss_pane.pack(fill=tk.BOTH, expand=True)

        # ------------------------------------------------------------------
        # Backtesting tab
        # ------------------------------------------------------------------
        self.backtestTab = ttk.Frame(self.notebook)
        self.notebook.add(self.backtestTab, text="Backtesting")

        self.backtest_frame = tk.LabelFrame(
            self.backtestTab,
            text="Model Backtest",
            padx=5,
            pady=5,
        )
        self.backtest_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.backtest_pane = tk.Frame(self.backtest_frame)
        self.backtest_pane.pack(fill=tk.BOTH, expand=True)

        # ------------------------------------------------------------------
        # Metrics tab
        # ------------------------------------------------------------------
        self.metricsTab = tk.Frame(self.notebook)
        self.notebook.add(self.metricsTab, text="Metrics")

        self.metrics_pane = tk.PanedWindow(self.metricsTab, orient=tk.HORIZONTAL)
        self.metrics_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.metrics_frame = tk.LabelFrame(
            self.metrics_pane,
            text="Model Metrics",
            padx=5,
            pady=5,
        )
        self.metrics_pane.add(self.metrics_frame, stretch="always")

        self.metrics_model_pane = tk.PanedWindow(self.metrics_frame, orient=tk.VERTICAL)
        self.metrics_model_pane.pack(fill=tk.BOTH, expand=True)

        self.roc_pane = tk.LabelFrame(self.metrics_model_pane, text="ROC / PR Curve")
        self.metrics_model_pane.add(self.roc_pane)

        self.threshold_pane = tk.LabelFrame(self.metrics_model_pane, text="Threshold Curve")
        self.metrics_model_pane.add(self.threshold_pane)

        self.set_active_model_titles(self.modelVar.get())
        self._sync_graph_backend_state()

        self.evaluator = EvaluationMethods(self)
        self.evaluator.reset_histories()

        self.root.after(100, self._schedule_rebalance())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.bind(
            "<Configure>",
            lambda e: self._schedule_rebalance() if e.widget == self.root else None,
            add="+"
        )

    def _rebalance_main_workspace(self):
        """Set sensible initial sash positions after widgets have sizes."""
        try:
            self.workspace.update_idletasks()
            h = self.workspace.winfo_height()
            if h > 10:
                self.workspace.sash_place(0, 0, int(h * 0.38))
        except Exception:
            pass

        try:
            self.topWorkspace.update_idletasks()
            w = self.topWorkspace.winfo_width()
            if w > 10:
                self.topWorkspace.sash_place(0, int(w * 0.78), 0)
        except Exception:
            pass

        try:
            self.bottomWorkspace.update_idletasks()
            w = self.bottomWorkspace.winfo_width()
            if w > 10:
                self.bottomWorkspace.sash_place(0, int(w * 0.70), 0)
        except Exception:
            pass

    def ui_call(self, fn, *args, **kwargs):
        if self._closing:
            return
        try:
            self.root.after(0, lambda: fn(*args, **kwargs))
        except (RuntimeError, tk.TclError):
            pass

    def bindMainApp(self, main_app):
        self._main_app = main_app
        self._sync_experiment_controls_from_args()
        ticker = self.stockVar.get()

        for t in self._main_app.args.tickers:
            if t in self._main_app.raw_feature_dfs:
                raw = self._main_app.raw_feature_dfs[t]
                trimmed = self._refresh_ticker_history(t, raw, num_bars=None)
                self.priceHistory[t] = trimmed
            else:
                print(f"[WARNING] Ticker '{t}' not found in raw_feature_dfs.")

        if ticker not in self.priceHistory:
            return

        df_trimmed = self.priceHistory[ticker]
        interval = Utils.infer_interval_label(df_trimmed.index)
        data_len = len(df_trimmed)
        est_days = self.bars_to_days(data_len, interval)
        start = df_trimmed.index[0]
        end = df_trimmed.index[-1]
        duration = end - start

        self.windowOptions = list(range(1, data_len + 1))
        self.currentWindowIdx = data_len - 1
        self.zoomSlider.config(from_=1, to=data_len, resolution=1)
        self.zoomSlider.set(self.currentWindowIdx + 1)

        print(
            f"[DEBUG] {ticker}: {data_len} bars ({interval}) ~= {est_days:.1f} trading days, "
            f"{duration.days} calendar days from {start.date()} to {end.date()} -> windowOptions=1...{data_len}"
        )

        raw_df = self._main_app.raw_feature_dfs[ticker]
        self._refresh_and_plot(ticker, raw_df, num_bars=self.currentWindowIdx + 1)

    def _sync_experiment_controls_from_args(self):
        if self._main_app is None:
            return

        args = self._main_app.args
        self.kVar.set(str(getattr(args, "k", 3)))
        self.graphModeVar.set(str(getattr(args, "graph_mode", "knn_mst")))
        self.graphEmbedVar.set(str(getattr(args, "graph_embed", "pca")))
        self.graphAblationVar.set(str(getattr(args, "graph_ablation", "none")))
        self.ablateFeatureVar.set(str(getattr(args, "ablate_feature", "none")))
        self.seqLenVar.set(str(getattr(args, "seq_len", 10)))
        self.batchSizeVar.set(str(getattr(args, "batch_size", 256)))
        self.lstmEpochsVar.set(str(getattr(args, "lstm_epochs", 200)))
        self.stgnnEpochsVar.set(str(getattr(args, "stgnn_epochs", 200)))

    @staticmethod
    def _parse_positive_int(raw: str, name: str) -> int:
        value = int(str(raw).strip())
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
        return value

    @staticmethod
    def _parse_nonnegative_int(raw: str, name: str) -> int:
        value = int(str(raw).strip())
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
        return value

    def export_experiment_controls(self) -> dict:
        return {
            "k": self._parse_nonnegative_int(self.kVar.get(), "k"),
            "graph_mode": str(self.graphModeVar.get()).strip().lower(),
            "graph_embed": str(self.graphEmbedVar.get()).strip().lower(),
            "graph_ablation": str(self.graphAblationVar.get()).strip().lower(),
            "ablate_feature": str(self.ablateFeatureVar.get()).strip().lower(),
            "seq_len": self._parse_positive_int(self.seqLenVar.get(), "seq_len"),
            "batch_size": self._parse_positive_int(self.batchSizeVar.get(), "batch_size"),
            "lstm_epochs": self._parse_positive_int(self.lstmEpochsVar.get(), "lstm_epochs"),
            "stgnn_epochs": self._parse_positive_int(self.stgnnEpochsVar.get(), "stgnn_epochs"),
        }

    def set_active_model_titles(self, model_name: str):
        labels = {
            "lstm": "LSTM",
            "gru": "GRU",
            "panel_gru": "PANEL GRU",
            "panel_lstm": "PANEL LSTM",
            "gcn": "GCN",
            "gat": "GAT",
            "nnconv": "NNConv",
            "graphsage": "GraphSAGE",
            "stgnn": "STGNN",
        }

        key = str(model_name).strip().lower()
        pretty = labels.get(key, key.upper())

        if key in ("lstm", "gru"):
            self.graphFrame.config(text=f"{pretty} Target Node")
        elif key in ("panel_gru", "panel_lstm"):
            self.graphFrame.config(text=f"{pretty} Panel Nodes (no graph edges used)")
        else:
            self.graphFrame.config(text=f"{pretty} Graph Structure")

        self.modelRes.config(text=f"{pretty} Prediction")
        self.modelNameLabel.config(text=f"Model: {pretty}")
        self.eval_summary_frame.config(text=f"{pretty} Evaluation")
        self.loss_pane_frame.config(text=f"{pretty} Training Loss")
        self.backtest_frame.config(text=f"{pretty} Backtest")
        self.metrics_frame.config(text=f"{pretty} Metrics")
        self.roc_pane.config(text=f"{pretty} ROC / PR Curve")
        self.threshold_pane.config(text=f"{pretty} Threshold Curve")

    def get_selected_model(self) -> str:
        return str(self.modelVar.get()).strip().lower()

    def bindTickerClick(self, callback):
        self.on_ticker_click = callback

    def setComputeCallback(self, cb):
        self._compute_callback = cb

    def setQueueAddCallback(self, cb):
        self._queue_add_callback = cb

    def setQueueRunCallback(self, cb):
        self._queue_run_callback = cb

    def setQueueRemoveCallback(self, cb):
        self._queue_remove_callback = cb

    def setQueueClearCallback(self, cb):
        self._queue_clear_callback = cb

    def setCloseCallback(self, cb):
        self._close_callback = cb

    def _is_graph_backend_applicable(self, model_name: str) -> bool:
        return str(model_name).strip().lower() == "stgnn"

    def _sync_graph_backend_state(self):
        selected_model = str(self.modelVar.get()).strip().lower()
        enabled = self._is_graph_backend_applicable(selected_model)
        self.graphModelMenu.config(state="readonly" if enabled else "disabled")
        if getattr(self, "queueGraphModelMenu", None) is not None:
            try:
                self.queueGraphModelMenu.config(state="readonly" if enabled else "disabled")
            except tk.TclError:
                self.queueGraphModelMenu = None
        if not enabled:
            self.graphModelVar.set("GCN")

    def _parse_selected_tickers(self) -> List[str]:
        seed_raw = str(self.seedVar.get()).strip()
        first_seed = 42
        if seed_raw:
            first_seed = int(seed_raw.split(",")[0].split("-")[0].strip())

        return parse_ticker_spec(
            ticker_spec=str(self.stockVar.get()).strip(),
            available_tickers=list(self.stockMenu.cget("values")),
            rng_seed=first_seed,
        )

    def _build_current_jobs(self) -> List[QueueJob]:
        seed_values = parse_seed_spec(str(self.seedVar.get()).strip())
        ticker_values = self._parse_selected_tickers()

        model_name = str(self.modelVar.get()).strip().lower()
        graph_model = (
            str(self.graphModelVar.get()).strip().lower()
            if self._is_graph_backend_applicable(model_name)
            else "gcn"
        )
        controls = self.export_experiment_controls()

        jobs: List[QueueJob] = []
        for ticker in ticker_values:
            for seed in seed_values:
                jobs.append(
                    QueueJob(
                        job_id=JobQueueController.make_job_id(),
                        created_at="now",
                        prediction_window=str(self.windowVar.get()).strip(),
                        ticker=ticker,
                        model=model_name,
                        seed=int(seed),
                        graph_model=graph_model,
                        k=controls["k"],
                        graph_mode=controls["graph_mode"],
                        graph_embed=controls["graph_embed"],
                        graph_ablation=controls["graph_ablation"],
                        ablate_feature=controls["ablate_feature"],
                        seq_len=controls["seq_len"],
                        batch_size=controls["batch_size"],
                        lstm_epochs=controls["lstm_epochs"],
                        stgnn_epochs=controls["stgnn_epochs"],
                    )
                )
        return jobs

    def _onAddToQueue(self):
        if self._background_busy:
            self.set_status("Please wait until loading finishes.")
            return

        if self._queue_add_callback is None:
            return
        try:
            jobs = self._build_current_jobs()
            self._queue_add_callback(jobs)
            self.set_status(f"Added {len(jobs)} job(s) to queue")
        except Exception as exc:
            self.set_status(f"Queue add failed: {exc}")

    def _onRunQueue(self):
        if self._queue_run_callback is not None:
            self._queue_run_callback()

    def _onRemoveQueueItem(self):
        selected = self.queueTable.selection()
        if not selected or self._queue_remove_callback is None:
            return
        idx = self.queueTable.index(selected[0])
        self._queue_remove_callback(idx)

    def _onRemoveQueuePopoutItem(self):
        if self.queue_popout_table is None or self._queue_remove_callback is None:
            return
        selected = self.queue_popout_table.selection()
        if not selected:
            return
        idx = self.queue_popout_table.index(selected[0])
        self._queue_remove_callback(idx)

    def _onClearQueue(self):
        if self._queue_clear_callback is not None:
            self._queue_clear_callback()

    def _clear_dead_queue_popout_refs(self):
        try:
            if self.queue_popout is not None and not self.queue_popout.winfo_exists():
                self.queue_popout = None
                self.queue_popout_table = None
                return
        except tk.TclError:
            self.queue_popout = None
            self.queue_popout_table = None
            return

        try:
            if self.queue_popout_table is not None and not self.queue_popout_table.winfo_exists():
                self.queue_popout_table = None
        except tk.TclError:
            self.queue_popout_table = None

    def _on_queue_popout_closed(self):
        self.queue_popout_table = None
        self.queueGraphModelMenu = None
        try:
            if self.queue_popout is not None and self.queue_popout.winfo_exists():
                self.queue_popout.destroy()
        except tk.TclError:
            pass
        self.queue_popout = None

    def refresh_queue_table(self, jobs):
        def _apply():
            self.queueTable.delete(*self.queueTable.get_children())
            for job in jobs:
                self.queueTable.insert(
                    "",
                    "end",
                    values=(
                        job.job_id,
                        job.prediction_window,
                        job.ticker,
                        job.model.upper(),
                        job.seed,
                        job.graph_model.upper(),
                        job.k,
                        job.graph_mode,
                        job.graph_embed,
                        job.graph_ablation,
                        job.seq_len,
                    ),
                )

            self._clear_dead_queue_popout_refs()

            if self.queue_popout_table is not None:
                try:
                    self.queue_popout_table.delete(*self.queue_popout_table.get_children())
                    for job in jobs:
                        self.queue_popout_table.insert(
                            "",
                            "end",
                            values=(
                                job.job_id,
                                job.prediction_window,
                                job.ticker,
                                job.model.upper(),
                                job.seed,
                                job.graph_model.upper(),
                                job.k,
                                job.graph_mode,
                                job.graph_embed,
                                job.graph_ablation,
                                job.seq_len,
                            ),
                        )
                except tk.TclError:
                    self.queue_popout_table = None

        self.ui_call(_apply)

    def _open_queue_popout(self):
        self._clear_dead_queue_popout_refs()

        if self.queue_popout is not None:
            try:
                if self.queue_popout.winfo_exists():
                    self.queue_popout.lift()
                    return
            except tk.TclError:
                self.queue_popout = None
                self.queue_popout_table = None

        self.queue_popout = tk.Toplevel(self.root)
        self.queue_popout.title("Prediction Queue")
        self.queue_popout.geometry("1220x560")
        self.queue_popout.protocol("WM_DELETE_WINDOW", self._on_queue_popout_closed)

        self._build_queue_popout_controls(self.queue_popout)

        cols = ("Job ID", "Window", "Ticker", "Model", "Seed", "Graph", "k", "Mode", "Embed", "Ablation", "Seq")
        self.queue_popout_table = ttk.Treeview(self.queue_popout, columns=cols, show="headings")
        for c in cols:
            self.queue_popout_table.heading(c, text=c)
            self.queue_popout_table.column(c, width=90 if c in {"k", "Seq"} else 120, anchor=tk.CENTER)
        self.queue_popout_table.pack(fill=tk.BOTH, expand=True)
        self._copy_queue_strip_to_popout()

    def _copy_queue_strip_to_popout(self):
        if self.queue_popout_table is None:
            return

        try:
            self.queue_popout_table.delete(*self.queue_popout_table.get_children())
            for item in self.queueTable.get_children():
                values = self.queueTable.item(item).get("values", [])
                self.queue_popout_table.insert("", "end", values=values)
        except tk.TclError:
            self.queue_popout_table = None

    def _build_queue_popout_controls(self, parent):
        controls = tk.LabelFrame(parent, text="Queue Experiment Controls", padx=8, pady=8)
        controls.pack(fill=tk.X, padx=8, pady=(8, 4))

        for col in range(10):
            controls.columnconfigure(col, weight=1)

        def add_label(text, row, col):
            tk.Label(controls, text=text).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=(0, 2))

        def add_entry(var, row, col, width=10):
            entry = ttk.Entry(controls, textvariable=var, width=width)
            entry.grid(row=row, column=col, sticky="ew", padx=(0, 8), pady=(0, 8))
            return entry

        def add_combo(var, values, row, col, width=12):
            combo = ttk.Combobox(controls, textvariable=var, values=values, width=width, state="readonly")
            combo.grid(row=row, column=col, sticky="ew", padx=(0, 8), pady=(0, 8))
            return combo

        add_label("Window", 0, 0)
        add_combo(self.windowVar, self.windowValues, 1, 0, width=8)

        add_label("Stock spec", 0, 1)
        add_combo(self.stockVar, list(self.stockMenu.cget("values")), 1, 1, width=18).configure(state="normal")

        add_label("Model", 0, 2)
        add_combo(self.modelVar, self.modelValues, 1, 2, width=14).bind("<<ComboboxSelected>>", self._on_model_selection_change)

        add_label("Seeds", 0, 3)
        add_entry(self.seedVar, 1, 3, width=12)

        add_label("STGNN backend", 0, 4)
        graph_backend_combo = add_combo(self.graphModelVar, self.graphModelValues, 1, 4, width=14)
        self.queueGraphModelMenu = graph_backend_combo

        add_label("k", 0, 5)
        add_entry(self.kVar, 1, 5, width=6)

        add_label("Graph mode", 0, 6)
        add_combo(self.graphModeVar, self.graphModeValues, 1, 6, width=10)

        add_label("Embed", 0, 7)
        add_combo(self.graphEmbedVar, self.graphEmbedValues, 1, 7, width=8)

        add_label("Universe", 0, 8)
        universe_combo = add_combo(self.universeVar, self.universeValues, 1, 8, width=12)
        universe_combo.bind("<<ComboboxSelected>>", self._on_universe_change)
        tk.Button(controls, text="Import CSV", command=self._on_import_csv).grid(
            row=1,
            column=9,
            sticky="ew",
            padx=(0, 8),
            pady=(0, 8),
        )

        add_label("Graph ablation", 2, 0)
        add_combo(self.graphAblationVar, self.graphAblationValues, 3, 0, width=12)

        add_label("Feature ablation", 2, 1)
        add_combo(self.ablateFeatureVar, self.ablateFeatureValues, 3, 1, width=14)

        add_label("Seq len", 2, 2)
        add_entry(self.seqLenVar, 3, 2, width=8)

        add_label("Batch", 2, 3)
        add_entry(self.batchSizeVar, 3, 3, width=8)

        add_label("Recurrent epochs", 2, 4)
        add_entry(self.lstmEpochsVar, 3, 4, width=10)

        add_label("Graph/STGNN epochs", 2, 5)
        add_entry(self.stgnnEpochsVar, 3, 5, width=10)

        tk.Button(controls, text="Add to queue +", command=self._onAddToQueue).grid(
            row=3,
            column=6,
            sticky="ew",
            padx=(0, 8),
            pady=(0, 8),
        )
        tk.Button(controls, text="Run queue", command=self._onRunQueue).grid(
            row=3,
            column=7,
            sticky="ew",
            padx=(0, 8),
            pady=(0, 8),
        )
        tk.Button(controls, text="Remove selected", command=self._onRemoveQueuePopoutItem).grid(
            row=3,
            column=8,
            sticky="ew",
            padx=(0, 8),
            pady=(0, 8),
        )
        tk.Button(controls, text="Clear", command=self._onClearQueue).grid(
            row=3,
            column=9,
            sticky="ew",
            padx=(0, 8),
            pady=(0, 8),
        )

        self._sync_graph_backend_state()

    def _onCompute(self):
        if self._background_busy:
            self.set_status("Please wait until loading finishes.")
            return

        selected_model = self.get_selected_model()

        try:
            tickers = self._parse_selected_tickers()
        except Exception as exc:
            self.set_status(str(exc))
            return

        if len(tickers) != 1:
            self.set_status("Compute ▶ supports one ticker only. Use Add To Queue for multi-stock specs.")
            return

        self.set_status(f"Queued {selected_model.upper()} run...")

        self.stop_event.clear()
        self.btnCompute.config(state=tk.DISABLED, text="…Running")
        self.btnStop.config(state=tk.NORMAL)
        self.updateProgress(0.0)

        worker = threading.Thread(
            target=self._run_and_capture,
            args=(self.windowVar.get(), tickers[0]),
            daemon=True,
        )
        worker.start()

    def _run_and_capture(self, window, stock):
        cancelled = False
        try:
            result = self._compute_callback(window, stock, self.stop_event)
            if self.stop_event.is_set():
                cancelled = True
            else:
                self.root.after(
                    0,
                    lambda: self.updateResults(
                        result[0], result[1], result[2]
                    ) if result else self.updateResults(self.modelVar.get(), "—", 0.0)
                )
        except Cancelled:
            cancelled = True
        finally:
            if cancelled:
                self.root.after(0, self._reset_ui)
            else:
                self.root.after(
                    0,
                    lambda: (
                        self.btnCompute.config(state=tk.NORMAL, text="Compute ▶"),
                        self.btnStop.config(state=tk.DISABLED),
                    )
                )

    def _plot_df(self, df_view: pd.DataFrame, label: str):
        self.priceAx.clear()

        candidates = [c for c in df_view.columns if c.lower() == "close"]
        if not candidates:
            return
        price_col = candidates[0]

        self.priceAx.plot(
            df_view.index,
            df_view[price_col],
            label=label,
            linewidth=1.5,
        )

        self.priceAx.set_xlim(df_view.index[0], df_view.index[-1])

        for spine in ("top", "right"):
            self.priceAx.spines[spine].set_visible(False)

        if df_view.index[0] < df_view.index[-1]:
            self.priceAx.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            self.priceAx.set_xlim(df_view.index[0], df_view.index[-1])
        else:
            self.priceAx.yaxis.grid(False)

        self.priceAx.tick_params(
            axis="both",
            direction="in",
            length=4,
            width=0.5,
            labelsize="small",
        )
        leg = self.priceAx.legend(frameon=False, fontsize="small", loc="upper left")
        leg.set_alpha(0.8)
        self.priceAx.set_title(f"{label} Price History")
        self.priceAx.set_xlabel("Date")
        self.priceAx.set_ylabel("Closing Price (USD)")
        self.priceFig.tight_layout(pad=1.0)
        self.priceCanvas.draw_idle()

    def _onSelectionChange(self, event=None):
        if self._main_app is None:
            return

        ticker = self.stockVar.get()
        if ticker not in self._main_app.priceHistory:
            return

        df_raw = self._main_app.priceHistory.get(ticker)
        if df_raw is None:
            return

        df_full_trimmed = self._refresh_ticker_history(ticker, df_raw, num_bars=None)
        self.priceHistory[ticker] = df_full_trimmed

        data_len = len(df_full_trimmed)
        self.windowOptions = list(range(1, data_len + 1))
        self.currentWindowIdx = data_len - 1

        self.zoomSlider.config(from_=1, to=data_len, resolution=1)
        self.zoomSlider.set(self.currentWindowIdx + 1)

        self._refresh_and_plot(
            ticker,
            df_raw,
            num_bars=self.currentWindowIdx + 1,
        )

    def _refresh_ticker_history(
        self,
        ticker: str,
        df: pd.DataFrame,
        num_bars: Optional[int] = None,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df.get("Date", df.index)), drop=False)

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.tz_convert(None)

        price_col = "Close" if "Close" in df.columns else "close"
        mask = df[price_col] > 0
        if mask.any():
            first_real = mask.idxmax()
            df = df.loc[first_real:]

        if num_bars is not None:
            num_bars = max(1, min(len(df), num_bars))
            df = df.iloc[-num_bars:]

        return df

    def _refresh_and_plot(self, ticker: str, raw_df: pd.DataFrame, num_bars: int):
        df_view = self._refresh_ticker_history(ticker, raw_df, num_bars)
        if len(df_view) < 2:
            return
        self._plot_df(df_view, label=ticker)

    def _onPriceScroll(self, event):
        old_idx = self.currentWindowIdx

        raw_step = math.log(self.currentWindowIdx + 2)
        step = max(1, int(raw_step ** 2 / 4))
        delta = step if event.step > 0 else -step

        new_idx = self.currentWindowIdx + delta
        max_idx = len(self.windowOptions) - 1
        self.currentWindowIdx = max(0, min(new_idx, max_idx))

        if self.currentWindowIdx == old_idx:
            return

        self.zoomSlider.set(self.currentWindowIdx + 1)

        num_bars = self.currentWindowIdx + 1
        ticker = self.stockVar.get()
        if ticker not in self.priceHistory:
            return

        df_full = self.priceHistory[ticker]
        self._refresh_and_plot(ticker, df_full, num_bars=num_bars)

    def _onSliderChange(self, value_str):
        try:
            val = int(value_str)
        except ValueError:
            return

        new_idx = val - 1
        if new_idx == self.currentWindowIdx:
            return

        max_idx = len(self.windowOptions) - 1
        self.currentWindowIdx = max(0, min(new_idx, max_idx))

        ticker = self.stockVar.get()
        if ticker not in self.priceHistory:
            return

        raw_df = self.priceHistory[ticker]
        self._refresh_and_plot(ticker, raw_df, num_bars=self.currentWindowIdx + 1)

    def updateTable(self, feature_df):
        if feature_df is None or feature_df.empty:
            return

        self.table.configure(takefocus=False)
        self.table.delete(*self.table.get_children())
        self.table["displaycolumns"] = ("Stock", "Return", "Volatility", "Volume", "Momentum")

        rows = []
        for idx, row in feature_df.iterrows():
            try:
                vals = (
                    idx,
                    f"{row.get('return', np.nan):.4f}",
                    f"{row.get('volatility', np.nan):.4f}",
                    f"{row.get('volume', np.nan):.4f}",
                    f"{row.get('momentum', np.nan):.4f}",
                )
                rows.append(vals)
            except Exception as exc:
                print(f"Skipping row {idx} due to error: {exc}")

        for vals in rows:
            self.table.insert("", "end", values=vals)
        self.table.configure(takefocus=True)

    def plot3d_on_ax(
        self,
        tickers: List[str],
        coords: np.ndarray,
        pruned_edges: Optional[List[Tuple[int, int, float]]] = None,
        mst_edges: Optional[List[Tuple[int, int, float]]] = None,
        ax=None,
    ) -> None:
        if ax is None:
            ax = self.graphAx

        coords = coords - coords.mean(axis=0)
        max_abs = abs(coords).max()
        if max_abs > 0:
            coords = coords / max_abs
        coords = coords * 0.8

        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        self._last_drawn_tickers = tickers
        self._last_drawn_pos = {t: (xs[i], ys[i], zs[i]) for i, t in enumerate(tickers)}
        self._last_pruned_edges = pruned_edges or []
        self._last_mst_edges = mst_edges or []

        ax.clear()

        ticker_to_sector = getattr(self, "ticker_to_sector", {}) or {}
        sectors = [ticker_to_sector.get(t, "Unknown") for t in tickers]
        self._sector_to_colour = self._sector_palette(sectors)
        node_colours = [self._sector_to_colour.get(s, (0.6, 0.6, 0.6, 1.0)) for s in sectors]

        ax.scatter(xs, ys, zs, s=75, depthshade=False, c=node_colours, alpha=1.0)
        self.update_sector_legend()

        for i, tkr in enumerate(tickers):
            ax.text(xs[i], ys[i], zs[i], tkr, size=6, alpha=0.95)

        self._last_edge_labels = []

        if pruned_edges:
            for i, j, w in pruned_edges:
                ax.plot(
                    [xs[i], xs[j]],
                    [ys[i], ys[j]],
                    [zs[i], zs[j]],
                    color="blue",
                    linewidth=1.5,
                    alpha=0.5,
                )
                mx, my, mz = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2, (zs[i] + zs[j]) / 2
                ax.text(mx, my, mz, f"{w:.2f}", fontsize=5, color="black", alpha=0.6)
                self._last_edge_labels.append(((i, j), (mx, my, mz), w))

        if mst_edges:
            for i, j, w in mst_edges:
                ax.plot(
                    [xs[i], xs[j]],
                    [ys[i], ys[j]],
                    [zs[i], zs[j]],
                    color="red",
                    linewidth=2.0,
                    alpha=0.7,
                    linestyle="--",
                )
                mx, my, mz = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2, (zs[i] + zs[j]) / 2
                self._last_edge_labels.append(((i, j), (mx, my, mz), w))

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.grid(False)
        for axis in ("x", "y", "z"):
            getattr(ax, f"{axis}axis").pane.fill = False

        fig = ax.get_figure()
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        ax.view_init(elev=20, azim=30)
        fig.canvas.draw_idle()

    def update_sector_legend(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        if not self._sector_to_colour:
            tk.Label(self.legend_frame, text="(sector map not loaded)").pack(anchor="w")
            return

        patches = []
        for sector, colour in sorted(self._sector_to_colour.items()):
            patches.append(Patch(facecolor=colour, edgecolor="black", label=sector))

        legend_fig = Figure(figsize=(2.6, 4.5), dpi=100)
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=patches,
            loc="upper left",
            frameon=False,
            fontsize=8,
        )
        canvas = FigureCanvasTkAgg(legend_fig, master=self.legend_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw_idle()

    def _sector_palette(self, sectors: List[str]):
        unique = sorted(set(sectors))
        if not unique:
            return {}

        cmap = cm.get_cmap("tab20", len(unique))
        return {sector: cmap(i) for i, sector in enumerate(unique)}

    def setTrainers(self, lstm=None, stgnn=None):
        self._trainers["lstm"] = lstm
        self._trainers["stgnn"] = stgnn

    def updateProgress(self, fraction: float):
        frac = max(0.0, min(1.0, float(fraction)))

        def _update():
            self.progressVar.set(frac)

        try:
            self.root.after(0, _update)
        except Exception as exc:
            print(f"[WARN] Progress update skipped: {exc}")

    def bars_to_days(self, bar_count: int, interval: str) -> float:
        bars_per_day_map = {
            "1m": 360,
            "5m": 72,
            "15m": 24,
            "30m": 12,
            "1h": 6,
            "1d": 1,
            "1wk": 1 / 5,
            "1mo": 1 / 21,
        }
        bars_per_day = bars_per_day_map.get(interval, 1)
        return bar_count / bars_per_day

    def _onStop(self):
        self.stop_event.set()
        self.set_status("Cancelling")
        self.btnStop.config(state=tk.DISABLED)

    def set_status(self, msg: str):
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self.statusVar.set(msg))
        except (RuntimeError, tk.TclError):
            pass

    def clear_status(self):
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self.statusVar.set("Idle"))
        except (RuntimeError, tk.TclError):
            pass

    def _set_loading_state(self, busy: bool, message: str = ""):
        """
        Keep the UI responsive during heavier data work such as universe reloads
        and custom CSV imports.
        """
        def _apply():
            self._background_busy = bool(busy)

            if message:
                self.statusVar.set(message)
            elif not busy:
                self.statusVar.set("Idle")

            controls = [
                getattr(self, "universeMenu", None),
                getattr(self, "stockMenu", None),
                getattr(self, "windowMenu", None),
                getattr(self, "modelMenu", None),
                getattr(self, "seedEntry", None),
                getattr(self, "graphModelMenu", None),
                getattr(self, "btnImport", None),
                getattr(self, "btnCompute", None),
                getattr(self, "btnQueueAdd", None),
                getattr(self, "btnRunQueue", None),
                getattr(self, "btnQueuePopout", None),
            ]

            for widget in controls:
                if widget is None:
                    continue
                try:
                    if busy:
                        widget.configure(state=tk.DISABLED)
                    else:
                        if widget is self.btnImport:
                            widget.configure(state=tk.NORMAL if self.universeVar.get() == "Custom" else tk.DISABLED)
                        elif widget is self.graphModelMenu:
                            selected_model = str(self.modelVar.get()).strip().lower()
                            widget.configure(state="readonly" if self._is_graph_backend_applicable(selected_model) else tk.DISABLED)
                        elif widget in (self.universeMenu, self.windowMenu, self.modelMenu):
                            widget.configure(state="readonly")
                        elif widget is self.stockMenu:
                            widget.configure(state="normal")
                        else:
                            widget.configure(state=tk.NORMAL)
                except tk.TclError:
                    pass

            try:
                if busy:
                    self.progressBar.configure(mode="indeterminate", maximum=100)
                    self.progressBar.start(12)
                else:
                    self.progressBar.stop()
                    self.progressBar.configure(mode="determinate", maximum=1.0)
                    self.progressVar.set(0.0)
            except tk.TclError:
                pass

        self.ui_call(_apply)

    def _run_background_task(self, *, title: str, work, on_success=None, on_error=None):
        """
        Run blocking IO/data preparation away from the Tk event loop.

        `work` runs on a worker thread. `on_success` and `on_error` run on
        the Tk thread.
        """
        if self._background_busy:
            self.set_status("Another loading task is already running.")
            return

        self._set_loading_state(True, title)

        def _worker():
            try:
                result = work()
            except Exception as exc:
                def _fail(exc=exc):
                    self._set_loading_state(False, f"{title} failed")
                    if on_error is not None:
                        on_error(exc)
                self.ui_call(_fail)
                return

            def _finish(result=result):
                try:
                    if on_success is not None:
                        on_success(result)
                finally:
                    self._set_loading_state(False, "Idle")
            self.ui_call(_finish)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_loaded_universe(self, selected_label: str, valid_tickers):
        valid_tickers = list(valid_tickers or [])
        self.stockMenu.configure(values=valid_tickers)

        if valid_tickers:
            self.stockVar.set(valid_tickers[0])
            self._onSelectionChange()

        self.set_status(f"Loaded universe: {selected_label} ({len(valid_tickers)} ticker(s))")

    def clear_axis(self, pane):
        if pane is None:
            return
        for widget in pane.winfo_children():
            widget.destroy()

    def _reset_ui(self):
        self.clear_status()
        self.btnCompute.config(state=tk.NORMAL, text="Compute ▶")
        self.updateProgress(0.0)

        try:
            self.updateResults(self.modelVar.get(), "—", 0.0)
        except Exception as exc:
            print(f"[WARNING] updateResults failed during UI reset: {exc}")

        self.graphAx.clear()
        self.graphCanvas.draw_idle()

        for row in self.table.get_children():
            self.table.delete(row)

        clear_targets = [
            self.eval_pane,
            self.backtest_pane,
            self.roc_pane,
            self.threshold_pane,
        ]
        self.root.after(0, lambda: [self.clear_axis(pane) for pane in clear_targets])

    def updateResults(self, model_name, trend, confidence):
        is_up = "Upwards" in str(trend)
        colour = "green" if is_up else "red" if trend not in ("—", "-", None) else "black"

        self.modelStatus.config(text=f"Trending: {trend}", fg=colour)
        self.modelConf.config(text=f"Confidence: {confidence:.1f}%", fg=colour)

    def refresh_selected_tabs(self):
        for tab in [self.metricsTab, self.evalTab, self.backtestTab]:
            if tab:
                tab.update_idletasks()

        def _rebalance():
            try:
                if hasattr(self, "metrics_pane") and isinstance(self.metrics_pane, tk.PanedWindow):
                    self.metrics_pane.update_idletasks()
                    w = self.metrics_pane.winfo_width()
                    if w > 10:
                        self.metrics_pane.sash_place(0, w // 2, 0)

                if hasattr(self, "eval_vertical_pane") and isinstance(self.eval_vertical_pane, tk.PanedWindow):
                    self.eval_vertical_pane.update_idletasks()
                    h = self.eval_vertical_pane.winfo_height()
                    if h > 10:
                        self.eval_vertical_pane.sash_place(0, 0, int(h * 0.5))

                if hasattr(self, "metrics_model_pane") and isinstance(self.metrics_model_pane, tk.PanedWindow):
                    self.metrics_model_pane.update_idletasks()
                    h = self.metrics_model_pane.winfo_height()
                    if h > 10:
                        self.metrics_model_pane.sash_place(0, 0, int(h * 0.5))
            except Exception as exc:
                print(f"[WARN] UI rebalance skipped: {exc}")

        self.root.after(50, _rebalance)

    def _on_graph_keypress(self, event):
        return

    def _on_table_select(self, event):
        selected = self.table.selection()
        if not selected:
            return

        item = self.table.item(selected[0])
        values = item.get("values", [])
        if not values:
            return

        ticker = str(values[0])
        self._selected_ticker = ticker
        self.highlight_ticker(ticker)

        if hasattr(self, "on_ticker_click") and callable(self.on_ticker_click):
            try:
                self.on_ticker_click(ticker)
            except Exception as exc:
                print(f"[WARN] ticker click handler failed: {exc}")

    def highlight_ticker(self, ticker: str):
        if ticker not in self._last_drawn_pos or not self._last_drawn_tickers:
            return

        current_elev = getattr(self.graphAx, "elev", 20)
        current_azim = getattr(self.graphAx, "azim", 30)

        self.graphAx.clear()

        tickers = self._last_drawn_tickers
        xs = [self._last_drawn_pos[t][0] for t in tickers]
        ys = [self._last_drawn_pos[t][1] for t in tickers]
        zs = [self._last_drawn_pos[t][2] for t in tickers]

        sectors = [self.ticker_to_sector.get(t, "Unknown") for t in tickers]
        if not self._sector_to_colour:
            self._sector_to_colour = self._sector_palette(sectors)

        for i, t in enumerate(tickers):
            base_colour = self._sector_to_colour.get(sectors[i], (0.6, 0.6, 0.6, 1.0))
            is_sel = t == ticker
            size = 120 if is_sel else 75
            self.graphAx.scatter(
                [xs[i]],
                [ys[i]],
                [zs[i]],
                s=size,
                depthshade=True,
                alpha=0.98,
                facecolors=[base_colour],
                edgecolors="red" if is_sel else "black",
                linewidths=2.0 if is_sel else 0.6,
                zorder=6 if is_sel else 3,
            )
            self.graphAx.text(xs[i], ys[i], zs[i], t, size=6, alpha=1.0)

        for i, j, _w in self._last_pruned_edges:
            self.graphAx.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                [zs[i], zs[j]],
                color="blue",
                linewidth=1.5,
                alpha=0.5,
            )

        for i, j, _w in self._last_mst_edges:
            self.graphAx.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                [zs[i], zs[j]],
                color="red",
                linewidth=2.0,
                alpha=0.7,
                linestyle="--",
            )

        coord = self._last_drawn_pos[ticker]
        x, y, z = coord
        halo = self.graphAx.scatter(
            [x],
            [y],
            [z],
            s=400,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=10,
        )
        self._highlight_markers = [halo]

        self.graphAx.view_init(elev=current_elev, azim=current_azim)
        self.graphAx.set_xlim(-1, 1)
        self.graphAx.set_ylim(-1, 1)
        self.graphAx.set_zlim(-1, 1)
        self.graphAx.set_box_aspect((1, 1, 1))
        for axis in ("x", "y", "z"):
            getattr(self.graphAx, f"{axis}axis").pane.fill = False
        self.graphAx.grid(False)
        self.graphAx.set_axis_off()
        self.graphCanvas.draw_idle()

    @property
    def backtestLSTM(self):
        return self.backtest_pane

    @property
    def backtestSTGNN(self):
        return self.backtest_pane

    def set_sector_map(self, ticker_to_sector):
        self.ticker_to_sector = dict(ticker_to_sector or {})
        sectors = list(self.ticker_to_sector.values())
        self._sector_to_colour = self._sector_palette(sectors) if sectors else {}
        self.update_sector_legend()

    def _on_model_selection_change(self, event=None):
        if self.btnCompute["state"] == tk.DISABLED:
            return

        selected_model = str(self.modelVar.get()).strip().lower()

        def _update():
            self.set_active_model_titles(selected_model)
            self._sync_graph_backend_state()
            self.refresh_selected_tabs()

        self.ui_call(_update)

    def _on_universe_change(self, event=None):
        """
        Handle universe dropdown selection change without blocking Tk.
        """
        selected = self.universeVar.get()

        if selected == "Custom":
            self.btnImport.configure(state=tk.NORMAL)
        else:
            self.btnImport.configure(state=tk.DISABLED)

        if self._main_app is None:
            return

        universe_map = {
            "S&P 500": "sp500",
            "NASDAQ 100": "nasdaq100",
            "Custom": "custom",
        }
        universe_id = universe_map.get(selected, "sp500")

        def _work():
            return self._main_app.reload_universe(universe_id)

        def _success(valid_tickers):
            self._apply_loaded_universe(selected, valid_tickers)

        def _error(exc):
            self.set_status(f"Error loading universe: {exc}")

        self._run_background_task(
            title=f"Loading {selected} universe...",
            work=_work,
            on_success=_success,
            on_error=_error,
        )

    def _on_import_csv(self, event=None):
        """
        Import a custom ticker CSV without blocking the UI.

        Expected CSV columns:
          - ticker
          - sector
        """
        if self._background_busy:
            self.set_status("Please wait until loading finishes.")
            return

        filepath = filedialog.askopenfilename(
            title="Select Ticker CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not filepath:
            return

        filepath = str(filepath)

        def _work():
            df = pd.read_csv(filepath)

            required = {"ticker", "sector"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(
                    f"CSV must include columns: {', '.join(sorted(required))}. Missing: {', '.join(missing)}"
                )

            clean = df.copy()
            clean["ticker"] = clean["ticker"].astype(str).str.strip().str.upper()
            clean["sector"] = clean["sector"].astype(str).str.strip()
            clean = clean[(clean["ticker"] != "") & (clean["sector"] != "")]
            clean = clean.drop_duplicates(subset=["ticker"], keep="first")

            if clean.empty:
                raise ValueError("CSV contains no usable ticker rows after cleaning.")

            custom_path = self.project_root / "static" / "universes" / "custom_tickers.csv"
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            clean.to_csv(custom_path, index=False)

            custom_tickers = clean["ticker"].tolist()

            if self._main_app is None:
                valid_tickers = custom_tickers
            else:
                valid_tickers = self._main_app.reload_universe("custom", custom_tickers)

            return {
                "valid_tickers": valid_tickers,
                "imported_count": len(custom_tickers),
                "custom_path": custom_path,
            }

        def _success(result):
            valid_tickers = list(result.get("valid_tickers") or [])
            imported_count = int(result.get("imported_count") or 0)

            self.universeVar.set("Custom")
            self.btnImport.configure(state=tk.NORMAL)
            self.stockMenu.configure(values=valid_tickers)

            if valid_tickers:
                self.stockVar.set(valid_tickers[0])
                self._onSelectionChange()

            self.set_status(
                f"Imported {imported_count} ticker(s); {len(valid_tickers)} valid ticker(s) loaded."
            )

        def _error(exc):
            self.set_status(f"Error importing CSV: {exc}")

        self._run_background_task(
            title="Importing custom universe...",
            work=_work,
            on_success=_success,
            on_error=_error,
        )

    def _schedule_rebalance(self):
        # Cancel previous job safely
        job = getattr(self, "_rebalance_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass

        # Schedule safely via wrapper (prevents stale callback errors)
        def _run():
            self._rebalance_job = None
            try:
                self._rebalance_main_workspace()
            except Exception:
                pass  # prevents Tk "invalid command" crash

        self._rebalance_job = self.root.after(120, _run)

    def _on_close(self):
        if self._closing:
            return
        self._closing = True
        self.stop_event.set()

        job = getattr(self, "_rebalance_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass

        try:
            if self._close_callback is not None:
                self._close_callback()
        except Exception as exc:
            print(f"[WARN] Close callback failed: {exc}")

        try:
            self._on_queue_popout_closed()
        except Exception:
            pass

        try:
            self.root.quit()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass
