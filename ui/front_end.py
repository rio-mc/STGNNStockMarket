import math
import threading
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Optional

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from core.job_queue import QueueJob, JobQueueController, parse_seed_spec
from evaluation.evaluation_methods import EvaluationMethods

class Cancelled(Exception):
    """Raised when training is aborted by user."""
    pass


class FrontEnd:
    """
    Handles front-end logic for a single selected-model run.

    Refactor intent:
    - one selected model
    - one evaluation pane
    - one backtest pane
    - one metrics pane
    """

    def __init__(self, availableTickers):
        # ====================================
        # === 1. Top-level window
        self.root = tk.Tk()
        self.root.title("Stock Trend Explorer")
        self.root.geometry("1650x1050")
        self.root.minsize(1300, 900)

        # ====================================
        # === 2. Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ====================================
        # === 3. Main tab
        self.mainTab = tk.Frame(self.notebook)
        self.notebook.add(self.mainTab, text="Main")

        # ====================================
        # === 4. Toolbar frame
        self.toolbar = tk.Frame(self.mainTab, height=40, bd=1, relief=tk.RAISED)
        self.toolbar.pack(fill=tk.X)

        tk.Label(self.toolbar, text="Prediction range:").pack(side=tk.LEFT, padx=(10, 2))
        self.windowValues = ["1d", "2d", "5d", "1w"]
        self.windowOptions = []
        self.windowVar = tk.StringVar(value=self.windowValues[0])
        self.windowMenu = ttk.Combobox(
            self.toolbar,
            textvariable=self.windowVar,
            values=self.windowValues,
            width=6,
            state="readonly",
        )
        self.windowMenu.pack(side=tk.LEFT, padx=2)
        self.windowMenu.bind("<<ComboboxSelected>>", self._onSelectionChange)

        tk.Label(self.toolbar, text="Stock:").pack(side=tk.LEFT, padx=(20, 2))
        self.stockVar = tk.StringVar(value=availableTickers[0] if availableTickers else "")
        self.stockMenu = ttk.Combobox(
            self.toolbar,
            textvariable=self.stockVar,
            values=availableTickers,
            width=8,
            state="readonly",
        )
        self.stockMenu.pack(side=tk.LEFT, padx=2)
        self.stockMenu.bind("<<ComboboxSelected>>", self._onSelectionChange)

        tk.Label(self.toolbar, text="Model:").pack(side=tk.LEFT, padx=(20, 2))
        
        self.modelValues = [
            "LSTM",
            "GRU",
            "PANEL_GRU",
            "PANEL_LSTM",
            "GCN",
            "NNCONV",
            "GRAPHSAGE",
            "STGNN",
        ]

        self.modelVar = tk.StringVar(value=self.modelValues[0])
        self.modelMenu = ttk.Combobox(
            self.toolbar,
            textvariable=self.modelVar,
            values=self.modelValues,
            width=10,
            state="readonly"
        )
        self.modelMenu.pack(side=tk.LEFT, padx=2)
        self.modelMenu.bind("<<ComboboxSelected>>", self._on_model_selection_change)

        tk.Label(self.toolbar, text="Seed(s):").pack(side=tk.LEFT, padx=(20, 2))
        self.seedVar = tk.StringVar(value="42")
        self.seedEntry = ttk.Entry(self.toolbar, textvariable=self.seedVar, width=12)
        self.seedEntry.pack(side=tk.LEFT, padx=2)

        tk.Label(self.toolbar, text="Graph backend:").pack(side=tk.LEFT, padx=(20, 2))
        self.graphModelValues = ["GCN", "GRAPHSAGE", "GAT", "NNCONV"]
        self.graphModelVar = tk.StringVar(value=self.graphModelValues[0])
        self.graphModelMenu = ttk.Combobox(
            self.toolbar,
            textvariable=self.graphModelVar,
            values=self.graphModelValues,
            width=12,
            state="readonly"
        )
        self.graphModelMenu.pack(side=tk.LEFT, padx=2)
        self._sync_graph_backend_state()
        self.stop_event = threading.Event()
        self.statusVar = tk.StringVar(value="Idle")

        self.btnCompute = tk.Button(self.toolbar, text="Compute ▶", command=self._onCompute)
        self.btnCompute.pack(side=tk.RIGHT, padx=10)

        self.btnStop = tk.Button(self.toolbar, text="Stop ■", command=self._onStop, state=tk.DISABLED)
        self.btnStop.pack(side=tk.RIGHT, padx=(0, 10))

        self.btnRunQueue = tk.Button(self.toolbar, text="Run Queue ▷", command=self._onRunQueue)
        self.btnRunQueue.pack(side=tk.RIGHT, padx=(0, 10))

        self.btnQueueAdd = tk.Button(self.toolbar, text="Add To Queue +", command=self._onAddToQueue)
        self.btnQueueAdd.pack(side=tk.RIGHT, padx=(0, 10))

        self.btnQueuePopout = tk.Button(self.toolbar, text="Queue Popout", command=self._open_queue_popout)
        self.btnQueuePopout.pack(side=tk.RIGHT, padx=(0, 10))

        self.progressVar = tk.DoubleVar(value=0.0)
        self.progressBar = ttk.Progressbar(
            self.toolbar,
            variable=self.progressVar,
            maximum=1.0,
            length=200,
        )
        self.progressBar.pack(side=tk.RIGHT, padx=(0, 10))

        self.statusLabel = tk.Label(
            self.toolbar,
            textvariable=self.statusVar,
            anchor="e",
            fg="gray",
            font=("Arial", 9),
        )
        self.statusLabel.pack(side=tk.RIGHT, padx=10)

        self.queueFrame = tk.LabelFrame(self.mainTab, text="Prediction Queue", padx=6, pady=6)
        self.queueFrame.pack(fill=tk.X, padx=10, pady=(6, 4))

        queue_cols = ("Job ID", "Window", "Ticker", "Model", "Seed", "Graph")
        self.queueTable = ttk.Treeview(self.queueFrame, columns=queue_cols, show="headings", height=5)
        for c in queue_cols:
            self.queueTable.heading(c, text=c)
            self.queueTable.column(c, width=110, anchor=tk.CENTER)
        self.queueTable.pack(side=tk.LEFT, fill=tk.X, expand=True)

        queue_scroll = ttk.Scrollbar(self.queueFrame, orient=tk.VERTICAL, command=self.queueTable.yview)
        queue_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.queueTable.configure(yscrollcommand=queue_scroll.set)

        self.queueButtons = tk.Frame(self.queueFrame)
        self.queueButtons.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        tk.Button(self.queueButtons, text="Remove", command=self._onRemoveQueueItem).pack(fill=tk.X, pady=2)
        tk.Button(self.queueButtons, text="Clear", command=self._onClearQueue).pack(fill=tk.X, pady=2)

        self.queue_popout = None
        self.queue_popout_table = None

        # ====================================
        # === 5. Price chart
        self.priceFrame = tk.Frame(self.mainTab, bd=1, relief=tk.SUNKEN)
        self.priceFrame.pack(fill=tk.BOTH, expand=True)

        self.priceFig, self.priceAx = plt.subplots(figsize=(8, 2.5))
        self.priceCanvas = FigureCanvasTkAgg(self.priceFig, master=self.priceFrame)
        self.priceCanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.priceCanvas.mpl_connect("scroll_event", self._onPriceScroll)

        self.zoomSlider = tk.Scale(
            self.priceFrame,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            length=600,
            showvalue=False,
            command=self._onSliderChange,
        )
        self.zoomSlider.pack(fill=tk.X, padx=10, pady=(0, 40))

        # ====================================
        # === 6. Prediction results panel
        self.resultFrame = tk.Frame(self.mainTab)
        self.resultFrame.pack(fill=tk.X)

        self.modelRes = tk.LabelFrame(self.resultFrame, text="Model Prediction", padx=10, pady=10)
        self.modelRes.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20, pady=5)

        self.modelNameLabel = tk.Label(
            self.modelRes,
            text=f"Model: {self.modelVar.get()}",
            font=("Helvetica", 12, "bold"),
        )
        self.modelNameLabel.pack()
        self.modelStatus = tk.Label(
            self.modelRes,
            text="Trending: —",
            font=("Helvetica", 12, "bold"),
            fg="black",
        )
        self.modelStatus.pack()

        self.modelConf = tk.Label(
            self.modelRes,
            text="Confidence: —",
        )
        self.modelConf.pack()

        # ====================================
        # === 7. Bottom pane: graph + feature table
        self.bottom_pane = tk.PanedWindow(self.mainTab, orient=tk.VERTICAL)
        self.bottom_pane.pack(fill=tk.BOTH, expand=True)

        self.graphFrame = tk.LabelFrame(self.bottom_pane, text="Graph Output")
        self.bottom_pane.add(self.graphFrame, minsize=200)

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

        self.table_pane = tk.Frame(self.bottom_pane, height=150)
        self.bottom_pane.add(self.table_pane, minsize=150)

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

        # ====================================
        # === 8. Evaluation tab (single active model)
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

        # ====================================
        # === 9. Backtesting tab (single active model)
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

        # ====================================
        # === 10. Metrics tab (single active model)
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

        # ====================================
        # === 11. State
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

        # ====================================
        # === 12. Evaluation logic
        self.evaluator = EvaluationMethods(self)
        self.evaluator.reset_histories()

    def bindMainApp(self, main_app):
        self._main_app = main_app
        interval = self._main_app.args.interval
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
            f"[DEBUG] {ticker}: {data_len} bars ({interval}) ≈ {est_days:.1f} trading days, "
            f"{duration.days} calendar days from {start.date()} to {end.date()} → windowOptions=1…{data_len}"
        )

        raw_df = self._main_app.raw_feature_dfs[ticker]
        self._refresh_and_plot(ticker, raw_df, num_bars=self.currentWindowIdx + 1)

    def set_active_model_titles(self, model_name: str):
        labels = {
            "lstm": "LSTM",
            "gru": "GRU",
            "panel_gru": "PANEL GRU",
            "panel_lstm": "PANEL LSTM",
            "gcn": "GCN",
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

    def _build_current_jobs(self) -> List[QueueJob]:
        seed_values = parse_seed_spec(str(self.seedVar.get()).strip())

        model_name = str(self.modelVar.get()).strip().lower()
        graph_model = (
            str(self.graphModelVar.get()).strip().lower()
            if self._is_graph_backend_applicable(model_name)
            else "gcn"
        )

        jobs: List[QueueJob] = []
        for seed in seed_values:
            jobs.append(
                QueueJob(
                    job_id=JobQueueController.make_job_id(),
                    created_at="now",
                    prediction_window=str(self.windowVar.get()).strip(),
                    ticker=str(self.stockVar.get()).strip().upper(),
                    model=model_name,
                    seed=int(seed),
                    graph_model=graph_model,
                )
            )
        return jobs

    def _onAddToQueue(self):
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

    def _onClearQueue(self):
        if self._queue_clear_callback is not None:
            self._queue_clear_callback()

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
        self.queue_popout.geometry("900x300")
        self.queue_popout.protocol("WM_DELETE_WINDOW", self._on_queue_popout_closed)

        cols = ("Job ID", "Window", "Ticker", "Model", "Seed", "Graph")
        self.queue_popout_table = ttk.Treeview(
            self.queue_popout,
            columns=cols,
            show="headings",
        )
        for c in cols:
            self.queue_popout_table.heading(c, text=c)
            self.queue_popout_table.column(c, width=130, anchor=tk.CENTER)
        self.queue_popout_table.pack(fill=tk.BOTH, expand=True)

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
        try:
            if self.queue_popout is not None and self.queue_popout.winfo_exists():
                self.queue_popout.destroy()
        except tk.TclError:
            pass
        self.queue_popout = None

    def _onCompute(self):
        selected_model = self.get_selected_model()
        self.set_status(f"Queued {selected_model.upper()} run...")

        self.stop_event.clear()
        self.btnCompute.config(state=tk.DISABLED, text="…Running")
        self.btnStop.config(state=tk.NORMAL)
        self.updateProgress(0.0)

        worker = threading.Thread(
            target=self._run_and_capture,
            args=(self.windowVar.get(), self.stockVar.get()),
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
        self.priceCanvas.draw()

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
        canvas.draw()

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
        self.root.after(0, lambda: self.statusVar.set(msg))

    def clear_status(self):
        self.root.after(0, lambda: self.statusVar.set("Idle"))

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
        self.graphCanvas.draw()

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
        # Placeholder for future graph keyboard interactions
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

    def ui_call(self, fn, *args, **kwargs):
        try:
            self.root.after(0, lambda: fn(*args, **kwargs))
        except RuntimeError:
            pass

    def _is_graph_backend_applicable(self, model_name: str) -> bool:
        key = str(model_name).strip().lower()
        return key == "stgnn"


    def _sync_graph_backend_state(self):
        selected_model = str(self.modelVar.get()).strip().lower()
        enabled = self._is_graph_backend_applicable(selected_model)

        self.graphModelMenu.config(state="readonly" if enabled else "disabled")
        if not enabled:
            self.graphModelVar.set("GCN")