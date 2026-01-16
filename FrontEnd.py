import math
import threading
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.cm as cm
from EvaluationMethods import EvaluationMethods

class Cancelled(Exception):
    """Raised when training is aborted by user."""
    pass

class FrontEnd:
    """
    Handles all non-evaluatory front-end logic.
    """
    def __init__(self, availableTickers):
        #   1. Top-level window
        self.root = tk.Tk()
        self.root.title("Stock Trend Explorer")
        self.root.geometry("1500x1000")
        self.root.minsize(1200, 900)

        #   2. Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        #   3. Main tab
        self.mainTab = tk.Frame(self.notebook)
        self.notebook.add(self.mainTab, text="Main")

        #   4. Toolbar frame
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
            state="readonly"
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
            state="readonly"
        )
        self.stockMenu.pack(side=tk.LEFT, padx=2)
        self.stockMenu.bind("<<ComboboxSelected>>", self._onSelectionChange)
        self.stop_event = threading.Event()
        self.statusVar = tk.StringVar()
        self.statusVar.set("Idle")
        self.btnCompute = tk.Button(self.toolbar, text="Compute ▶", command=self._onCompute)
        self.btnCompute.pack(side=tk.RIGHT, padx=10)
        self.btnStop = tk.Button(self.toolbar, text="Stop ■", command=self._onStop, state=tk.DISABLED)
        self.btnStop.pack(side=tk.RIGHT, padx=(0, 10))
        self.progressVar = tk.DoubleVar(value=0.0)
        self.progressBar = ttk.Progressbar(
            self.toolbar,
            variable=self.progressVar,
            maximum=1.0,
            length=200
        )
        self.progressBar.pack(side=tk.RIGHT, padx=(0, 10))
        self.statusLabel = tk.Label(self.toolbar, textvariable=self.statusVar, anchor="e", fg="gray", font=("Arial", 9))
        self.statusLabel.pack(side=tk.RIGHT, padx=10)

        #   5. Price chart
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
            command=self._onSliderChange
        )
        self.zoomSlider.pack(fill=tk.X, padx=10, pady=(0, 40))

        #   6. Prediction results panel
        self.resultFrame = tk.Frame(self.mainTab)
        self.resultFrame.pack(fill=tk.X)

        # --- LSTM
        self.lstmRes = tk.LabelFrame(self.resultFrame, text="LSTM Prediction", padx=10, pady=10)
        self.lstmRes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=5)
        self.lstmStatus = tk.Label(self.lstmRes, text="Trending: —", font=("Helvetica", 12, "bold"), fg="green")
        self.lstmConf = tk.Label(self.lstmRes, text="Confidence: —")
        self.lstmStatus.pack()
        self.lstmConf.pack()

        # --- GRU
        self.gruRes = tk.LabelFrame(self.resultFrame, text="GRU Prediction", padx=10, pady=10)
        self.gruRes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=5)
        self.gruStatus = tk.Label(self.gruRes, text="Trending: —", font=("Helvetica", 12, "bold"), fg="blue")
        self.gruConf = tk.Label(self.gruRes, text="Confidence: —")
        self.gruStatus.pack()
        self.gruConf.pack()

        # --- STGNN
        self.stgnnRes = tk.LabelFrame(self.resultFrame, text="STGNN Prediction", padx=10, pady=10)
        self.stgnnRes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=5)
        self.stgnnStatus = tk.Label(self.stgnnRes, text="Trending: —", font=("Helvetica", 12, "bold"), fg="red")
        self.stgnnConf = tk.Label(self.stgnnRes, text="Confidence: —")
        self.stgnnStatus.pack()
        self.stgnnConf.pack()

        #   7. Bottom pane: graph + feature table
        self.bottom_pane = tk.PanedWindow(self.mainTab, orient=tk.VERTICAL)
        self.bottom_pane.pack(fill=tk.BOTH, expand=True)

        # ===== Graph frame (top of bottom pane)
        self.graphFrame = tk.LabelFrame(self.bottom_pane, text="Graph Output")
        self.bottom_pane.add(self.graphFrame, minsize=200)

        # Side-by-side container for (graph canvas | legend panel)
        self.graph_container = tk.Frame(self.graphFrame)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        # LEFT: matplotlib canvas frame
        self.graph_canvas_frame = tk.Frame(self.graph_container)
        self.graph_canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.graphFig = Figure(figsize=(6, 4), dpi=100)
        self.graphAx = self.graphFig.add_subplot(111, projection="3d")
        self.graphCanvas = FigureCanvasTkAgg(self.graphFig, master=self.graph_canvas_frame)
        self.graphCanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # RIGHT: sector legend frame (outside the plot)
        self.legend_frame = tk.LabelFrame(self.graph_container, text="Sector Legend", padx=8, pady=8)
        self.legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(self.legend_frame, text="(sector map not loaded)").pack(anchor="w")

        # ===== Table pane (bottom of bottom pane)
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


        #   8. Evaluation tab
        self.evalTab = tk.Frame(self.notebook)
        self.notebook.add(self.evalTab, text="Evaluation")
        self.eval_vertical_pane = tk.PanedWindow(self.evalTab, orient=tk.VERTICAL)
        self.eval_vertical_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.eval_horizontal_pane = tk.PanedWindow(self.eval_vertical_pane, orient=tk.HORIZONTAL)
        self.eval_vertical_pane.add(self.eval_horizontal_pane, stretch="always")
        self.loss_pane_frame = tk.LabelFrame(self.eval_vertical_pane, text="Training Loss", padx=5, pady=5)
        self.eval_vertical_pane.add(self.loss_pane_frame, stretch="always")
        self.lstm_eval_frame = tk.LabelFrame(self.eval_horizontal_pane, text="LSTM Evaluation", padx=5, pady=5)
        self.stgnn_eval_frame = tk.LabelFrame(self.eval_horizontal_pane, text="STGNN Evaluation", padx=5, pady=5)
        self.eval_horizontal_pane.add(self.lstm_eval_frame, stretch="always")
        self.eval_horizontal_pane.add(self.stgnn_eval_frame, stretch="always")
        self.lstm_eval_pane = tk.PanedWindow(self.lstm_eval_frame, orient=tk.VERTICAL)
        self.lstm_eval_pane.pack(fill=tk.BOTH, expand=True)
        self.stgnn_eval_pane = tk.PanedWindow(self.stgnn_eval_frame, orient=tk.VERTICAL)
        self.stgnn_eval_pane.pack(fill=tk.BOTH, expand=True)
        self.loss_pane = tk.Frame(self.loss_pane_frame)
        self.loss_pane.pack(fill=tk.BOTH, expand=True)
        self.gru_eval_frame = tk.LabelFrame(self.eval_horizontal_pane, text="GRU Evaluation", padx=5, pady=5)
        self.eval_horizontal_pane.add(self.gru_eval_frame, stretch="always")
        self.gru_eval_pane = tk.PanedWindow(self.gru_eval_frame, orient=tk.VERTICAL)
        self.gru_eval_pane.pack(fill=tk.BOTH, expand=True)

        #   9. Backtesting tab
        self.backtestTab = ttk.Frame(self.notebook)
        self.notebook.add(self.backtestTab, text='Backtesting')
        self.backtest_lstm_pane = tk.LabelFrame(self.backtestTab, text="LSTM Backtest")
        self.backtest_lstm_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.backtest_stgnn_pane = tk.LabelFrame(self.backtestTab, text="STGNN Backtest")
        self.backtest_stgnn_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.backtest_lstm_pane = tk.Frame(self.backtest_lstm_pane)
        self.backtest_stgnn_pane = tk.Frame(self.backtest_stgnn_pane)
        self.backtest_lstm_pane.pack(fill=tk.BOTH, expand=True)
        self.backtest_stgnn_pane.pack(fill=tk.BOTH, expand=True)
        self.backtest_gru_pane = tk.LabelFrame(self.backtestTab, text="GRU Backtest")
        self.backtest_gru_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.backtest_gru_pane = tk.Frame(self.backtest_gru_pane)
        self.backtest_gru_pane.pack(fill=tk.BOTH, expand=True)

        #   10. Metrics tab
        self.metricsTab = tk.Frame(self.notebook)
        self.notebook.add(self.metricsTab, text="Metrics")
        self.metrics_pane = tk.PanedWindow(self.metricsTab, orient=tk.HORIZONTAL)
        self.metrics_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.lstm_metrics_frame = tk.LabelFrame(self.metrics_pane, text="LSTM Metrics", padx=5, pady=5)
        self.metrics_pane.add(self.lstm_metrics_frame, stretch="always")
        self.lstm_metrics_pane = tk.PanedWindow(self.lstm_metrics_frame, orient=tk.VERTICAL)
        self.lstm_metrics_pane.pack(fill=tk.BOTH, expand=True)
        self.lstm_roc_pane = tk.LabelFrame(self.lstm_metrics_pane, text="ROC / PR Curve")
        self.lstm_metrics_pane.add(self.lstm_roc_pane)
        self.lstm_threshold_pane = tk.LabelFrame(self.lstm_metrics_pane, text="Threshold Curve")
        self.lstm_metrics_pane.add(self.lstm_threshold_pane)
        self.stgnn_metrics_frame = tk.LabelFrame(self.metrics_pane, text="STGNN Metrics", padx=5, pady=5)
        self.metrics_pane.add(self.stgnn_metrics_frame, stretch="always")
        self.stgnn_metrics_pane = tk.PanedWindow(self.stgnn_metrics_frame, orient=tk.VERTICAL)
        self.stgnn_metrics_pane.pack(fill=tk.BOTH, expand=True)
        self.stgnn_roc_pane = tk.LabelFrame(self.stgnn_metrics_pane, text="ROC / PR Curve")
        self.stgnn_metrics_pane.add(self.stgnn_roc_pane)
        self.stgnn_threshold_pane = tk.LabelFrame(self.stgnn_metrics_pane, text="Threshold Curve")
        self.stgnn_metrics_pane.add(self.stgnn_threshold_pane)
        self.gru_metrics_frame = tk.LabelFrame(self.metrics_pane, text="GRU Metrics", padx=5, pady=5)
        self.metrics_pane.add(self.gru_metrics_frame, stretch="always")
        self.gru_metrics_pane = tk.PanedWindow(self.gru_metrics_frame, orient=tk.VERTICAL)
        self.gru_metrics_pane.pack(fill=tk.BOTH, expand=True)
        self.gru_roc_pane = tk.LabelFrame(self.gru_metrics_pane, text="ROC / PR Curve")
        self.gru_metrics_pane.add(self.gru_roc_pane)
        self.gru_threshold_pane = tk.LabelFrame(self.gru_metrics_pane, text="Threshold Curve")
        self.gru_metrics_pane.add(self.gru_threshold_pane)

        #   11. Placeholders & state
        self._compute_callback = None
        self._trainers = {}
        self.priceHistory = {}
        self.currentWindowIdx = 0

        #   12. Evaluation logic (must come AFTER panes exist)
        self.evaluator = EvaluationMethods(self)
        self.evaluator.reset_histories()

    def bindMainApp(self, main_app):
        # ====================================
		# === Helper to bind main app to front-end
        self._main_app = main_app
        interval = self._main_app.args.interval
        ticker = self.stockVar.get()

        #   1. Cache trimmed versions
        for t in self._main_app.args.tickers:
            if t in self._main_app.raw_feature_dfs:
                raw = self._main_app.raw_feature_dfs[t]
                trimmed = self._refresh_ticker_history(t, raw, num_bars=None)
                self.priceHistory[t] = trimmed
            else:
                print(f"[WARNING] Ticker '{t}' not found in raw_feature_dfs.")

        #   2. GUI state setup for selected ticker
        df_trimmed = self.priceHistory[ticker]
        data_len = len(df_trimmed)
        est_days = self.bars_to_days(data_len, interval)
        start = df_trimmed.index[0]
        end = df_trimmed.index[-1]
        duration = end - start

        #   3. Initialise interactable components
        self.windowOptions = list(range(1, data_len + 1))
        self.currentWindowIdx = data_len - 1
        self.zoomSlider.config(from_=1, to=data_len, resolution=1)
        self.zoomSlider.set(self.currentWindowIdx + 1)

        print(f"[DEBUG] {ticker}: {data_len} bars ({interval}) ≈ {est_days:.1f} trading days, "
            f"{duration.days} calendar days from {start.date()} to {end.date()} → windowOptions=1…{data_len}")

        #   4. Plot using cached raw data
        raw_df = self._main_app.raw_feature_dfs[ticker]
        self._refresh_and_plot(ticker, raw_df, num_bars=self.currentWindowIdx + 1)

    def _onCompute(self):
        # ====================================
		# === Helper to attach main computation pipline

        #   1. Stop and start computation
        self.stop_event.clear()
        self.btnCompute.config(state=tk.DISABLED, text="…Running")
        self.btnStop.config(state=tk.NORMAL)  # Ensures Stop is always clickable
        self.updateProgress(0.0)

        #   2. Launch background thread
        worker = threading.Thread(
            target=self._run_and_capture,
            args=(self.windowVar.get(), self.stockVar.get()),
            daemon=True
        )
        worker.start()

    def setComputeCallback(self, cb):
        # ====================================
		# === Helper to start main computation pipeline on button click
        self._compute_callback = cb

    def _run_and_capture(self, window, stock):
        # ====================================
		# === Helper to run main computation pipline
        cancelled = False
        try:
            metrics = self._compute_callback(window, stock, self.stop_event)
            if self.stop_event.is_set():
                cancelled = True
            else:
                self.root.after(0, lambda:
                    self.updateResults(*metrics if metrics else ("—", 0.0, "—", 0.0, "—", 0.0))
                )
        except Cancelled:
            cancelled = True
        finally:
            if cancelled:
                self.root.after(0, self._reset_ui)
            else:
                self.root.after(0, lambda: (
                    self.btnCompute.config(state=tk.NORMAL, text="Compute ▶"),
                    self.btnStop.config(state=tk.DISABLED)
                ))

    def _plot_df(self, df_view: pd.DataFrame, label: str):
        # ====================================
		# === Helper to plot price history

        #   1. Clear previous
        self.priceAx.clear()

        #   2. Identify price column
        candidates = [c for c in df_view.columns if c.lower() == 'close']
        if not candidates:
            return
        price_col = candidates[0]

        #   3. Plot line
        self.priceAx.plot(
            df_view.index,
            df_view[price_col],
            label=label,
            linewidth=1.5,
        )

        #   4. Lock x-axis to data bounds
        self.priceAx.set_xlim(df_view.index[0], df_view.index[-1])

        #   5. Minimal styling
        for spine in ("top", "right"):
            self.priceAx.spines[spine].set_visible(False)
        if df_view.index[0] < df_view.index[-1]:
            self.priceAx.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        else:
            self.priceAx.yaxis.grid(False)
        if df_view.index[0] < df_view.index[-1]:
            self.priceAx.set_xlim(df_view.index[0], df_view.index[-1])
        self.priceAx.tick_params(
            axis="both",
            direction="in",
            length=4,
            width=0.5,
            labelsize="small"
        )
        leg = self.priceAx.legend(frameon=False, fontsize="small", loc="upper left")
        leg.set_alpha(0.8)
        self.priceAx.set_title(f"{label} Price History")
        self.priceAx.set_xlabel("Date")
        self.priceAx.set_ylabel("Closing Price (USD)")
        self.priceFig.tight_layout(pad=1.0)
        self.priceCanvas.draw()

    def _onSelectionChange(self, event=None):
        # ====================================
		# === Helper to update front-end for chosen stock

        #   1. Which ticker was chosen?
        ticker = self.stockVar.get()

        #   2. Fetch the raw history for that ticker
        df_raw = self._main_app.priceHistory.get(ticker)

        #   3. Compute & store the full trimmed history (drop leading zeros)
        df_full_trimmed = self._refresh_ticker_history(ticker, df_raw, num_bars=None)
        self.priceHistory[ticker] = df_full_trimmed

        #   4. Rebuild windowOptions & reset to full span
        data_len              = len(df_full_trimmed)
        self.windowOptions    = list(range(1, data_len + 1))
        self.currentWindowIdx = data_len - 1
        self.zoomSlider.config(
            from_=1,
            to=data_len,
            resolution=1
        )
        self.zoomSlider.set(self.currentWindowIdx + 1)

        #   6. One call to refresh + plot using the new wrapper
        self._refresh_and_plot(
            ticker,
            df_raw,
            num_bars=self.currentWindowIdx + 1
        )

    def _refresh_ticker_history(self,
                                ticker: str,
                                df: pd.DataFrame,
                                num_bars: int = None) -> pd.DataFrame:
        # ====================================
		# === Helper to refresh stock history

        #   1. Normalise index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df.get('Date', df.index)), drop=False)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.tz_convert(None)

        #   2. Drop leading zeros
        price_col = 'Close' if 'Close' in df.columns else 'close'
        mask      = df[price_col] > 0
        if mask.any():
            first_real = mask.idxmax()
            df         = df.loc[first_real:]

        #   3. If the caller asked for only the last N bars, slice by iloc
        if num_bars is not None:
            # clamp into [1, len(df)]
            num_bars = max(1, min(len(df), num_bars))
            df       = df.iloc[-num_bars:]
        return df
    
    def _refresh_and_plot(self, ticker: str, raw_df: pd.DataFrame, num_bars: int):
        # ====================================
		# === Helper to refresh and plot price history

        #   1. Refresh history
        df_view = self._refresh_ticker_history(ticker, raw_df, num_bars)

        #   2. Prevent plotting if there's only 1 row (can't set xlim)
        if len(df_view) < 2:
            return  # or optionally plot with a dummy previous day for visual continuity

        #   3. Cache full trimmed history if not already stored
        if num_bars is None:
            self.priceHistory[ticker] = df_view.copy()

        #   4. Plot history
        self._plot_df(df_view, label=ticker)

    def _onPriceScroll(self, event):
        # ====================================
		# === Helper to allow for scroll wheel on price graph (fine adjustment)
        old_idx = self.currentWindowIdx

        #   1. Logarithmic step size
        raw_step = math.log(self.currentWindowIdx + 2)
        step     = max(1, int(raw_step ** 2 / 4))
        delta    = step if event.step > 0 else -step

        #   2. Update & clamp
        new_idx = self.currentWindowIdx + delta
        max_idx = len(self.windowOptions) - 1
        self.currentWindowIdx = max(0, min(new_idx, max_idx))

        #   3. If nothing changed, do nothing
        if self.currentWindowIdx == old_idx:
            return

        #   4. move the slider thumb to match (slider values run 1…len)
        self.zoomSlider.set(self.currentWindowIdx + 1)

        #   5. Refresh & plot just the new slice
        num_bars = self.currentWindowIdx + 1
        ticker = self.stockVar.get()
        if ticker not in self.priceHistory:
            return
        df_full = self.priceHistory[ticker]

        #   6. Refresh and plot
        self._refresh_and_plot(ticker, df_full, num_bars=num_bars)

    def _onSliderChange(self, value_str):
        # ====================================
		# === Helper to allow for zoom slider on price graph (bulk adjustment)
        try:
            val = int(value_str)
        except ValueError:
            return

        #   1. Convert slider value (1…data_len) to index (0…data_len-1)
        new_idx = val - 1
        if new_idx == self.currentWindowIdx:
            return

        #   2. Clamp just in case
        max_idx = len(self.windowOptions) - 1
        self.currentWindowIdx = max(0, min(new_idx, max_idx))

        #   3. Refresh & plot
        ticker = self.stockVar.get()
        if ticker not in self.priceHistory:
            return
        raw_df = self.priceHistory[ticker]
        self._refresh_and_plot(ticker, raw_df, num_bars=self.currentWindowIdx + 1)

    def updateTable(self, feature_df):
        # ====================================
		# === Helper to update table
        if feature_df is None or feature_df.empty:
            return

        #   1. Clear contents
        self.table.configure(takefocus=False)
        self.table.delete(*self.table.get_children())
        self.table["displaycolumns"] = ("Stock", "Return", "Volatility", "Volume", "Momentum")

        #   2. Grab contents
        rows = []
        for idx, row in feature_df.iterrows():
            try:
                vals = (
                    idx,
                    f"{row.get('return', np.nan):.4f}",
                    f"{row.get('volatility', np.nan):.4f}",
                    f"{row.get('volume', np.nan):.4f}",
                    f"{row.get('momentum', np.nan):.4f}"
                )
                rows.append(vals)
            except Exception as e:
                print(f"Skipping row {idx} due to error: {e}")

        #   3. Insert all rows at once with .insert()
        for vals in rows:
            self.table.insert("", "end", values=vals)
        self.table.configure(takefocus=True)

    def plot3d_on_ax(
        self,
        tickers: List[str],
        coords: np.ndarray,
        pruned_edges: List[Tuple[int,int,float]] = None,
        mst_edges   : List[Tuple[int,int,float]] = None,
        ax=None
    ) -> None:
		# ====================================
		# === Helper to plot stock graph on front-end

        #   1. Default to embedded 3D Axes
        if ax is None:
            ax = self.graphAx

        #   2. Normalise into a [-1,1] cube
        coords = coords - coords.mean(axis=0)
        max_abs = abs(coords).max()
        if max_abs > 0:
            coords = coords / max_abs
        coords = coords * 0.8  # leave 10% margin

        #   3. Stash for later lookup
        xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
        self._last_drawn_tickers = tickers
        self._last_drawn_pos = {t: (xs[i], ys[i], zs[i]) for i, t in enumerate(tickers)}
        self._last_pruned_edges = pruned_edges
        self._last_mst_edges = mst_edges

        #   4. Clear & scatter
        ax.clear()

        # Sector-based colouring
        ticker_to_sector = getattr(self, "ticker_to_sector", {}) or {}
        sectors = [ticker_to_sector.get(t, "Unknown") for t in tickers]

        # single source of truth for this draw
        self._sector_to_colour = self._sector_palette(sectors)

        node_colours = [self._sector_to_colour.get(s, (0.6, 0.6, 0.6, 1.0)) for s in sectors]
        ax.scatter(xs, ys, zs, s=75, depthshade=False, c=node_colours, alpha=1.0)

        # keep legend synced with *current* palette
        self.update_sector_legend()

        # labels
        for i, tkr in enumerate(tickers):
            ax.text(xs[i], ys[i], zs[i], tkr, size=6, alpha=0.95)

        # 5. Draw edges (pruned & MST)
        self._last_edge_labels = []  # reset cache

        if pruned_edges:
            for i, j, w in pruned_edges:
                ax.plot(
                    [xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color='blue', linewidth=1.5, alpha=0.5
                )
                mx, my, mz = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2, (zs[i] + zs[j]) / 2
                ax.text(mx, my, mz, f"{w:.2f}", fontsize=5, color='black', alpha=0.6)
                # Store distinct midpoints and weights
                self._last_edge_labels.append(((i, j), (mx, my, mz), w))

        if mst_edges:
            for i, j, w in mst_edges:
                ax.plot(
                    [xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color='red', linewidth=2.0, alpha=0.7, linestyle='--'
                )
                mx, my, mz = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2, (zs[i] + zs[j]) / 2
                self._last_edge_labels.append(((i, j), (mx, my, mz), w))

        #   7. Fix axes to a cube
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.grid(False)
        for axis in ('x','y','z'):
            getattr(ax, f"{axis}axis").pane.fill = False
        fig = ax.get_figure()
        ax.set_position([0,0,1,1])
        ax.set_axis_off()

        #   8. Initialise view & redraw
        ax.view_init(elev=20, azim=30)
        fig.canvas.draw_idle()

    def setTrainers(self, lstm=None, stgnn=None):
        # ====================================
		# === Helper to set trainers for front-end updates
        self._trainers['lstm'] = lstm
        self._trainers['stgnn'] = stgnn

    def updateProgress(self, fraction: float):
        # ====================================
		# === Helper to update training progress
        frac = max(0.0, min(1.0, fraction))
        self.progressVar.set(frac)
        self.root.update_idletasks()

    def bars_to_days(self, bar_count: int, interval: str) -> float:
        # ====================================
		# === Helper to map prediction days to interval bars
        BARS_PER_DAY_MAP = {
            '1m': 360,
            '5m': 72,
            '15m': 24,
            '30m': 12,
            '1h': 6,
            '1d': 1,
            '1wk': 1 / 5,
            '1mo': 1 / 21,
        }
        bars_per_day = BARS_PER_DAY_MAP.get(interval, 1)  # fallback to 1 if unknown
        return bar_count / bars_per_day
    
    def _onStop(self):
        # ====================================
		# === Helper to terminate processes

        #   1. Signal cancellation
        self.stop_event.set()
        self.set_status("Cancelling")

        #   2. Disable Stop button to prevent multiple presses
        self.btnStop.config(state=tk.DISABLED)

    def set_status(self, msg: str):
        # ====================================
		# === Helper to show a status message in the toolbar
        self.root.after(0, lambda: self.statusVar.set(msg))

    def clear_status(self):
        # ====================================
		# === Helper to reset status display
        self.root.after(0, lambda: self.statusVar.set("Idle"))    
    
    def clear_axis(self, pane):
        # ====================================
		# === Helper to clear axis
        if pane is not None:
            for widget in pane.winfo_children():
                widget.destroy()    

    def bindTickerClick(self, fn):
        # ====================================
		# === Helper to bind graph interactability 
        self._onTickerClick = fn

    def _on_table_select(self, event):
        # ====================================
		# === Helper to delegate clicked ticker in table
        sel = self.table.selection()
        if not sel:
            return

        ticker = self.table.item(sel[0], "values")[0]

        # Toggle deselection
        if getattr(self, "_highlighted_ticker", None) == ticker:
            self.table.selection_remove(sel[0])
            self._highlighted_ticker = None
            # Redraw the original graph with weights
            if hasattr(self, "_last_drawn_tickers"):
                tickers = self._last_drawn_tickers
                coords = np.array([self._last_drawn_pos[t] for t in tickers])
                self.plot3d_on_ax(
                    tickers,
                    coords,
                    getattr(self, "_last_pruned_edges", None),
                    getattr(self, "_last_mst_edges", None)
                )
            return

        # Otherwise, highlight new ticker
        self._highlighted_ticker = ticker
        if hasattr(self, "_onTickerClick"):
            self._onTickerClick(ticker)

    def on_ticker_click(self, ticker: str):
        # ====================================
		# === Helper to highlight node selected in table
        x, y, z = self._last_drawn_pos[ticker]
        self.highlight_node(ticker, (x, y, z))
        
    def highlight_node(self, ticker: str, coord: Tuple[float, float, float]):
        # ====================================
		# === Helper to highlight selected node

        # 1. Remove old highlights
        for m in getattr(self, "_highlight_markers", []):
            try:
                m.remove()
            except Exception:
                pass
        self._highlight_markers = []

        # 2. Sanity checks
        if not hasattr(self, "_last_drawn_pos") or not hasattr(self, "_last_drawn_tickers"):
            print("[Warning] No graph data available to highlight.")
            return

        xs, ys, zs = [], [], []
        for t in self._last_drawn_tickers:
            x, y, z = self._last_drawn_pos[t]
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        tickers = self._last_drawn_tickers

        # 3. Compute node distances to selected node
        target = np.array(coord)
        distances = np.linalg.norm(np.stack([xs, ys, zs], axis=1) - target, axis=1)
        max_d = distances.max() if distances.max() > 0 else 1.0
        norm_d = distances / max_d

        # 4. Retrieve stored edges
        pruned_edges = getattr(self, "_last_pruned_edges", None)
        mst_edges = getattr(self, "_last_mst_edges", None)

        ax = self.graphAx
        current_elev, current_azim = ax.elev, ax.azim
        ax.clear()

        # 5. Draw edges with exponential fade
        def edge_alpha(i, j):
            d_mean = (norm_d[i] + norm_d[j]) / 2.0
            return float(np.exp(-2.5 * d_mean))
        
        if pruned_edges:
            for i, j, w in pruned_edges:
                alpha = edge_alpha(i, j)
                ax.plot(
                    [xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color='blue', linewidth=1.2, alpha=alpha
                )

        if mst_edges:
            for i, j, _ in mst_edges:
                alpha = edge_alpha(i, j)
                ax.plot(
                    [xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color='red', linewidth=1.8, linestyle='--', alpha=alpha
                )

        if hasattr(self, "_last_edge_labels"):
            for (i_j, (mx, my, mz), w) in self._last_edge_labels:
                i, j = i_j
                d_mean = (norm_d[i] + norm_d[j]) / 2.0
                alpha = float(np.exp(-2.5 * d_mean))
                ax.text(
                    mx, my, mz, f"{w:.2f}",
                    fontsize=5, color='black',
                    alpha=alpha, ha='center', va='center'
                )

        # 6. Scatter all nodes unchanged
        ticker_to_sector = getattr(self, "ticker_to_sector", {}) or {}
        sector_to_colour = getattr(self, "_sector_to_colour", {}) or {}

        for i, t in enumerate(tickers):
            sector = ticker_to_sector.get(t, "Unknown")
            base_colour = sector_to_colour.get(sector, (0.5, 0.5, 0.5, 1.0))

            is_sel = (t == ticker)
            size = 90 if is_sel else 45

            ax.scatter(
                [xs[i]], [ys[i]], [zs[i]],
                s=size,
                depthshade=True,
                alpha=0.98,
                facecolors=[base_colour],
                edgecolors="red" if is_sel else "black",
                linewidths=2.0 if is_sel else 0.6,
                zorder=6 if is_sel else 3
            )
            ax.text(xs[i], ys[i], zs[i], t, size=6, alpha=1.0)

        # 7. Halo highlight around selected node
        x, y, z = coord
        halo = ax.scatter(
            [x], [y], [z],
            s=400,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=10
        )
        self._highlight_markers.append(halo)

        # 8. View and render
        ax.view_init(elev=current_elev, azim=current_azim)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        for axis in ('x', 'y', 'z'):
            getattr(ax, f"{axis}axis").pane.fill = False
        ax.grid(False)
        ax.set_axis_off()
        self.graphCanvas.draw_idle()

    # ====================================
    # === Helper to expose backtest frame for LSTM
    @property
    def backtestLSTM(self):
        return self.backtest_lstm_pane
    
    # ====================================
    # === Helper to expose backtest frame for STGNN
    @property
    def backtestSTGNN(self):
        return self.backtest_stgnn_pane
    
    def _reset_ui(self):
        # ====================================
		# === Helper to reset the UI

        #   1. Clear processes
        self.clear_status()
        self.btnCompute.config(state=tk.NORMAL, text="Compute ▶")
        self.updateProgress(0.0)
        try:
            #   2. Provide fallback placeholders (safe defaults)
            self.updateResults("—", 0.0, "—", 0.0, "—", 0.0)
        except Exception as e:
            print(f"[WARNING] updateResults failed during UI reset: {e}")

        #   3. Clear graph
        self.graphAx.clear()
        self.graphCanvas.draw()

        #   4. Clear table
        for row in self.table.get_children():
            self.table.delete(row)

        #   5. Clear all evaluation panes in the GUI thread
        clear_targets = [
            self.lstm_eval_pane,
            self.gru_eval_pane,
            self.stgnn_eval_pane,
            self.backtest_lstm_pane,
            self.backtest_gru_pane,
            self.backtest_stgnn_pane,
            self.lstm_roc_pane,
            self.gru_roc_pane,
            self.stgnn_roc_pane,
            self.lstm_threshold_pane,
            self.gru_threshold_pane,
            self.stgnn_threshold_pane
        ]

        self.root.after(0, lambda: [self.clear_axis(pane) for pane in clear_targets])    

    def updateResults(self, lstm_trend, lstm_conf, gru_trend, gru_conf, stgnn_trend, stgnn_conf):
        """Update the three result label boxes."""
        # LSTM
        self.lstmStatus.config(
            text=f"Trending: {lstm_trend}",
            fg="green" if "Upwards" in lstm_trend else "red"
        )
        self.lstmConf.config(
            text=f"Confidence: {lstm_conf:.1f}%",
            fg="green" if "Upwards" in lstm_trend else "red"
        )

        # GRU
        self.gruStatus.config(
            text=f"Trending: {gru_trend}",
            fg="green" if "Upwards" in gru_trend else "red"
        )
        self.gruConf.config(
            text=f"Confidence: {gru_conf:.1f}%",
            fg="green" if "Upwards" in gru_trend else "red"
        )

        # STGNN
        self.stgnnStatus.config(
            text=f"Trending: {stgnn_trend}",
            fg="green" if "Upwards" in stgnn_trend else "red"
        )
        self.stgnnConf.config(
            text=f"Confidence: {stgnn_conf:.1f}%",
            fg="green" if "Upwards" in stgnn_trend else "red"
        )
    
    def refresh_selected_tabs(self):
        # ====================================
		# === Helper to refresh UI
        for tab in [self.metricsTab, self.evalTab, self.backtestTab]:
            if tab:
                tab.update_idletasks()
        #   1. Rebalancing
        def _rebalance():
            try:
                # metricsPane (horizontal)
                if hasattr(self, 'metrics_pane') and isinstance(self.metrics_pane, tk.PanedWindow):
                    self.metrics_pane.update_idletasks()
                    w = self.metrics_pane.winfo_width()
                    if w > 10:
                        self.metrics_pane.sash_place(0, w // 2, 0)

                # eval_vertical_pane (top vs bottom)
                if hasattr(self, 'eval_vertical_pane') and isinstance(self.eval_vertical_pane, tk.PanedWindow):
                    self.eval_vertical_pane.update_idletasks()
                    h = self.eval_vertical_pane.winfo_height()
                    if h > 10:
                        self.eval_vertical_pane.sash_place(0, 0, int(h * 0.5))

                # eval_horizontal_pane (LSTM vs STGNN)
                if hasattr(self, 'eval_horizontal_pane') and isinstance(self.eval_horizontal_pane, tk.PanedWindow):
                    self.eval_horizontal_pane.update_idletasks()
                    w = self.eval_horizontal_pane.winfo_width()
                    if w > 10:
                        self.eval_horizontal_pane.sash_place(0, w // 2, 0)

            except Exception as e:
                print("Warning: Failed to rebalance panes:", e)

        self.root.after_idle(_rebalance)

    def set_sector_map(self, ticker_to_sector: dict):
        """
        Optional: set mapping {ticker -> sector}. If not set, nodes default to 'Unknown'.
        Also refreshes the external legend panel.
        """
        self.ticker_to_sector = ticker_to_sector or {}
        self.update_sector_legend()


    def _sector_palette(self, sectors: List[str]):
        """
        Deterministic palette for sectors present in current plot.
        """
        sectors = sorted(set(sectors))
        # tab20 supports up to 20 distinct colours reasonably well
        cmap = cm.get_cmap("tab20", max(len(sectors), 1))
        return {s: cmap(i) for i, s in enumerate(sectors)}

    def update_sector_legend(self):
        """
        Render the sector legend in the dedicated Tkinter panel (not inside matplotlib).
        """
        if not hasattr(self, "legend_frame"):
            return

        # Clear previous legend widgets
        for w in self.legend_frame.winfo_children():
            w.destroy()

        ticker_to_sector = getattr(self, "ticker_to_sector", {}) or {}
        if not ticker_to_sector:
            tk.Label(self.legend_frame, text="No sector data").pack(anchor="w")
            return

        sector_to_colour = getattr(self, "_sector_to_colour", None)
        if not sector_to_colour:
            tk.Label(self.legend_frame, text="No graph drawn yet").pack(anchor="w")
            return

        sectors = sorted(sector_to_colour.keys())

        for sector in sectors:
            row = tk.Frame(self.legend_frame)
            row.pack(anchor="w", pady=2)

            rgba = sector_to_colour[sector]
            hex_colour = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
            )

            swatch = tk.Canvas(row, width=14, height=14, bg=hex_colour, highlightthickness=1)
            swatch.pack(side=tk.LEFT, padx=(0, 6))

            tk.Label(row, text=sector).pack(side=tk.LEFT)
