import tkinter as tk
from tkinter import ttk


class BacktestingTab:
    """
    Single-model backtesting tab.

    Exposes:
    - frame: the top-level tab frame
    - backtest_frame: labelled container for equity/backtest figures
    - backtest_pane: plain frame used by EvaluationMethods for embedded canvases
    """

    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Backtesting")

        self.backtest_frame = tk.LabelFrame(
            self.frame,
            text="Model Backtest",
            padx=5,
            pady=5,
        )
        self.backtest_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.backtest_pane = tk.Frame(self.backtest_frame)
        self.backtest_pane.pack(fill=tk.BOTH, expand=True)

    def set_model_title(self, model_name: str):
        pretty = str(model_name).upper()
        self.backtest_frame.config(text=f"{pretty} Backtest")