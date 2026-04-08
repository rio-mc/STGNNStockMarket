import tkinter as tk


class MetricsTab:
    """
    Single-model metrics tab.

    Exposes:
    - frame: the top-level tab frame
    - metrics_frame: labelled outer frame
    - metrics_model_pane: vertical pane containing metric sections
    - roc_pane: frame for ROC / PR figure
    - threshold_pane: frame for threshold figure
    """

    def __init__(self, notebook):
        self.frame = tk.Frame(notebook)
        notebook.add(self.frame, text="Metrics")

        self.metrics_pane = tk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
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

    def set_model_title(self, model_name: str):
        pretty = str(model_name).upper()
        self.metrics_frame.config(text=f"{pretty} Metrics")
        self.roc_pane.config(text=f"{pretty} ROC / PR Curve")
        self.threshold_pane.config(text=f"{pretty} Threshold Curve")