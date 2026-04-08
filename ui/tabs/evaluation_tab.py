import tkinter as tk


class EvaluationTab:
    """
    Single-model evaluation tab.

    Exposes:
    - frame: the top-level tab frame
    - eval_summary_frame: container for confusion matrix / evaluation summary
    - eval_pane: pane where evaluation figures are rendered
    - loss_pane_frame: labelled container for train/validation loss
    - loss_pane: plain frame used by EvaluationMethods for embedded canvases
    """

    def __init__(self, notebook):
        self.frame = tk.Frame(notebook)
        notebook.add(self.frame, text="Evaluation")

        self.vertical_pane = tk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.vertical_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.eval_summary_frame = tk.LabelFrame(
            self.vertical_pane,
            text="Model Evaluation",
            padx=5,
            pady=5,
        )
        self.vertical_pane.add(self.eval_summary_frame, stretch="always")

        self.eval_pane = tk.PanedWindow(self.eval_summary_frame, orient=tk.VERTICAL)
        self.eval_pane.pack(fill=tk.BOTH, expand=True)

        self.loss_pane_frame = tk.LabelFrame(
            self.vertical_pane,
            text="Training Loss",
            padx=5,
            pady=5,
        )
        self.vertical_pane.add(self.loss_pane_frame, stretch="always")

        self.loss_pane = tk.Frame(self.loss_pane_frame)
        self.loss_pane.pack(fill=tk.BOTH, expand=True)

    def set_model_title(self, model_name: str):
        pretty = str(model_name).upper()
        self.eval_summary_frame.config(text=f"{pretty} Evaluation")
        self.loss_pane_frame.config(text=f"{pretty} Training Loss")