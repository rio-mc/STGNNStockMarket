from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Optional
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from evaluation_types import EvaluationResult

class EvaluationMethods:
    """
    Encapsulates evaluation logic for LSTM and STGNN models,
    producing loss curves, confusion heatmaps, classification metrics,
    ROC/PR curves, and additional statistics (MCC, Kappa, Balanced Accuracy).
    """
    def __init__(self, frontend):
        #   1. Set up from front-end
        self.frontend = frontend
        self.lstm_eval_pane = frontend.lstm_eval_pane
        self.gru_eval_pane = frontend.gru_eval_pane
        self.stgnn_eval_pane = frontend.stgnn_eval_pane
        self.loss_pane = frontend.loss_pane

        #   2. Configure loss pane
        self.loss_pane.rowconfigure(0, weight=1)
        self.loss_pane.rowconfigure(1, weight=1)
        self.loss_pane.columnconfigure(0, weight=1)
        self.train_frame = tk.Frame(self.loss_pane)
        self.train_frame.grid(row=0, column=0, sticky='nsew')
        self.val_frame = tk.Frame(self.loss_pane)
        self.val_frame.grid(row=1, column=0, sticky='nsew')
        self.loss_train_fig = Figure(figsize=(10, 3))
        self.ax_train = self.loss_train_fig.add_subplot(111)
        self.loss_train_canvas = FigureCanvasTkAgg(self.loss_train_fig, master=self.train_frame)
        self.loss_train_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.loss_val_fig = Figure(figsize=(10, 3))
        self.ax_val = self.loss_val_fig.add_subplot(111)
        self.loss_val_canvas = FigureCanvasTkAgg(self.loss_val_fig, master=self.val_frame)
        self.loss_val_canvas.get_tk_widget().pack(fill='both', expand=True)

        #   3. Configure backtest pane
        self.backtest_lstm_pane = frontend.backtest_lstm_pane
        self.backtest_gru_pane = frontend.backtest_gru_pane
        self.backtest_stgnn_pane = frontend.backtest_stgnn_pane

        #   4. Configure metrics pane
        self.lstm_roc_pane = frontend.lstm_roc_pane
        self.gru_roc_pane = frontend.gru_roc_pane
        self.stgnn_roc_pane = frontend.stgnn_roc_pane
        self.lstm_threshold_pane = frontend.lstm_threshold_pane
        self.gru_threshold_pane = frontend.gru_threshold_pane
        self.stgnn_threshold_pane = frontend.stgnn_threshold_pane

        #   5. Initialise histories
        self.loss_history_lstm = []
        self.loss_history_gru = []
        self.loss_history_stgnn = []
        self.val_loss_history_lstm = []
        self.val_loss_history_gru = []
        self.val_loss_history_stgnn = []

    def evaluate(
        self,
        model_name: str,
        result: EvaluationResult,
        price_df: pd.DataFrame,
    ):
        # === STEP 1: Check Results ===
        # ------------------------------------
        if not result.y_true or not result.y_pred:
            print(f"[{model_name} Evaluation] No predictions to evaluate.")
            return

		# === STEP 2: Compute Accuracy, Precision, Recall, F1-Score ===
        # ------------------------------------
        
        #   1. Accuracy and F1-score
        acc = accuracy_score(result.y_true, result.y_pred)
        f1 = precision_recall_fscore_support(result.y_true, result.y_pred, average="binary")[2]
        print(f"[{model_name} Evaluation] Accuracy = {acc:.3f} | F1 = {f1:.3f}")

        #   2. Confusion Matrix
        self.frontend.root.after(0, lambda: self.plot_confusion_matrix(result.y_true, result.y_pred, model_name))

        #   3. ROC & PR
        self.frontend.root.after(0, lambda: self.plot_roc_and_pr(result.y_true, result.probs, model_name))

        #   4. Threshold Curve
        if model_name.upper() == "LSTM":
            threshold_pane = self.lstm_threshold_pane
        elif model_name.upper() == "GRU":
            threshold_pane = self.gru_threshold_pane
        else:
            threshold_pane = self.stgnn_threshold_pane

        self.frontend.root.after(0, lambda: self.plot_recall_threshold(
            truths=result.y_true,
            probs=result.probs,
            pane=threshold_pane,
            model_name=model_name
        ))

        # === STEP 3: Backtesting And Equity Curve ===
        # ------------------------------------
        if result.prediction_dates and result.horizon:
            #   1. Initialise equity with baseline of 1.0
            equity = [1.0]

            #   2. Prepare labels
            x_labels = result.prediction_dates
            price_series = price_df["close"].copy()
            price_series.index = price_series.index.tz_localize(None)

            #   3. Iterate over each prediction
            for i, date in enumerate(result.prediction_dates):
                date = pd.Timestamp(date).tz_localize(None)

                if date not in price_series.index:
                    equity.append(equity[-1])
                    continue
                #   4. Entry price at prediction date
                try:
                    entry_price = price_series.loc[date]
                    exit_idx = price_series.index.get_loc(date) + result.horizon
                    #   5. Carry equity forward if exit index exceeds series length (weekends, bank holidays, etc.)
                    if exit_idx >= len(price_series):
                        equity.append(equity[-1])
                        continue
                    #   6. Exit price 
                    exit_price = price_series.iloc[exit_idx]
                    ret = (exit_price - entry_price) / entry_price

                    #   7. Apply trade direction based on prediction
                    direction = 1 if result.y_pred[i] == 1 else -1
                    equity.append(equity[-1] * (1 + direction * ret))

                except Exception as e:
                    print(f"[EQUITY] Failed at {date}: {e}")
                    equity.append(equity[-1])

		    # === STEP 4: Compute Sharpe ratio from equity curve ===
            # ------------------------------------
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

            # === STEP 5: Plot Equity Curve ===
            # ------------------------------------
            pane = self.backtest_lstm_pane if model_name.upper() == "LSTM" \
            else self.backtest_gru_pane if model_name.upper() == "GRU" \
            else self.backtest_stgnn_pane

            self.frontend.root.after(0, lambda: self.plot_equity_curve(
                equity=equity,
                pane=pane,
                label=model_name,
                sharpe=sharpe,
                x_labels=x_labels
            ))

        # === STEP 6: Plot Losses ===
        # ------------------------------------
        if model_name.upper() == "LSTM":
            self._val_lstm = [v["loss"] for v in result.val_lstm] if result.val_lstm else []
            self._hist_lstm = result.hist_lstm or []
        elif model_name.upper() == "GRU":
            self._val_gru = [v["loss"] for v in getattr(result, "val_gru", [])] if getattr(result, "val_gru", None) else []
            self._hist_gru = getattr(result, "hist_gru", []) or []
        else:
            self._val_stgnn = [v["loss"] for v in result.val_stgnn] if result.val_stgnn else []
            self._hist_stgnn = result.hist_stgnn or []

    def plot_loss(
        self,
        hist_l: Optional[List[float]] = None,
        hist_s: Optional[List[float]] = None,
        hist_g: Optional[List[float]] = None,
        val_l: Optional[List[float]] = None,
        val_s: Optional[List[float]] = None,
        val_g: Optional[List[float]] = None
    ):
        # === STEP 1: Grab Histories ===
        # ------------------------------------
        hist_l = hist_l if hist_l is not None else self.loss_history_lstm
        hist_s = hist_s if hist_s is not None else self.loss_history_stgnn
        hist_g = hist_g if hist_g is not None else self.loss_history_gru
        val_l = val_l if val_l is not None else self.val_loss_history_lstm
        val_s = val_s if val_s is not None else self.val_loss_history_stgnn
        val_g = val_g if val_g is not None else self.val_loss_history_gru

        # === STEP 2: Clear Plots ===
        # ------------------------------------
        self.ax_train.clear()
        self.ax_val.clear()

        # === STEP 3: Training Plot And Formatting ===
        # ------------------------------------
        epochs = list(range(1, max(len(hist_l), len(hist_s), len(hist_g)) + 1))
        if hist_l:
            self.ax_train.plot(epochs[:len(hist_l)], hist_l, label='LSTM Train', linewidth=1.5)
        if hist_s:
            self.ax_train.plot(epochs[:len(hist_s)], hist_s, label='STGNN Train', linewidth=1.5)
        if hist_g:
            self.ax_train.plot(epochs[:len(hist_g)], hist_g, label='GRU Train', linewidth=1.5)
        max_ticks = 10
        if len(epochs) > max_ticks:
            step = max(1, len(epochs) // max_ticks)
            ticks_to_show = epochs[::step]
        else:
            ticks_to_show = epochs
        self.ax_train.set_xticks(ticks_to_show)
        self.ax_train.set_xticklabels([str(e) for e in ticks_to_show], rotation=0, ha='center')
        self.ax_train.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{int(val)}'))
        self.ax_train.set_xlabel('Epoch')
        self.ax_train.set_title('Training Loss')
        self.ax_train.set_ylabel('Loss')
        self.ax_train.legend(loc='upper right', frameon=False)
        self.ax_train.grid(True, linestyle='--', alpha=0.5)

        # === STEP 4: Validation Plot And Formatting ===
        # ------------------------------------
        self._plot_validation_series(
            self.ax_val,
            self.val_loss_history_lstm,
            'LSTM',
            '--',
            'tab:blue'
        )
        self._plot_validation_series(
            self.ax_val,
            self.val_loss_history_gru,
            'GRU',
            '-.',
            'tab:green'
        )
        self._plot_validation_series(
            self.ax_val,
            self.val_loss_history_stgnn,
            'STGNN',
            ':',
            'tab:orange'
        )
        self.ax_val.set_title('Validation Loss')
        self.ax_val.set_xlabel('Date')
        self.ax_val.set_ylabel('Loss')
        if self.ax_val.get_legend_handles_labels()[0]:
            self.ax_val.legend(loc='upper right', frameon=False)
        self.ax_val.grid(True, linestyle='--', alpha=0.5)
        self.ax_val.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_val.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.loss_val_fig.autofmt_xdate(rotation=30)

        # === STEP 5: Finalise Plots ===
        # ------------------------------------
        self.loss_train_fig.tight_layout(pad=1.0)
        self.loss_train_canvas.draw()
        self.loss_val_fig.tight_layout(pad=1.0)
        self.loss_val_canvas.draw()

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], model_name: str):
        # === STEP 1: Create Labels and Confusion Matrix ===
        # ------------------------------------

        #   1. Convert numeric to label strings for metric computation
        labels = ["Down", "Up"]
        truths = [labels[y] for y in y_true]
        preds  = [labels[p] for p in y_pred]

        #   2. Compute confusion matrix and metrics
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(truths, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(
            truths, preds, labels=labels, zero_division=0
        )

        # === STEP 2: Plot Confusion Matrix ===
        # ------------------------------------

        #   1. Establish which to plot
        if model_name.upper() == "LSTM":
            pane = self.lstm_eval_pane
        elif model_name.upper() == "GRU":
            pane = self.gru_eval_pane
        else:
            pane = self.stgnn_eval_pane

        fig, (ax_cm, ax_txt) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(5, 5),
            gridspec_kw={'height_ratios': [3, 1]}
        )

        #   2. Create and format heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax_cm, square=True, linewidths=0.5, linecolor='lightgrey')
        sns.despine(fig=fig, ax=ax_cm, left=False, bottom=False)
        ax_cm.set_xlabel('Predicted', fontsize=10)
        ax_cm.set_ylabel('Actual',    fontsize=10)
        ax_cm.set_title(f"{model_name} Confusion Matrix", fontsize=12, pad=8)
        ax_cm.tick_params(axis='both', which='major', labelsize=9)
        ax_txt.axis('off')

        # === STEP 3: Metrics table from confusion matrix ===
        # ------------------------------------
        rows = ['Overall'] + labels
        cols = ['Accuracy', 'Precision', 'Recall', 'F1']
        cell_text = [[f"{acc:.2f}", '', '', '']] + [[
            '', f"{pr[i]:.2f}", f"{rc[i]:.2f}", f"{f1[i]:.2f}"
        ] for i in range(len(labels))]
        tbl = ax_txt.table(cellText=cell_text, rowLabels=rows,
                        colLabels=cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)

        # === STEP 4: Formatting ===
        # ------------------------------------
        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        plt.close(fig)

    def plot_roc_and_pr(
        self,
        truths: List[int],
        probs: Optional[List[float]] = None,
        model_name: str = "Model"
    ) -> None:
		# === STEP 1: Create Figures ===
        # ------------------------------------
        if model_name.upper() == "LSTM":
            pane = self.lstm_roc_pane
        elif model_name.upper() == "GRU":
            pane = self.gru_roc_pane
        else:
            pane = self.stgnn_roc_pane
        if pane is None:
            print(f"[WARN] No pane found for {model_name} ROC/PR plot.")
            return
        fig, axes = (plt.subplots(1, 2, figsize=(8, 3)) if probs is not None
                    else (plt.figure(figsize=(5, 4)), plt.gca()))

		# === STEP 2: Calculate Further Metrics ===
        # ------------------------------------

        #   1. ROC curve
        fpr, tpr, _ = roc_curve(truths, probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axes[0].plot([0, 1], [0, 1], '--', color='grey')
        axes[0].set_title(f'{model_name} ROC')
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
        axes[0].legend(loc='lower right')

        #   2. PR curve
        precision, recall, _ = precision_recall_curve(truths, probs, pos_label=1)
        ap_score = average_precision_score(truths, probs, pos_label=1)
        axes[1].plot(recall, precision, label=f'AP = {ap_score:.2f}')
        axes[1].set_title(f'{model_name} PR')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left')

		# === STEP 3: Formatting ===
        # ------------------------------------
        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig if probs is not None else axes.figure, master=pane)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        plt.close(fig if probs is not None else axes.figure)

    def plot_recall_threshold(
        self,
        truths: List[int],
        probs: List[float],
        pane,
        model_name: str
    ) -> None:
		# === STEP 1: Metrics and Thresholds ===
        # ------------------------------------
        precision, recall, thresholds = precision_recall_curve(truths, probs, pos_label=1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # === STEP 2: Formatting ===
        # ------------------------------------
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(thresholds, recall[:-1], label='Recall', linewidth=1.5)
        ax.plot(thresholds, precision[:-1], '--', label='Precision')
        ax.plot(thresholds, f1_scores[:-1], ':', label='F1 Score')
        ax.set_title(f'{model_name} Scores vs Threshold')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()
        plt.close(fig)

    def plot_equity_curve(self, equity, pane, label, sharpe, x_labels=None):
		# === STEP 1: Set Up Plots ===
        # ------------------------------------
        self._clear_pane(pane)
        fig = Figure(figsize=(8, 3))
        ax = fig.add_subplot(111)

        # === STEP 2: Label Creation ===
        # ------------------------------------
        if x_labels and len(equity) == len(x_labels) + 1: 
            delta = x_labels[1] - x_labels[0]
            first_date = x_labels[0] - delta
            x_labels = [first_date] + x_labels
        elif len(equity) != len(x_labels):
            min_len = min(len(equity), len(x_labels))
            equity  = equity[:min_len]
            x_labels = x_labels[:min_len]

        # === STEP 3: Plot Validation Predictions Across Labels ===
        # ------------------------------------
        ax.plot(x_labels if x_labels else range(len(equity)),
                equity,
                label=f'{label} (Sharpe: {sharpe:.2f})',
                linewidth=1.5)

        # === STEP 4: Formatting ===
        # ------------------------------------
        if x_labels and isinstance(x_labels[0], pd.Timestamp):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate(rotation=45)
        ax.set_title(f'{label} Equity Curve')
        ax.set_xlabel('Date' if x_labels else 'Step')
        ax.set_ylabel('Equity (Normalised)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

    def record_training_loss(self, model_name: str, loss: float):
        # ====================================
		# === Helper to record training loss
        if model_name.upper() == "LSTM":
            self.loss_history_lstm.append(loss)
        elif model_name.upper() == "GRU":
            self.loss_history_gru.append(loss)
        elif model_name.upper() == "STGNN":
            self.loss_history_stgnn.append(loss)

    def get_training_loss(self, model_name: str):
        # ====================================
		# === Helper to get training loss
        if model_name.upper() == "LSTM":
            return self.loss_history_lstm
        elif model_name.upper() == "GRU":
            return self.loss_history_gru
        else:
            return self.loss_history_stgnn
    
    def record_validation_loss(self, model_name: str, loss: float, timestamp: pd.Timestamp):
        # ====================================
		# === Helper to record validation loss
        entry = {"loss": loss, "date": timestamp}
        if model_name.upper() == "LSTM":
            self.val_loss_history_lstm.append(entry)
        elif model_name.upper() == "GRU":
            self.val_loss_history_gru.append(entry)
        elif model_name.upper() == "STGNN":
            self.val_loss_history_stgnn.append(entry)

    def get_validation_loss(self, model_name: str):
        # ====================================
		# === Helper to get validation loss
        if model_name.upper() == "LSTM":
            return self.val_loss_history_lstm
        elif model_name.upper() == "GRU":
            return self.val_loss_history_gru
        else:
            return self.val_loss_history_stgnn
    
    def _plot_validation_series(self, ax, loss_history, label_prefix, line_style, colour):
        # ====================================
		# === Helper to plot validation loss (with date)

        #   1. Match predictions to date
        if not loss_history:
            return
        dates = [entry['date'] for entry in loss_history]
        y = np.array([entry['loss'] for entry in loss_history])
        avg = y.mean()
        label = f'{label_prefix} Val (μ={avg:.2f})'
        ax.plot(dates, y, line_style, label=label, linewidth=1.5)

        #   2. Trend line
        trend = np.polyfit(mdates.date2num(dates), y, 1)
        ax.plot(
            dates,
            np.polyval(trend, mdates.date2num(dates)),
            color=colour,
            linewidth=1
        )

    def reset_histories(self):
        # ====================================
		# === Helper to reset histories
        self.loss_history_lstm.clear()
        self.loss_history_stgnn.clear()
        self.loss_history_gru.clear()
        self.val_loss_history_lstm.clear()
        self.val_loss_history_stgnn.clear()
        self.val_loss_history_gru.clear()
    
    def _clear_pane(self, pane):
        # ====================================
		# === Helper to clear panes
        if pane is None:
            return
        for widget in pane.winfo_children():
            widget.destroy()