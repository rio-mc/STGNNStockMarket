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
    average_precision_score,
    f1_score
)

from evaluation.evaluation_types import EvaluationResult

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

        self.colour_map = {
            "LSTM":  "tab:blue",
            "GRU":   "tab:green",
            "STGNN": "tab:orange",
        }

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
        best_thr = 0.5
        rule_name = "default (0.5)"

        y_true = np.asarray(result.y_true, dtype=int)

        # Initialise report fields (so return dict always has something sensible)
        acc_05 = f1_05 = macro_f1_05 = None
        acc   = f1   = macro_f1   = None

        # If we have probabilities, compute baseline @0.5 and tuned threshold metrics
        if result.probs is not None and len(result.probs) == len(result.y_true):
            probs = np.asarray(result.probs, dtype=float)

            # ---- baseline @ 0.5 (paper baseline) ----
            y_pred_05 = (probs >= 0.5).astype(int)
            acc_05 = accuracy_score(y_true, y_pred_05)
            f1_05  = precision_recall_fscore_support(y_true, y_pred_05, average="binary")[2]  # F1 for positive class
            macro_f1_05 = f1_score(y_true, y_pred_05, average="macro")

            # ---- tuned threshold (balanced accuracy rule) ----
            best_thr = self._best_threshold_macro_f1(y_true, probs)
            rule_name = "Decision Threshold - Macro-F1"

            y_pred_used = (probs >= best_thr).astype(int)
            acc = accuracy_score(y_true, y_pred_used)
            f1  = precision_recall_fscore_support(y_true, y_pred_used, average="binary")[2]
            macro_f1 = f1_score(y_true, y_pred_used, average="macro")

            print(
                f"[{model_name} Evaluation] "
                f"thr=0.500 | Acc={acc_05:.3f} | F1(pos)={f1_05:.3f} | F1(macro)={macro_f1_05:.3f}  ||  "
                f"{rule_name}={best_thr:.3f} | Acc={acc:.3f} | F1(pos)={f1:.3f} | F1(macro)={macro_f1:.3f}"
            )

        else:
            # Fallback if probs missing: use whatever y_pred was provided
            probs = None
            y_pred_used = np.asarray(result.y_pred, dtype=int)
            acc = accuracy_score(y_true, y_pred_used)
            f1  = precision_recall_fscore_support(y_true, y_pred_used, average="binary")[2]
            macro_f1 = f1_score(y_true, y_pred_used, average="macro")

            print(
                f"[{model_name} Evaluation] thr=N/A | Acc={acc:.3f} | F1(pos)={f1:.3f} | F1(macro)={macro_f1:.3f}"
            )

        # after acc/f1
        roc_auc = None
        ap = None
        if probs is not None:
            fpr, tpr, _ = roc_curve(y_true, probs, pos_label=1)
            roc_auc = auc(fpr, tpr)
            ap = average_precision_score(y_true, probs, pos_label=1)

        # Confusion Matrix / ROC-PR / Threshold
        self.frontend.root.after(
            0,
            lambda: self.plot_confusion_matrix(y_true, y_pred_used, model_name, thr=best_thr, rule_name=rule_name)
        )

        if probs is not None:
            self.frontend.root.after(
                0,
                lambda: self.plot_roc_and_pr(y_true, probs, model_name)
            )

        if model_name.upper() == "LSTM":
            threshold_pane = self.lstm_threshold_pane
        elif model_name.upper() == "GRU":
            threshold_pane = self.gru_threshold_pane
        else:
            threshold_pane = self.stgnn_threshold_pane

        if probs is not None:
            self.frontend.root.after(
                0,
                lambda: self.plot_recall_threshold(
                    truths=y_true,
                    probs=probs,
                    pane=threshold_pane,
                    model_name=model_name,
                    best_thr=best_thr,
                    rule_name=rule_name
                )
            )


        # === STEP 3: Backtesting and Equity Curve (non-overlapping trades) ===
        # ------------------------------------
        sharpe = None
        if result.prediction_dates and result.horizon:
            # 0) Config
            H = int(result.horizon)              # holding period in bars (e.g., 24 for 24h on 1h data)
            tc_per_side = 0.0                    # 5 bps per side (example); set 0.0 if you want no costs
            rf_annual = 0.0                      # annual risk-free (e.g., 0.03). If >0, excess will subtract per-period rf.

            # 1) Prepare price series and labels
            price_series = price_df["close"].copy()
            price_series.index = price_series.index.tz_localize(None)

            # If prediction_dates are hourly stamps, we can infer the bar duration later
            pred_dates = [pd.Timestamp(d).tz_localize(None) for d in result.prediction_dates]
            equity = [1.0]
            trade_returns = []
            trade_times = []

            # 2) Walk through signals without allowing overlap
            i = 0
            while i < len(pred_dates):
                date = pred_dates[i]
                if date not in price_series.index:
                    i += 1
                    continue

                try:
                    entry_idx = price_series.index.get_loc(date)
                    exit_idx  = entry_idx + H
                    if exit_idx >= len(price_series):
                        break  # cannot complete trade

                    entry = price_series.iloc[entry_idx]
                    exit_ = price_series.iloc[exit_idx]
                    raw_ret = (exit_ - entry) / entry

                    direction = 1 if result.y_pred[i] == 1 else -1
                    r_gross = direction * raw_ret

                    # round-trip transaction cost
                    r_net = r_gross - 2.0 * tc_per_side

                    # update equity on a per-trade basis
                    equity.append(equity[-1] * (1.0 + r_net))
                    trade_returns.append(r_net)
                    trade_times.append(date)

                    # jump forward by holding period -> no overlap
                    i += H
                except Exception as e:
                    print(f"[EQUITY] Failed at {date}: {e}")
                    i += 1
                    continue

            # 3) Annualised excess Sharpe from per-trade returns
            if len(trade_returns) > 1:
                # infer trade interval (median gap)
                if len(trade_times) > 1:
                    dt = np.median(np.diff(pd.to_datetime(trade_times)).astype('timedelta64[s]').astype(float))
                    secs_per_period = max(dt, 1.0)
                else:
                    secs_per_period = 3600.0  # fallback: one hour

                periods_per_year = (365.25 * 24 * 3600) / secs_per_period

                r = np.array(trade_returns, dtype=float)
                # convert annual rf to per-period rf
                rf_per_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0 if rf_annual != 0.0 else 0.0
                excess = r - rf_per_period
                sd = np.std(excess, ddof=1)
                sharpe = (excess.mean() / (sd + 1e-12)) * np.sqrt(periods_per_year)
            else:
                sharpe = 0.0

            # 4) Plot Equity Curve with dates (use prediction dates for x-axis)
            x_labels = trade_times if trade_times else result.prediction_dates
            pane = (self.backtest_lstm_pane if model_name.upper() == "LSTM"
                    else self.backtest_gru_pane if model_name.upper() == "GRU"
                    else self.backtest_stgnn_pane)
            self.frontend.root.after(0, lambda: self.plot_equity_curve(
                equity=equity, pane=pane, label=model_name, sharpe=sharpe, x_labels=x_labels
            ))

        # === STEP 4: Plot Losses (unchanged) ===
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


        # === STEP 5: Return summary ===
        # ------------------------------------
        return {
            "model": model_name,

            # tuned operating point (what your UI/confusion matrix uses)
            "accuracy": float(acc) if acc is not None else None,
            "f1": float(f1) if f1 is not None else None,                 # F1(pos)
            "macro_f1": float(macro_f1) if macro_f1 is not None else None,

            # baseline @ 0.5 (paper baseline)
            "accuracy_05": float(acc_05) if acc_05 is not None else None,
            "f1_05": float(f1_05) if f1_05 is not None else None,        # F1(pos) @0.5
            "macro_f1_05": float(macro_f1_05) if macro_f1_05 is not None else None,

            # Threshold metadata
            "best_threshold": float(best_thr),
            "threshold_rule": rule_name,

            # ranking metrics
            "roc_auc": roc_auc,
            "ap": ap,

            "sharpe": sharpe,
            "ticker": price_df.columns[0] if hasattr(price_df, "columns") else None,
            "n_predictions": len(result.y_pred),
            "horizon": result.horizon,
        }


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
            self.ax_train.plot(epochs[:len(hist_l)], hist_l, label='LSTM Train',
                            linewidth=1.5, color=self.colour_map["LSTM"])
        if hist_s:
            self.ax_train.plot(epochs[:len(hist_s)], hist_s, label='STGNN Train',
                            linewidth=1.5, color=self.colour_map["STGNN"])
        if hist_g:
            self.ax_train.plot(epochs[:len(hist_g)], hist_g, label='GRU Train',
                            linewidth=1.5, color=self.colour_map["GRU"])
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
        self._plot_validation_series(self.ax_val, self.val_loss_history_lstm, 'LSTM', '--', self.colour_map["LSTM"])
        self._plot_validation_series(self.ax_val, self.val_loss_history_gru,  'GRU',  '-.', self.colour_map["GRU"])
        self._plot_validation_series(self.ax_val, self.val_loss_history_stgnn,'STGNN',':',  self.colour_map["STGNN"])
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

    def plot_confusion_matrix(self, y_true, y_pred, model_name, thr=None, rule_name=None):
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

        title = f"{model_name} Confusion Matrix"
        if thr is not None:
            rule = rule_name or "thr"
            title += f"\n({rule}={thr:.3f})"
        ax_cm.set_title(title)

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
        
        return {
            "roc_auc": roc_auc,
            "ap": ap_score
        }

    def plot_recall_threshold(
            self,
            truths: List[int],
            probs: List[float],
            pane,
            model_name: str,
            best_thr: Optional[float] = None,
            rule_name: str = "default (0.5)",
        ) -> None:

        precision, recall, thresholds = precision_recall_curve(truths, probs, pos_label=1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.plot(thresholds, recall[:-1], label="Recall", linewidth=1.5)
        ax.plot(thresholds, precision[:-1], "--", label="Precision")
        ax.plot(thresholds, f1_scores[:-1], ":", label="F1 Score")

        # ---- Threshold marker (neutral color, no extra text box) ----
        if best_thr is not None:
            ax.axvline(
                best_thr,
                linestyle="--",
                linewidth=1.5,
                color="black",  # distinct from blue/orange/green
                label=f"{rule_name}: {best_thr:.3f}",
            )

        ax.set_title(f"{model_name} Scores vs Threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Legend: choose a stable corner (no overlapping text box anymore)
        ax.legend(loc="lower left", fontsize=8)

        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill="both", expand=True)
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
                linewidth=1.5,
                color=self.colour_map.get(label.upper(), None))

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
        if not loss_history:
            return
        dates = [entry['date'] for entry in loss_history]
        y = np.array([entry['loss'] for entry in loss_history])
        avg = y.mean()
        label = f'{label_prefix} Val (\u03BC={avg:.2f})'

        # main series: give it the colour so legend matches
        ax.plot(dates, y, line_style, label=label, linewidth=1.5, color=colour)

        # trend line in the same colour, thinner, and hidden from legend
        trend = np.polyfit(mdates.date2num(dates), y, 1)
        ax.plot(
            dates,
            np.polyval(trend, mdates.date2num(dates)),
            color=colour, linewidth=1, alpha=0.85, label='_nolegend_'
        )


    def _best_threshold_macro_f1(self, truths, probs) -> float:
        """
        Threshold in [0,1] that maximizes macro-F1.
        Macro-F1 = (F1_down + F1_up)/2, so it discourages collapsing to one class.
        """
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(truths, dtype=int)

        thr_grid = np.unique(probs)
        if thr_grid.size == 0:
            return 0.5
        thr_grid = np.concatenate(([0.0], thr_grid, [1.0]))

        best_thr, best_score = 0.5, -1.0
        for thr in thr_grid:
            pred = (probs >= thr).astype(int)
            score = f1_score(y, pred, average="macro")
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        return best_thr


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
