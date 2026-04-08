from typing import List, Optional

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from backtesting.engine import BacktestEngine
from backtesting.plotters import BacktestPlotter
from evaluation.evaluation_types import EvaluationResult


class EvaluationMethods:
    """
    Single-model evaluation and rendering.

    Responsibilities:
    - compute dense predictive metrics on the full rolling validation set
    - compute trade-aligned predictive metrics on the exact non-overlapping
      subset executed by the backtest
    - render confusion matrix, ROC/PR, threshold curve, and loss plots
    - delegate backtesting to BacktestEngine / BacktestPlotter
    """

    def __init__(self, frontend):
        self.frontend = frontend

        self.eval_pane = frontend.eval_pane
        self.loss_pane = frontend.loss_pane
        self.backtest_pane = frontend.backtest_pane
        self.roc_pane = frontend.roc_pane
        self.threshold_pane = frontend.threshold_pane

        self.loss_pane.rowconfigure(0, weight=1)
        self.loss_pane.rowconfigure(1, weight=1)
        self.loss_pane.columnconfigure(0, weight=1)

        self.train_frame = tk.Frame(self.loss_pane)
        self.train_frame.grid(row=0, column=0, sticky="nsew")

        self.val_frame = tk.Frame(self.loss_pane)
        self.val_frame.grid(row=1, column=0, sticky="nsew")

        self.loss_train_fig = Figure(figsize=(10, 3))
        self.ax_train = self.loss_train_fig.add_subplot(111)
        self.loss_train_canvas = FigureCanvasTkAgg(self.loss_train_fig, master=self.train_frame)
        self.loss_train_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.loss_val_fig = Figure(figsize=(10, 3))
        self.ax_val = self.loss_val_fig.add_subplot(111)
        self.loss_val_canvas = FigureCanvasTkAgg(self.loss_val_fig, master=self.val_frame)
        self.loss_val_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.current_model_name: Optional[str] = None
        self.loss_history_train: List[float] = []
        self.loss_history_val: List[dict] = []

        self.colour_map = {
            "LSTM": "tab:blue",
            "GRU": "tab:green",
            "STGNN": "tab:orange",
        }

        self.backtest_engine = BacktestEngine(tc_per_side=0.0, rf_annual=0.0)
        self.backtest_plotter = BacktestPlotter(
            pane=self.backtest_pane,
            clear_callback=self._clear_pane,
        )

    def evaluate(
        self,
        model_name: str,
        result: EvaluationResult,
        price_df: pd.DataFrame,
    ):
        if not result.y_true or not result.probs:
            print(f"[{model_name} Evaluation] No predictions to evaluate.")
            return

        self.current_model_name = str(model_name).upper()
        self.loss_history_train = list(result.hist_train or [])
        self.loss_history_val = list(result.hist_val or [])

        y_true = np.asarray(result.y_true, dtype=int)
        probs = np.asarray(result.probs, dtype=float)

        dense_thr = float(getattr(result, "decision_threshold", 0.5) or 0.5)
        dense_pred = (probs >= dense_thr).astype(int)
        result.y_pred = dense_pred.astype(int).tolist()

        best_thr_macro_f1 = self._best_threshold_macro_f1(y_true, probs)
        tuned_pred = (probs >= best_thr_macro_f1).astype(int)

        dense_metrics = self._classification_metrics(y_true=y_true, y_pred=dense_pred, probs=probs)
        tuned_dense_metrics = self._classification_metrics(y_true=y_true, y_pred=tuned_pred, probs=probs)

        print(
            f"[{model_name} Evaluation][Dense] "
            f"thr={dense_thr:.3f} (fixed) | "
            f"Acc={dense_metrics['accuracy']:.3f} | "
            f"F1(pos)={dense_metrics['f1']:.3f} | "
            f"F1(macro)={dense_metrics['macro_f1']:.3f} | "
            f"ROC-AUC={dense_metrics['roc_auc']:.3f if dense_metrics['roc_auc'] is not None else float('nan'):.3f} | "
            f"AP={dense_metrics['ap']:.3f if dense_metrics['ap'] is not None else float('nan'):.3f}"
        )
        print(
            f"[{model_name} Evaluation][Dense Supplementary] "
            f"best macro-F1 threshold={best_thr_macro_f1:.3f} | "
            f"Acc={tuned_dense_metrics['accuracy']:.3f} | "
            f"F1(pos)={tuned_dense_metrics['f1']:.3f} | "
            f"F1(macro)={tuned_dense_metrics['macro_f1']:.3f}"
        )

        self.frontend.root.after(
            0,
            lambda: self.plot_confusion_matrix(
                y_true,
                dense_pred,
                model_name,
                thr=dense_thr,
                rule_name="Fixed threshold",
            ),
        )
        self.frontend.root.after(0, lambda: self.plot_roc_and_pr(y_true, probs, model_name))
        self.frontend.root.after(
            0,
            lambda: self.plot_recall_threshold(
                truths=y_true,
                probs=probs,
                model_name=model_name,
                selected_thr=dense_thr,
                selected_label="Fixed threshold",
                best_thr=best_thr_macro_f1,
            ),
        )

        backtest_result = self.backtest_engine.run(
            model_name=model_name,
            evaluation_result=result,
            price_df=price_df,
            price_column="close" if "close" in price_df.columns else price_df.columns[0],
            threshold=dense_thr,
        )

        trade_metrics = self._empty_classification_metrics()
        strategy_metrics = {
            "sharpe": None,
            "n_trades": 0,
            "mean_trade_return": None,
            "hit_rate": None,
            "final_equity": None,
            "max_drawdown": None,
        }

        if backtest_result is not None:
            exec_idx = list(backtest_result.executed_indices)
            result.trade_aligned_indices = exec_idx
            result.trade_aligned_threshold = dense_thr
            result.trade_aligned_y_true = y_true[exec_idx].astype(int).tolist() if exec_idx else []
            result.trade_aligned_probs = probs[exec_idx].astype(float).tolist() if exec_idx else []
            result.trade_aligned_y_pred = ((probs[exec_idx] >= dense_thr).astype(int).tolist() if exec_idx else [])
            result.trade_aligned_prediction_dates = [result.prediction_dates[i] for i in exec_idx]

            if exec_idx:
                trade_metrics = self._classification_metrics(
                    y_true=np.asarray(result.trade_aligned_y_true, dtype=int),
                    y_pred=np.asarray(result.trade_aligned_y_pred, dtype=int),
                    probs=np.asarray(result.trade_aligned_probs, dtype=float),
                )

            strategy_metrics = {
                "sharpe": float(backtest_result.sharpe) if backtest_result.sharpe is not None else None,
                "n_trades": int(backtest_result.n_trades),
                "mean_trade_return": float(backtest_result.mean_trade_return),
                "hit_rate": float(backtest_result.hit_rate),
                "final_equity": float(backtest_result.final_equity) if backtest_result.final_equity is not None else None,
                "max_drawdown": float(backtest_result.max_drawdown),
            }

            print(
                f"[{model_name} Evaluation][Trade-aligned] "
                f"thr={dense_thr:.3f} (fixed) | "
                f"N={len(exec_idx)} | "
                f"Acc={trade_metrics['accuracy']:.3f} | "
                f"F1(pos)={trade_metrics['f1']:.3f} | "
                f"F1(macro)={trade_metrics['macro_f1']:.3f} | "
                f"Sharpe={strategy_metrics['sharpe']:.3f if strategy_metrics['sharpe'] is not None else float('nan'):.3f} | "
                f"HitRate={strategy_metrics['hit_rate']:.3f if strategy_metrics['hit_rate'] is not None else float('nan'):.3f} | "
                f"MeanTradeRet={strategy_metrics['mean_trade_return']:.6f if strategy_metrics['mean_trade_return'] is not None else float('nan'):.6f}"
            )

            self.frontend.root.after(
                0,
                lambda: self.backtest_plotter.plot_equity_curve(backtest_result)
            )

        self.frontend.root.after(0, self.plot_loss)

        return {
            "model": model_name,
            "threshold_fixed": dense_thr,
            "threshold_macro_f1_dense": float(best_thr_macro_f1),
            "val_loss_dense": float(result.dense_val_loss) if result.dense_val_loss is not None else None,
            "accuracy_dense": float(dense_metrics["accuracy"]),
            "f1_dense": float(dense_metrics["f1"]),
            "macro_f1_dense": float(dense_metrics["macro_f1"]),
            "roc_auc_dense": dense_metrics["roc_auc"],
            "ap_dense": dense_metrics["ap"],
            "accuracy_dense_macro_f1_threshold": float(tuned_dense_metrics["accuracy"]),
            "f1_dense_macro_f1_threshold": float(tuned_dense_metrics["f1"]),
            "macro_f1_dense_macro_f1_threshold": float(tuned_dense_metrics["macro_f1"]),
            "accuracy_trade_aligned": float(trade_metrics["accuracy"]),
            "f1_trade_aligned": float(trade_metrics["f1"]),
            "macro_f1_trade_aligned": float(trade_metrics["macro_f1"]),
            "roc_auc_trade_aligned": trade_metrics["roc_auc"],
            "ap_trade_aligned": trade_metrics["ap"],
            "sharpe": strategy_metrics["sharpe"],
            "n_trades": strategy_metrics["n_trades"],
            "mean_trade_return": strategy_metrics["mean_trade_return"],
            "hit_rate": strategy_metrics["hit_rate"],
            "final_equity": strategy_metrics["final_equity"],
            "max_drawdown": strategy_metrics["max_drawdown"],
            "ticker": price_df.columns[0] if hasattr(price_df, "columns") else None,
            "n_predictions_dense": len(result.probs),
            "n_predictions_trade_aligned": len(result.trade_aligned_indices),
            "horizon": result.horizon,
        }

    def plot_loss(
        self,
        hist_train: Optional[List[float]] = None,
        hist_val: Optional[List[dict]] = None,
    ):
        hist_train = hist_train if hist_train is not None else self.loss_history_train
        hist_val = hist_val if hist_val is not None else self.loss_history_val

        self.ax_train.clear()
        self.ax_val.clear()

        epochs = list(range(1, len(hist_train) + 1))
        colour = self.colour_map.get(self.current_model_name or "", "tab:blue")
        label_name = self.current_model_name or "MODEL"

        if hist_train:
            self.ax_train.plot(
                epochs,
                hist_train,
                label=f"{label_name} Train",
                linewidth=1.5,
                color=colour,
            )

        max_ticks = 10
        if len(epochs) > max_ticks:
            step = max(1, len(epochs) // max_ticks)
            ticks_to_show = epochs[::step]
        else:
            ticks_to_show = epochs

        if ticks_to_show:
            self.ax_train.set_xticks(ticks_to_show)
            self.ax_train.set_xticklabels([str(e) for e in ticks_to_show], rotation=0, ha="center")
        self.ax_train.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val)}"))
        self.ax_train.set_xlabel("Epoch")
        self.ax_train.set_title("Training Loss")
        self.ax_train.set_ylabel("Loss")
        if hist_train:
            self.ax_train.legend(loc="upper right", frameon=False)
        self.ax_train.grid(True, linestyle="--", alpha=0.5)

        self._plot_validation_series(
            self.ax_val,
            hist_val,
            label_name,
            "--",
            colour,
        )

        self.ax_val.set_title("Validation Loss")
        self.ax_val.set_xlabel("Date")
        self.ax_val.set_ylabel("Loss")
        if self.ax_val.get_legend_handles_labels()[0]:
            self.ax_val.legend(loc="upper right", frameon=False)
        self.ax_val.grid(True, linestyle="--", alpha=0.5)
        self.ax_val.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_val.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.loss_val_fig.autofmt_xdate(rotation=30)

        self.loss_train_fig.tight_layout(pad=1.0)
        self.loss_train_canvas.draw()
        self.loss_val_fig.tight_layout(pad=1.0)
        self.loss_val_canvas.draw()

    def plot_confusion_matrix(self, y_true, y_pred, model_name, thr=None, rule_name=None):
        labels = ["Down", "Up"]
        truths = [labels[y] for y in y_true]
        preds = [labels[p] for p in y_pred]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(truths, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(
            truths, preds, labels=labels, zero_division=0
        )

        pane = self.eval_pane

        fig, (ax_cm, ax_txt) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(5, 5),
            gridspec_kw={"height_ratios": [3, 1]},
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greys",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax_cm,
            square=True,
            linewidths=0.5,
            linecolor="lightgrey",
        )
        sns.despine(fig=fig, ax=ax_cm, left=False, bottom=False)
        ax_cm.set_xlabel("Predicted", fontsize=10)
        ax_cm.set_ylabel("Actual", fontsize=10)

        title = f"{model_name} Confusion Matrix"
        if thr is not None:
            rule = rule_name or "thr"
            title += f"\n({rule}={thr:.3f})"
        ax_cm.set_title(title)

        ax_cm.tick_params(axis="both", which="major", labelsize=9)
        ax_txt.axis("off")

        rows = ["Overall"] + labels
        cols = ["Accuracy", "Precision", "Recall", "F1"]
        cell_text = [[f"{acc:.2f}", "", "", ""]] + [
            ["", f"{pr[i]:.2f}", f"{rc[i]:.2f}", f"{f1[i]:.2f}"] for i in range(len(labels))
        ]
        tbl = ax_txt.table(
            cellText=cell_text,
            rowLabels=rows,
            colLabels=cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)

        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        plt.close(fig)

    def plot_roc_and_pr(
        self,
        truths: List[int],
        probs: Optional[List[float]] = None,
        model_name: str = "Model",
    ) -> None:
        pane = self.roc_pane
        if pane is None:
            print(f"[WARN] No pane found for {model_name} ROC/PR plot.")
            return

        if probs is None:
            self._clear_pane(pane)
            return

        truths = np.asarray(truths, dtype=int)
        probs = np.asarray(probs, dtype=float)

        fpr, tpr, _ = roc_curve(truths, probs, pos_label=1)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(truths, probs, pos_label=1)
        ap = average_precision_score(truths, probs, pos_label=1)

        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 3))

        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=1.5)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        ax_roc.set_title(f"{model_name.upper()} ROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(frameon=False)
        ax_roc.grid(True, linestyle="--", alpha=0.5)

        ax_pr.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=1.5)
        ax_pr.set_title(f"{model_name.upper()} Precision-Recall Curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(frameon=False)
        ax_pr.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        plt.close(fig)

    def plot_recall_threshold(
        self,
        truths,
        probs,
        model_name,
        selected_thr,
        selected_label=None,
        best_thr=None,
    ):
        pane = self.threshold_pane

        thresholds = np.linspace(0.0, 1.0, 201)
        recalls = []
        precisions = []
        f1s = []

        truths = np.asarray(truths, dtype=int)
        probs = np.asarray(probs, dtype=float)

        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(
                truths,
                preds,
                average="binary",
                zero_division=0,
            )
            recalls.append(rc)
            precisions.append(pr)
            f1s.append(f1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thresholds, recalls, label="Recall", linewidth=1.5)
        ax.plot(thresholds, precisions, label="Precision", linewidth=1.5)
        ax.plot(thresholds, f1s, label="F1", linewidth=1.5)

        ax.axvline(selected_thr, linestyle="--", linewidth=1.2)
        if best_thr is not None:
            ax.axvline(best_thr, linestyle=":", linewidth=1.2)
        label = selected_label or "Selected threshold"
        title = f"{model_name.upper()} Threshold Curves\n{label} = {selected_thr:.3f}"
        if best_thr is not None:
            title += f" | Best macro-F1 = {best_thr:.3f}"
        ax.set_title(title)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        self._clear_pane(pane)
        canvas = FigureCanvasTkAgg(fig, master=pane)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        plt.close(fig)

    def _plot_validation_series(self, ax, loss_history, label_prefix, line_style, colour):
        if not loss_history:
            return

        dates = [entry["date"] for entry in loss_history]
        y = np.array([entry["loss"] for entry in loss_history])
        avg = y.mean()
        label = f"{label_prefix} Val (μ={avg:.2f})"

        ax.plot(dates, y, line_style, label=label, linewidth=1.5, color=colour)

        if len(dates) >= 2:
            trend = np.polyfit(mdates.date2num(dates), y, 1)
            ax.plot(
                dates,
                np.polyval(trend, mdates.date2num(dates)),
                color=colour,
                linewidth=1,
                alpha=0.85,
                label="_nolegend_",
            )

    def _best_threshold_macro_f1(self, truths, probs) -> float:
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

    def _classification_metrics(self, y_true, y_pred, probs=None):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if y_true.size == 0:
            return self._empty_classification_metrics()

        accuracy = accuracy_score(y_true, y_pred)
        f1 = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[2]
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        roc_auc = None
        ap = None
        if probs is not None:
            probs = np.asarray(probs, dtype=float)
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, probs, pos_label=1)
                roc_auc = float(auc(fpr, tpr))
                ap = float(average_precision_score(y_true, probs, pos_label=1))

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "macro_f1": float(macro_f1),
            "roc_auc": roc_auc,
            "ap": ap,
        }

    def _empty_classification_metrics(self):
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "roc_auc": None,
            "ap": None,
        }

    def reset_histories(self):
        self.current_model_name = None
        self.loss_history_train.clear()
        self.loss_history_val.clear()

    def _clear_pane(self, pane):
        if pane is None:
            return
        for widget in pane.winfo_children():
            widget.destroy()
