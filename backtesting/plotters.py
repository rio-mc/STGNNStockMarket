import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from backtesting.types import BacktestResult


class BacktestPlotter:
    """
    Tkinter plotter for single-model backtest results.
    """

    def __init__(self, pane, clear_callback):
        self.pane = pane
        self.clear_callback = clear_callback

    def plot_equity_curve(self, result: BacktestResult):
        fig, ax = plt.subplots(figsize=(8, 4))

        x_labels = result.trade_times
        equity = result.equity

        if x_labels and len(x_labels) == len(equity) - 1:
            x = [x_labels[0]] + list(x_labels)
        else:
            x = list(range(len(equity)))

        ax.plot(x, equity, linewidth=1.5, label=result.model_name)
        ax.set_title(
            f"{result.model_name} Equity Curve"
            + (f" | Sharpe: {result.sharpe:.3f}" if result.sharpe is not None else "")
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.5)

        if x_labels:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate(rotation=30)

        fig.tight_layout()
        self.clear_callback(self.pane)
        canvas = FigureCanvasTkAgg(fig, master=self.pane)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        plt.close(fig)