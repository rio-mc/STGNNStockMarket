from ui.tabs.evaluation_tab import EvaluationTab
from ui.tabs.backtesting_tab import BacktestingTab
from ui.tabs.metrics_tab import MetricsTab


class SingleModelTabBundle:
    """
    Convenience wrapper that builds and exposes the single-model tabs
    required by the refactored UI.
    """

    def __init__(self, notebook):
        self.evaluation = EvaluationTab(notebook)
        self.backtesting = BacktestingTab(notebook)
        self.metrics = MetricsTab(notebook)

    def set_model_titles(self, model_name: str):
        self.evaluation.set_model_title(model_name)
        self.backtesting.set_model_title(model_name)
        self.metrics.set_model_title(model_name)