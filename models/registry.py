from .runners.gcn_runner import GCNRunner
from .runners.gru_runner import GRURunner
from .runners.graphsage_runner import GraphSAGERunner
from .runners.lstm_runner import LSTMRunner
from .runners.nnconv_runner import NNConvRunner
from .runners.panel_gru_runner import PanelGRURunner
from .runners.panel_lstm_runner import PanelLSTMRunner
from .runners.stgnn_runner import STGNNRunner


class ModelRegistry:
    _registry = {
        "lstm": LSTMRunner,
        "gru": GRURunner,
        "panel_gru": PanelGRURunner,
        "panel_lstm": PanelLSTMRunner,
        "gcn": GCNRunner,
        "nnconv": NNConvRunner,
        "graphsage": GraphSAGERunner,
        "stgnn": STGNNRunner,
    }

    @classmethod
    def get_runner(cls, model_name: str):
        key = model_name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {', '.join(sorted(cls._registry.keys()))}"
            )
        return cls._registry[key]()

    @classmethod
    def available_models(cls):
        return sorted(cls._registry.keys())
