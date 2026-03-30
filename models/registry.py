from .lstm_runner import LSTMRunner
from .gru_runner import GRURunner
from .stgnn_runner import STGNNRunner


class ModelRegistry:
    _registry = {
        "lstm": LSTMRunner,
        "gru": GRURunner,
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