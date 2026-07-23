from __future__ import annotations

from importlib import import_module


class ModelRegistry:
    _registry = {
        "lstm": ("models.runners.lstm_runner", "LSTMRunner"),
        "gru": ("models.runners.gru_runner", "GRURunner"),
        "arima": ("models.runners.arima_runner", "ARIMARunner"),
        "random_forest": ("models.runners.random_forest_runner", "RandomForestRunner"),
        "rf": ("models.runners.random_forest_runner", "RandomForestRunner"),
        "panel_gru": ("models.runners.panel_gru_runner", "PanelGRURunner"),
        "panel_lstm": ("models.runners.panel_lstm_runner", "PanelLSTMRunner"),
        "gcn": ("models.runners.gcn_runner", "GCNRunner"),
        "nnconv": ("models.runners.nnconv_runner", "NNConvRunner"),
        "graphsage": ("models.runners.graphsage_runner", "GraphSAGERunner"),
        "stgnn": ("models.runners.stgnn_runner", "STGNNRunner"),
        "gat": ("models.runners.gat_runner", "GATRunner"),
    }

    @classmethod
    def get_runner(cls, model_name: str):
        key = model_name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {', '.join(sorted(cls._registry.keys()))}"
            )

        module_name, class_name = cls._registry[key]
        try:
            module = import_module(module_name)
        except ImportError as exc:
            if key in {"gcn", "gat", "graphsage", "nnconv", "stgnn"}:
                raise RuntimeError(
                    f"Model '{model_name}' requires torch_geometric/PyG dependencies. "
                    "Install the graph dependencies documented in INSTALL.md."
                ) from exc
            raise

        return getattr(module, class_name)()

    @classmethod
    def available_models(cls):
        return sorted(cls._registry.keys())
