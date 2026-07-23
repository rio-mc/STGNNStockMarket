from types import SimpleNamespace

import numpy as np
import torch

from models.base_runner import BaseModelRunner
from models.runners.arima_runner import ARIMARunner
from models.runners.random_forest_runner import RandomForestRunner


def test_neural_capacity_counts_parameters_and_storage():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.Linear(2, 1),
    )

    capacity = BaseModelRunner._neural_capacity_metadata(model)

    assert capacity["family"] == "neural"
    assert capacity["primary_measure"] == "trainable_parameters"
    assert capacity["primary_value"] == 11
    assert capacity["neural_total_parameters"] == 11
    assert capacity["parameter_storage_bytes"] == 44


class _Tree:
    def __init__(self, nodes, leaves, depth):
        self.node_count = nodes
        self.n_leaves = leaves
        self.max_depth = depth
        self._nodes = np.zeros(nodes, dtype=np.int64)
        self._values = np.zeros((nodes, 2), dtype=np.float64)

    def __getstate__(self):
        return {"nodes": self._nodes, "values": self._values}


def test_random_forest_capacity_counts_tree_structure():
    model = SimpleNamespace(
        estimators_=[
            SimpleNamespace(tree_=_Tree(5, 3, 2)),
            SimpleNamespace(tree_=_Tree(7, 4, 3)),
        ],
        n_estimators=2,
        n_features_in_=40,
    )

    capacity = RandomForestRunner._random_forest_capacity_metadata(model)

    assert capacity["family"] == "random_forest"
    assert capacity["primary_measure"] == "tree_nodes"
    assert capacity["primary_value"] == 12
    assert capacity["rf_total_leaves"] == 7
    assert capacity["rf_max_depth_observed"] == 3
    assert capacity["rf_mean_depth"] == 2.5
    assert capacity["rf_n_features_in"] == 40
    assert capacity["parameter_storage_bytes"] > 0


def test_arima_capacity_counts_fitted_coefficients_and_state_dimension():
    fitted = SimpleNamespace(
        params=np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        model=SimpleNamespace(k_states=2),
        aic=101.5,
        bic=109.25,
    )

    capacity = ARIMARunner._arima_capacity_metadata(fitted, order=(1, 1, 1))

    assert capacity["family"] == "arima"
    assert capacity["primary_measure"] == "fitted_coefficients"
    assert capacity["primary_value"] == 3
    assert capacity["parameter_storage_bytes"] == 24
    assert capacity["arima_state_dimension"] == 2
    assert capacity["arima_aic"] == 101.5
    assert (capacity["arima_p"], capacity["arima_d"], capacity["arima_q"]) == (1, 1, 1)
