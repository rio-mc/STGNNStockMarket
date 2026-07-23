import json
import shutil

from core.experiment_store import ExperimentStore


def test_model_groups_have_dedicated_research_families():
    assert ExperimentStore.model_group("lstm") == "recurrent"
    assert ExperimentStore.model_group("panel_gru") == "panel"
    assert ExperimentStore.model_group("arima") == "classical"
    assert ExperimentStore.model_group("random_forest") == "classical"
    assert ExperimentStore.model_group("gcn") == "graph"
    assert ExperimentStore.model_group("stgnn", "nnconv") == "stgnn"


def test_save_history_recreates_directory_removed_after_store_initialisation(
    workspace_tmp,
):
    store = ExperimentStore(workspace_tmp / "results")
    shutil.rmtree(store.histories_dir)

    history_path = store.save_history(
        run_id="RUN-recovery",
        hist_train=[0.7, 0.6],
        hist_val=[0.8, 0.7],
    )

    assert history_path is not None
    saved = json.loads(
        store.histories_dir.joinpath("RUN-recovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["hist_train"] == [0.7, 0.6]
    assert saved["hist_val"] == [0.8, 0.7]


def test_flat_result_row_exposes_stable_analysis_columns():
    store = object.__new__(ExperimentStore)
    payload = {
        "run_id": "RUN-1",
        "status": "success",
        "timestamp_start": "2026-07-22T15:00:00+00:00",
        "timestamp_end": "2026-07-22T15:00:05+00:00",
        "duration_sec": 5.0,
        "ticker": "AAPL",
        "prediction_window": "1d",
        "model": "stgnn",
        "seed": 42,
        "graph_model": "nnconv",
        "universe_id": "sp500",
        "interval": "1h",
        "k": 3,
        "graph_mode": "knn_mst",
        "graph_embed": "pca",
        "graph_ablation": "none",
        "ablate_feature": "none",
        "threshold_policy": "macro_f1_dense",
        "deterministic": True,
        "direction": "Upwards",
        "confidence": 50.11,
        "metrics": {
            "horizon": 24,
            "threshold_operational": 0.500,
            "threshold_macro_f1_dense": 0.500,
            "accuracy_dense": 0.549,
            "f1_dense": 0.703,
            "macro_f1_dense": 0.385,
            "roc_auc_dense": 0.440,
            "ap_dense": 0.568,
            "val_loss_dense": 0.626,
            "test_loss_dense": 0.628,
            "macro_f1_dense_fixed_05": 0.379,
            "macro_f1_trade_aligned": 0.344,
            "sharpe": 1.230,
            "final_equity": 1.197,
        },
        "extras": {
            "graph_backend": "nnconv",
            "compute": {
                "train_seconds": 4.40,
                "energy_wh": 0.0618,
                "gpu_peak_memory_mb": 283.36,
                "training_sample_unit": "supervised_window",
                "train_examples_unique": 2966,
                "sample_exposures": 47456,
                "epochs_completed": 16,
            },
            "capacity": {
                "family": "neural",
                "primary_measure": "trainable_parameters",
                "primary_value": 32661,
                "parameter_storage_bytes": 130644,
                "neural_total_parameters": 32661,
                "neural_trainable_parameters": 32661,
            },
            "graph_stats": {
                "requested_k": 3,
                "effective_k": 3,
                "num_nodes": 50,
                "num_edges": 94,
                "num_mst_edges": 49,
                "density": 0.076735,
                "mean_degree": 3.76,
                "connected_components": 1,
                "isolated_nodes": 0,
                "sector_edge_homophily": 0.266,
                "sector_homophilous_edges": 25,
                "sector_homophily_eligible_edges": 94,
                "sector_known_nodes": 50,
                "sector_unknown_nodes": 0,
            },
        },
    }

    row = store._flatten_result_row(payload)

    assert row["model_family"] == "stgnn"
    assert row["model_label"] == "stgnn+nnconv"
    assert row["threshold_operational"] == 0.500
    assert row["deterministic"] is True
    assert row["macro_f1_dense"] == 0.385
    assert row["macro_f1_fixed_05"] == 0.379
    assert row["macro_f1_trade_aligned"] == 0.344
    assert row["train_seconds"] == 4.40
    assert row["peak_gpu_memory_mb"] == 283.36
    assert row["train_examples_unique"] == 2966
    assert row["sample_exposures"] == 47456
    assert row["epochs_completed"] == 16
    assert row["capacity_primary_measure"] == "trainable_parameters"
    assert row["capacity_primary_value"] == 32661
    assert row["parameter_storage_bytes"] == 130644
    assert row["graph_stats_applicable"] is True
    assert row["graph_effective_k"] == 3
    assert row["graph_num_edges"] == 94
    assert row["graph_connected_components"] == 1
    assert row["sector_edge_homophily"] == 0.266
    assert row["sector_homophily_eligible_edges"] == 94


def test_flat_result_row_backfills_legacy_neural_capacity_and_sample_counts():
    store = object.__new__(ExperimentStore)
    payload = {
        "model": "lstm",
        "metrics": {},
        "extras": {
            "compute": {
                "train_samples": 47456,
                "total_params": 22818,
                "trainable_params": 22818,
            },
            "metadata": {
                "lr_history": [{} for _ in range(16)],
            },
        },
    }

    row = store._flatten_result_row(payload)

    assert row["training_sample_unit"] == "supervised_window"
    assert row["train_examples_unique"] == 2966
    assert row["sample_exposures"] == 47456
    assert row["epochs_completed"] == 16
    assert row["capacity_family"] == "neural"
    assert row["capacity_primary_measure"] == "trainable_parameters"
    assert row["capacity_primary_value"] == 22818
    assert row["parameter_storage_bytes"] == 91272
    assert row["graph_stats_applicable"] is False
    assert row["sector_edge_homophily"] is None


def test_flat_result_row_recovers_legacy_graph_stats_file(workspace_tmp):
    graph_stats_path = workspace_tmp / "graph_stats.json"
    graph_stats_path.write_text(
        json.dumps(
            {
                "num_nodes": 50,
                "num_edges": 94,
                "density": 0.076735,
                "homophily": 0.266,
                "effective_k": 3,
            }
        ),
        encoding="utf-8",
    )
    store = object.__new__(ExperimentStore)
    payload = {
        "model": "gcn",
        "k": 3,
        "metrics": {},
        "extras": {"graph_stats_path": str(graph_stats_path)},
    }

    row = store._flatten_result_row(payload)

    assert row["graph_stats_applicable"] is True
    assert row["graph_num_edges"] == 94
    assert row["sector_edge_homophily"] == 0.266


def test_flat_result_row_writes_undefined_homophily_as_missing():
    store = object.__new__(ExperimentStore)
    row = store._flatten_result_row(
        {
            "model": "gcn",
            "metrics": {},
            "extras": {
                "graph_stats": {
                    "sector_edge_homophily": float("nan"),
                    "sector_homophily_eligible_edges": 0,
                }
            },
        }
    )

    assert row["sector_edge_homophily"] is None
    assert row["sector_homophily_eligible_edges"] == 0
