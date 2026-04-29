# STGNN Stock Market Prediction

Spatio-temporal graph neural network framework for stock trend prediction under controlled experimental conditions.

This project is designed to evaluate whether graph structure adds value beyond temporal modelling alone.

---

## Overview

This repository compares temporal and graph-based representations for financial time series forecasting.

Models included:

- **LSTM / GRU**: single-asset temporal baselines
- **PANEL_GRU / PANEL_LSTM**: multi-asset temporal baselines without graph structure
- **STGNN**: spatio-temporal graph neural network using asset relationships

The research objective is:

> Isolate and evaluate the contribution of graph structure under controlled experimental conditions.

This project should not be framed simply as “implementing multiple models”. The core contribution is the controlled comparison between temporal-only and graph-aware representations.

---

## Key Design Principles

The experimental design aims to ensure that:

- all models use the same input features
- all models use comparable tensor construction
- graph-based models use the same temporal information as non-graph baselines
- ablations isolate specific causal factors
- experiments can be run without the GUI
- results, predictions, graph diagnostics, and configurations are saved

The intended comparison is:

```text
same data + same features + same target
                    |
        representation changes only
                    |
 temporal baseline vs graph-aware model
```

---

## Project Architecture

```text
:
:
├── core/
│   ├── main.py
│   ├── pipeline.py
│   ├── experiment_runner.py
│   └── experiment_store.py
:
:
├── data/
│   ├── dataset_registry.py
│   ├── tensor_factory.py
│   └── ...
:
:
├── features/
│   └── feature_extractor.py
│
├── graph/
│   ├── graph_builder.py
│   └── log_graph_stats.py
:
:
├── models/
│   └── ...
├── scripts/
│   ├── run_experiment.py
│   └── run_sweep.py
├── results/
:
:
```

---

## Dataset Layer

Dataset loading is decoupled from the pipeline through a dataset registry.

Example usage:

```python
dataset = DatasetRegistry.load(args.dataset_name, args)
data_bundle = dataset.load()
```

Each dataset returns the same structure:

```python
{
    "data": Dict[str, DataFrame],
    "tickers": List[str],
    "index": DatetimeIndex,
    "metadata": Dict
}
```

This keeps dataset-specific logic out of the modelling pipeline and reduces the risk of data leakage.

---

## Pipeline

The pipeline is responsible for preparing the shared experimental state.

Example usage:

```python
pipeline = Pipeline(args)
state = pipeline.run()
```

Pipeline responsibilities:

- load or receive standardised dataset output
- construct train and validation splits
- perform feature engineering
- build tensors
- construct graph inputs
- return shared state for model execution

The pipeline does not decide which model to train. That responsibility belongs to the experiment runner.

---

## Experiment Runner

Model execution is handled separately from the pipeline.

Example usage:

```python
runner = ExperimentRunner()
result = runner.run(args.model, state, args)
```

The experiment runner:

- retrieves the correct model runner
- passes the shared pipeline state into the model
- executes training and evaluation
- returns a standardised result object

This separation makes headless experiments possible and removes GUI dependence from research runs.

---

## Model Abstraction

The goal is to make the models differ only in representation.

Temporal models learn from sequence information.

Graph models learn from the same temporal information plus graph structure.

The intended comparison is:

| Model Type | Temporal Features | Multi-Asset Context | Graph Structure |
|---|---:|---:|---:|
| LSTM / GRU | Yes | No | No |
| PANEL_GRU / PANEL_LSTM | Yes | Yes | No |
| STGNN | Yes | Yes | Yes |

---

## Graph Construction

Graphs are built from engineered asset-level features:

- return
- volatility
- momentum

Rather than using a single final timestep, graph features are aggregated over a rolling window. This makes graph construction more statistically stable.

Key graph arguments:

```bash
--k                # Number of KNN neighbours
--graph_mode       # knn | mst | knn_mst
--graph_window     # Rolling window for graph feature aggregation
--graph_embed      # pca | raw
```

Default behaviour:

- if `--graph_window` is not provided, it falls back to `--seq_len`
- graph features use rolling aggregation
- graph construction supports KNN, MST, and combined KNN + MST modes

Supported graph modes:

| Mode | Description |
|---|---|
| `knn` | Builds a K-nearest-neighbour similarity graph |
| `mst` | Builds a minimum spanning tree graph |
| `knn_mst` | Combines MST backbone with KNN edges |

---

## Graph Diagnostics

Graph statistics are saved for defensibility and reproducibility.

Logged statistics include:

- number of nodes
- number of edges
- graph density
- sector homophily

Saved to:

```text
results/graph_logging/graph_stats.json
```

These diagnostics help verify whether the graph is meaningful, too sparse, too dense, or sector-biased.

---

## Ablation Framework

The ablation framework is designed to isolate causal factors.

### Graph Ablation

```bash
--graph_ablation none | identity | empty
```

| Option | Meaning |
|---|---|
| `none` | Use the constructed graph normally |
| `identity` | Use self-connections only |
| `empty` | Remove graph edges |

### Feature Ablation

```bash
--ablate_feature return | volatility | momentum
```

Feature ablation removes one engineered feature from the graph and model inputs.

### Embedding Ablation

```bash
--graph_embed pca | raw
```

| Option | Meaning |
|---|---|
| `pca` | Use PCA-projected graph features |
| `raw` | Use standardised raw graph features |

---

## Headless Experiment Execution

Headless execution is the preferred mode for research runs.

### Single Experiment

```bash
python run_experiment.py \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 20 \
    --k 5 \
    --seed 42
```

### Baseline Example

```bash
python run_experiment.py \
    --model gru \
    --dataset_name sp500 \
    --seq_len 20 \
    --seed 42
```

### Graph Ablation Example

```bash
python run_experiment.py \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 20 \
    --k 5 \
    --graph_ablation identity \
    --seed 42
```

### Feature Ablation Example

```bash
python run_experiment.py \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 20 \
    --k 5 \
    --ablate_feature volatility \
    --seed 42
```

---

## Sweep Execution

Use `run_sweep.py` for controlled experiment grids.

Example:

```bash
python run_sweep.py \
    --models lstm gru panel_gru stgnn \
    --k_values 3 5 10 \
    --num_seeds 3
```

A sweep should vary one experimental factor at a time where possible.

Recommended sweep structure:

```text
1. Baselines
   - lstm
   - gru
   - panel_gru
   - panel_lstm

2. Graph model
   - stgnn

3. Graph density sweep
   - k = 3, 5, 10

4. Graph ablations
   - none
   - identity
   - empty

5. Feature ablations
   - return
   - volatility
   - momentum

6. Embedding ablations
   - pca
   - raw
```

---

## Logging and Results

Per-run results are saved to:

```text
results/results.csv
```

Each run should log:

- run ID
- model
- dataset
- ticker universe
- sequence length
- prediction horizon
- graph mode
- graph window
- k
- graph ablation
- feature ablation
- graph embedding mode
- seed
- metrics
- runtime
- status

Raw predictions are saved to:

```text
results/predictions/{run_id}.csv
```

Prediction files should include:

- `y_true`
- `y_pred`
- class probabilities or confidence scores
- timestamp or sample index where applicable

Graph diagnostics are saved to:

```text
results/graph_logging/
```

---

## Setup

### Local Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r instructions/requirements.txt
```

### GPU Setup

For CUDA 12.1 PyTorch wheels:

```bash
pip install -r instructions/requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Running the GUI

The GUI remains available for interactive exploration.

```bash
python -m core.main
```

The GUI is not required for reproducible experiments.

---

## Running Headless

Run a single experiment:

```bash
python run_experiment.py --model lstm
```

Run a graph model:

```bash
python run_experiment.py \
    --model stgnn \
    --k 5 \
    --graph_mode knn_mst
```

Run with a specific seed:

```bash
python run_experiment.py \
    --model stgnn \
    --seed 123
```

---

## Common CLI Arguments

| Argument | Description |
|---|---|
| `--model` | Model to run |
| `--dataset_name` | Dataset or asset universe |
| `--seq_len` | Temporal lookback window |
| `--prediction_window` | Prediction horizon |
| `--k` | Number of graph neighbours |
| `--graph_mode` | Graph construction strategy |
| `--graph_window` | Rolling window for graph feature aggregation |
| `--graph_ablation` | Graph ablation mode |
| `--ablate_feature` | Feature ablation mode |
| `--graph_embed` | Graph embedding mode |
| `--seed` | Random seed |
| `--num_seeds` | Number of seeds for sweeps |
| `--results_dir` | Directory for saved outputs |

---

## Reproducibility

For reproducible runs:

```bash
python run_experiment.py \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 20 \
    --k 5 \
    --seed 42 \
    --deterministic
```

The system is designed to log all run-level configuration required to reproduce results.

---

## Recommended Experiment Order

For a clean research workflow:

1. Run a tiny smoke test.
2. Run temporal baselines.
3. Run panel temporal baselines.
4. Run the STGNN model.
5. Run graph ablations.
6. Run feature ablations.
7. Run graph embedding ablations.
8. Run multi-seed sweeps.
9. Analyse saved results and graph diagnostics.

Smoke test:

```bash
python run_experiment.py \
    --model lstm \
    --dataset_name sp500 \
    --seq_len 10 \
    --seed 42
```

Graph smoke test:

```bash
python run_experiment.py \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 10 \
    --k 3 \
    --seed 42
```

---

## Research Framing

Use this framing:

> We isolate and evaluate the contribution of graph structure under controlled experimental conditions.

Avoid this framing:

> We implemented multiple models.

The distinction matters because the contribution is not the existence of several architectures. The contribution is the controlled experimental comparison of temporal-only and graph-aware representations.

---

## Experimental Checklist

Before reporting results, confirm that:

- [ ] all models use the same input features
- [ ] all models use the same target definition
- [ ] train and validation splits are consistent
- [ ] graph construction uses only training data
- [ ] `graph_window` is logged
- [ ] `seq_len` is logged
- [ ] `k` is logged
- [ ] graph mode is logged
- [ ] graph ablations are logged
- [ ] feature ablations are logged
- [ ] embedding ablations are logged
- [ ] random seed is logged
- [ ] raw predictions are saved
- [ ] metrics are saved
- [ ] graph diagnostics are saved
- [ ] experiments can run without the GUI

---

## Notes on Deprecated Test Suite Documentation

Older documentation that describes GUI-driven experiment execution through `main.py` and `run_experiments()` is deprecated.

Experiments should now be run through:

```bash
python run_experiment.py
```

or:

```bash
python run_sweep.py
```

The GUI should be treated as an interactive interface, not as the primary research execution path.
