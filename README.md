# **STGNN Stock Market Prediction**

*A controlled experimental framework for evaluating temporal vs graph-based representations in financial time series.*

---

## **1. Core Objective**

This project is not centred on model implementation.

Its purpose is to:

> **Isolate and evaluate the contribution of graph structure under controlled experimental conditions.**

All experiments enforce:

```text
Same data + Same features + Same target
                    |
        Representation varies only
                    |
 Temporal baseline vs Graph-aware model
```

---

## **2. Model Library**

All models are accessed via a unified interface:

```python
runner = ModelRegistry.get_runner(model_name)
```

This guarantees consistency across architectures.

### **2.1 Model Categories**

|                                 Category |            Models            | Description                                             |
|------------------------------------------|------------------------------|---------------------------------------------------------|
|    **Temporal Baselines (Single Asset)** |         `lstm`, `gru`        | Standard sequence models                                |
|  **Panel Temporal Models (Multi-Asset)** |   `panel_lstm`, `panel_gru`  | Cross-sectional temporal modelling (no graph structure) |
|                **Graph Neural Networks** | `gcn`, `graphsage`, `nnconv` | Static relational modelling                             |
|                  **Spatio-Temporal GNN** |           `stgnn`            | Joint temporal + relational modelling                   |

---

## **3. System Architecture**

```text
Dataset → Pipeline → ExperimentRunner → Model → Evaluation → Backtest → Results
```

---

## **4. Configuration Space**

The configuration space is the primary research interface.

### **4.1 Temporal Parameters**

|             Parameter | Description           |
|-----------------------|-----------------------|
|           `--seq_len` | Input sequence length |
| `--prediction_window` | Forecast horizon      |

---

### **4.2 Graph Construction**

|       Parameter |         Options         | Description          |
|-----------------|-------------------------|----------------------|
|           `--k` |         Integer         | Number of neighbours |
|  `--graph_mode` | `knn`, `mst`, `knn_mst` | Graph topology       |
| `--graph_embed` |       `pca`, `raw`      | Feature embedding    |

Graph construction is based on rolling statistics of:

- returns  
- volatility  
- momentum  

---

## **5. Ablation Framework**

### **5.1 Graph Ablation**

|       Mode | Effect                   |
|------------|--------------------------|
|     `none` | Full graph               |
| `identity` | Removes cross-node edges |
|    `empty` | Removes graph entirely   |

---

### **5.2 Feature Ablation**

```bash
--ablate_feature return | volatility | momentum
```

---

### **5.3 Embedding Ablation**

```bash
--graph_embed pca | raw
```

---

### **5.4 Reproducibility**

|         Parameter | Description             |
|-------------------|-------------------------|
|          `--seed` | Random seed             |
|     `--num_seeds` | Number of runs          |
| `--deterministic` | Deterministic execution |

---

## **6. Running Experiments**

### **6.1 Single Run**

```bash
python -m scripts.run_experiment \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 10 \
    --k 3 \
    --seed 42
```

---

### **6.2 Baseline Comparison**

```bash
python -m scripts.run_experiment --model lstm
python -m scripts.run_experiment --model panel_gru
python -m scripts.run_experiment --model stgnn --k 3
```

---

### **6.3 Full Sweep**

```bash
python -m scripts.run_sweep \
    --models lstm panel_gru stgnn \
    --k_values 3 5 10 \
    --num_seeds 5
```

---

## **7. Outputs (Research-Grade)**

Each run produces the following artefacts:

### **7.1 Results Table**

**Path:** `results/results.csv`

|         Field | Description            |
|---------------|-------------           |
|        model  | Model name             |
|       seed    | Random seed            |
|       seq_len | Sequence length        |
|  graph config | Graph parameters       |
| dense metrics | Classification metrics |
| trade metrics | Strategy metrics       |
|       runtime | Execution time         |
|   graph stats | Structural metrics     |

---

### **7.2 Raw Predictions**

**Path:** `results/predictions/{run_id}.csv`

|         Field | Description      |
|---===---------|------------------|
|        y_true | Ground truth     |
|        y_pred | Predictions      |
| probabilities | Model confidence |
|    timestamps | Time index       |

---

### **7.3 Graph Diagnostics**

**Path:** `results/graph_logging/graph_stats.json`

|    Metric | Description        |
|-----------|--------------------|
|     nodes | Number of nodes    |
|     edges | Number of edges    |
|   density | Graph density      |
| homophily | Feature similarity |

---

### **7.4 Backtesting Metrics**

- Sharpe ratio  
- Hit rate  
- Trade returns  
- Equity curve  

---

## **8. Evaluation Pipeline**

Evaluation is structured as a multi-layer system.

### **8.1 Dense Metrics**

- Accuracy  
- F1 (positive class)  
- F1 (macro)  
- ROC-AUC  
- Average Precision  

---

### **8.2 Trade-Aligned Metrics**

Computed only on executed trades:

- Accuracy  
- Macro F1  
- Sharpe ratio  
- Hit rate  

---

### **8.3 Strategy Metrics**

- Equity curve  
- Drawdown  
- Return distribution  

---

### **8.4 Evaluation Interface**

```python
EvaluationMethods.evaluate(...)
```

---

## **9. Backtesting Engine**

Key assumptions:

|              Rule | Description    |
|-------------------|----------------|
| `prediction == 1` | Long position  |
| `prediction == 0` | Short position |

### **Features**

- Non-overlapping trades  
- Horizon-aligned execution  
- Transaction costs  
- Annualised Sharpe calculation  
- Trade-aligned evaluation  

---

## **10. Experimental Workflow**

Recommended sequence:

1. Smoke test  
2. LSTM / GRU baseline  
3. Panel baseline  
4. STGNN  
5. Graph ablation  
6. Feature ablation  
7. Embedding ablation  
8. Multi-seed sweep  
9. Aggregate results  

---