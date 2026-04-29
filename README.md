# STGNN Stock Market Prediction

A controlled experimental framework for evaluating **temporal vs graph-based representations** in financial time series.

---

## Core Idea

This project is not about “implementing models”.

It is about:

> isolating the contribution of graph structure under controlled experimental conditions.

Every component is designed so that:

```text
same data + same features + same target
                    |
        representation changes only
                    |
 temporal baseline vs graph-aware model
Model Library

The framework includes a full spectrum of representations:

Temporal Baselines (Single Asset)
lstm
gru
Panel Temporal Models (Multi-Asset, No Graph)
panel_gru
panel_lstm
Graph Neural Networks
stgnn (primary model)
gcn
graphsage
nnconv

All models are accessed via:

runner = ModelRegistry.get_runner(model_name)

This ensures consistent execution across architectures.

What Makes This Different

Most repos compare models loosely.

This system enforces:

identical feature sets
identical tensor construction
identical training loop structure
identical evaluation pipeline
identical data splits

Only one thing changes:

representation

System Overview
Dataset → Pipeline → ExperimentRunner → Model → Evaluation → Backtest → Results
The Config Space (This is your real “product”)

Think of this repo as a research configuration engine.

Temporal Dimension
--seq_len 10
--prediction_window 1
Graph Construction
--k 3
--graph_mode knn | mst | knn_mst
--graph_embed pca | raw

Graph is built from:

return
volatility
momentum

using rolling aggregation.

Ablation Framework

This is where the real research happens.

Graph Ablation
--graph_ablation none | identity | empty
Mode	Effect
none	Full graph
identity	No cross-node edges
empty	Graph removed
Feature Ablation
--ablate_feature return | volatility | momentum
Embedding Ablation
--graph_embed pca | raw
Seeds & Reproducibility
--seed 42
--num_seeds 5
--deterministic
Running Experiments
Single Run
python -m scripts.run_experiment \
    --model stgnn \
    --dataset_name sp500 \
    --seq_len 10 \
    --k 3 \
    --seed 42
Baseline Comparison
python -m scripts.run_experiment --model lstm
python -m scripts.run_experiment --model panel_gru
python -m scripts.run_experiment --model stgnn --k 3
Full Sweep
python -m scripts.run_sweep \
    --models lstm panel_gru stgnn \
    --k_values 3 5 10 \
    --num_seeds 5
Outputs (Research-Grade)

Each run produces:

1. Results Table
results/results.csv

Includes:

model
seed
seq_len
graph config
dense metrics
trade metrics
runtime
graph stats
2. Raw Predictions
results/predictions/{run_id}.csv

Includes:

y_true
y_pred
probabilities
timestamps
3. Graph Diagnostics
results/graph_logging/graph_stats.json

Includes:

nodes
edges
density
homophily
4. Backtesting Metrics

From your engine:

Sharpe ratio
hit rate
trade returns
equity curve
Evaluation Pipeline

Evaluation is multi-layered:

Dense Metrics
accuracy
F1 (positive)
F1 (macro)
ROC-AUC
Average Precision
Trade-Aligned Metrics

Computed only on executed trades:

accuracy
macro F1
Sharpe
hit rate
Strategy Metrics
equity curve
drawdown
return distribution

All computed via:

EvaluationMethods.evaluate(...)
Backtesting Engine

Non-overlapping trade simulation:

prediction == 1 → long
prediction == 0 → short

Key features:

horizon-aligned execution
transaction costs
annualised Sharpe
trade-aligned evaluation
Why This Matters

Most financial ML papers fail because:

inconsistent inputs
hidden leakage
incomparable baselines
unclear evaluation

This system explicitly addresses those issues.

Recommended Workflow
1. Smoke test
2. LSTM / GRU baseline
3. Panel baseline
4. STGNN
5. Graph ablation
6. Feature ablation
7. Embedding ablation
8. Multi-seed sweep
9. Aggregate results
Experimental Checklist

Before claiming results:

 same input features across all models
 same train/val split
 graph uses training data only
 seeds logged
 metrics logged
 predictions saved
 graph stats saved
 results reproducible via CLI
Key Insight

If your experiments are valid, you should be able to answer:

Does graph structure improve predictive performance beyond temporal modelling alone?

Final Note

Do not describe this project as:

“We implemented multiple models”

Instead:

“We isolate and evaluate the contribution of graph structure under controlled experimental conditions.”


---

## Why this version is better

You were close, but:

- you underplayed your **model library breadth**
- you didn’t emphasise **configurability as a research tool**
- your README was **linear**, not scannable
- your evaluation pipeline wasn’t framed as a **stack**

This version fixes all of that.

---

## If you want next level

I can upgrade this further with:

- badges + quickstart UX
- diagram (actual SVG architecture)
- results table example (paper-style)
- “expected outputs” screenshots

That’s what takes this from “good repo” → “top-tier submission repo”.