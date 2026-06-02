# STGNN STOCK MARKET PREDICTION

> **Controlled experimental framework.**
> Temporal vs panel vs graph-aware models.
> Same data. Same features. Same target. Only representation changes.

---

## QUICK START

Install from a terminal with Git and a Python virtual environment.

```powershell
git clone https://github.com/rio-mc/STGNNStockMarket.git
cd STGNNStockMarket
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Install one PyTorch stack.
See [INSTALL.md](INSTALL.md) for the full setup and verification guide.

---

## 01 / OBJECTIVE

This project is **not** about collecting models.

It is about isolating one thing:

> **Does graph structure improve predictive and trading performance under controlled experimental conditions?**

```text
SAME DATA
SAME FEATURES
SAME TARGET
    |
REPRESENTATION CHANGES ONLY
    |
TEMPORAL BASELINE
vs
PANEL TEMPORAL BASELINE
vs
GRAPH-AWARE MODEL
```

---

## 02 / MODEL LIBRARY

All models run through one registry.

```python
runner = ModelRegistry.get_runner(model_name)
```

### 02.1 / MODEL CATEGORIES

| Category | Models | Purpose |
|---|---|---|
| Temporal / Single Asset | `lstm`, `gru` | Target asset only |
| Panel Temporal / Multi-Asset | `panel_lstm`, `panel_gru` | Multi-asset context, no graph |
| Graph Neural Networks | `gcn`, `gat`, `graphsage`, `nnconv` | Static graph-aware baselines |
| Spatio-Temporal GNN | `stgnn` | Temporal + relational modelling |

Registered model names:

```text
lstm
gru
panel_lstm
panel_gru
gcn
gat
graphsage
nnconv
stgnn
```

---

## 03 / SYSTEM ARCHITECTURE

```text
UNIVERSE
  ↓
PRICE LOADER
  ↓
PIPELINE
  ↓
FEATURE ENGINEERING
  ↓
GRAPH CONSTRUCTION
  ↓
EXPERIMENT RUNNER
  ↓
MODEL RUNNER
  ↓
EVALUATION
  ↓
BACKTEST
  ↓
RESULTS STORE
```

Headless execution:

```powershell
python -m core.main --run_mode headless
```

Headless mode loads data, runs the pipeline, executes the selected model, stores results, and prints a terminal report.

---

## 04 / CONFIGURATION SPACE

The configuration space is the research interface.

### 04.1 / CORE RUNTIME PARAMETERS

| Parameter | Description |
|---|---|
| `--run_mode` | Execution mode, usually `headless` |
| `--model` | Model to run |
| `--target_stock` | Target ticker |
| `--dataset_name` | Dataset or universe identifier |
| `--custom_tickers` | Explicit ticker list |
| `--interval` | Price interval, for example `1h` |
| `--prediction_window` | Forecast horizon, for example `1d` |
| `--results_dir` | Output directory |
| `--experiment_name` | Human-readable experiment label |
| `--seed` | Random seed |

### 04.2 / TEMPORAL PARAMETERS

| Parameter | Description |
|---|---|
| `--seq_len` | Input sequence length |
| `--prediction_window` | Forecast horizon |
| `--batch_size` | Training batch size |
| `--lstm_epochs` | Epochs for recurrent and panel recurrent models |
| `--stgnn_epochs` | Epochs for graph and spatio-temporal models |

### 04.3 / GRAPH CONSTRUCTION

| Parameter | Options | Description |
|---|---|---|
| `--k` | Integer | Number of neighbours |
| `--graph_mode` | `knn`, `mst`, `knn_mst` | Graph topology |
| `--graph_embed` | `pca`, `raw` | Feature embedding for graph construction |

Graph construction uses rolling statistics of:

```text
returns
volatility
momentum
```

### 04.4 / STGNN GRAPH BACKEND

```powershell
--graph_model gcn
--graph_model gat
--graph_model graphsage
--graph_model nnconv
```

Same spatio-temporal framework. Different relational operators.

---

## 05 / ABLATION FRAMEWORK

### 05.1 / GRAPH ABLATION

| Mode | Effect |
|---|---|
| `none` | Full graph |
| `identity` | Removes cross-node relational structure |
| `empty` | Removes graph connectivity |

```powershell
--graph_ablation identity
```

### 05.2 / FEATURE ABLATION

Remove one engineered node feature:

```powershell
--ablate_feature return
--ablate_feature volatility
--ablate_feature momentum
```

### 05.3 / EMBEDDING ABLATION

```powershell
--graph_embed pca
--graph_embed raw
```

### 05.4 / REPRODUCIBILITY

| Parameter | Description |
|---|---|
| `--seed` | Random seed |
| `--deterministic` | Enables deterministic execution where supported |

Seeds are set centrally. Model runners use a deterministic DataLoader generator where supported.

---

## 06 / OUTPUTS

Each successful run writes structured outputs to the selected results directory.

```text
results_smoke/
  runs/
    experiments.jsonl
    experiments.csv
    histories/
    recurrent/
    graph/
    stgnn/
  graph_logging/
```

### 06.1 / EXPERIMENT INDEX

| File | Description |
|---|---|
| `runs/experiments.jsonl` | Full experiment records |
| `runs/experiments.csv` | Tabular experiment summary |

Each run record includes:

```text
model
ticker
seed
prediction window
graph configuration
metrics
metadata
result paths
```

### 06.2 / CANONICAL RUN PAYLOAD

| File | Description |
|---|---|
| `result.json` | Main result payload |
| `config.json` | Configuration snapshot |
| `graph_stats.json` | Graph diagnostics, where applicable |
| `history.json` | Training and validation history, where applicable |

### 06.3 / GRAPH DIAGNOSTICS

| Metric | Description |
|---|---|
| `nodes` | Number of graph nodes |
| `edges` | Number of graph edges |
| `density` | Graph density |
| `homophily` | Feature similarity measure |
| `graph_mode` | Graph construction mode |
| `graph_embed` | Embedding method |

---

## 07 / EVALUATION PIPELINE

Main interface:

```python
EvaluationMethods.evaluate(...)
```

Evaluation computes:

```text
dense classification metrics
threshold diagnostics
trade-aligned metrics
backtesting metrics
visual diagnostics for GUI runs
```

### 07.1 / DENSE METRICS

```text
Accuracy
F1, positive class
F1, macro
ROC-AUC
Average precision
Dense validation loss
```

### 07.2 / THRESHOLD DIAGNOSTICS

The system reports:

```text
fixed threshold performance, usually 0.500
best dense macro-F1 threshold
```

### 07.3 / TRADE-ALIGNED METRICS

Computed only on executed trades:

```text
Accuracy
F1, positive class
F1, macro
ROC-AUC
Average precision
Number of trades
Hit rate
Mean trade return
```

### 07.4 / STRATEGY METRICS

```text
Sharpe ratio
Final equity
Maximum drawdown
Trade returns
Equity curve
```

---

## 08 / BACKTESTING ASSUMPTIONS

| Rule | Meaning |
|---|---|
| `prediction == 1` | Long position |
| `prediction == 0` | Short position |

Backtesting is designed around:

```text
non-overlapping trades
horizon-aligned execution
transaction cost support
annualised Sharpe calculation
trade-aligned evaluation
```

---

## 09 / EXPERIMENTAL WORKFLOW

```text
01 smoke test
02 LSTM / GRU baseline
03 panel baseline
04 STGNN
05 graph ablation
06 feature ablation
07 embedding ablation
08 multi-seed sweep
09 aggregate results
```

Commands are formatted for PowerShell.

---

## 09.1 / SMOKE TEST

Purpose: verify the full headless pipeline end to end.

```powershell
python -m core.main `
  --run_mode headless `
  --model lstm `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --lstm_epochs 2 `
  --save_results `
  --results_dir ./results_smoke `
  --experiment_name "smoke_lstm" `
  --seed 42
```

---

## 09.2 / LSTM / GRU BASELINE

Purpose: establish single-asset temporal baselines.

### LSTM

```powershell
python -m core.main `
  --run_mode headless `
  --model lstm `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --lstm_epochs 5 `
  --save_results `
  --results_dir ./results_baselines `
  --experiment_name "baseline_lstm_seed42" `
  --seed 42
```

### GRU

```powershell
python -m core.main `
  --run_mode headless `
  --model gru `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --lstm_epochs 5 `
  --save_results `
  --results_dir ./results_baselines `
  --experiment_name "baseline_gru_seed42" `
  --seed 42
```

> `--lstm_epochs` is used by both LSTM and GRU recurrent runners.

---

## 09.3 / PANEL BASELINE

Purpose: test multi-asset temporal context without explicit graph structure.

### Panel LSTM

```powershell
python -m core.main `
  --run_mode headless `
  --model panel_lstm `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --lstm_epochs 5 `
  --save_results `
  --results_dir ./results_panel `
  --experiment_name "panel_lstm_seed42" `
  --seed 42
```

### Panel GRU

```powershell
python -m core.main `
  --run_mode headless `
  --model panel_gru `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --lstm_epochs 5 `
  --save_results `
  --results_dir ./results_panel `
  --experiment_name "panel_gru_seed42" `
  --seed 42
```

---

## 09.4 / STGNN

Purpose: test joint temporal and graph-aware modelling.

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --save_results `
  --results_dir ./results_stgnn `
  --experiment_name "stgnn_gcn_seed42" `
  --seed 42
```

Alternative backends:

```powershell
--graph_model gat
--graph_model graphsage
--graph_model nnconv
```

---

## 09.5 / GRAPH ABLATION

Purpose: test whether graph connectivity contributes to performance.

### Full Graph

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --graph_ablation none `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --save_results `
  --results_dir ./results_graph_ablation `
  --experiment_name "graph_ablation_none_seed42" `
  --seed 42
```

### Identity Graph

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --graph_ablation identity `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --save_results `
  --results_dir ./results_graph_ablation `
  --experiment_name "graph_ablation_identity_seed42" `
  --seed 42
```

### Empty Graph

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --graph_ablation empty `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --save_results `
  --results_dir ./results_graph_ablation `
  --experiment_name "graph_ablation_empty_seed42" `
  --seed 42
```

---

## 09.6 / FEATURE ABLATION

Purpose: test which engineered node features contribute to prediction.

### Remove Returns

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --ablate_feature return `
  --save_results `
  --results_dir ./results_feature_ablation `
  --experiment_name "feature_ablation_return_seed42" `
  --seed 42
```

### Remove Volatility

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --ablate_feature volatility `
  --save_results `
  --results_dir ./results_feature_ablation `
  --experiment_name "feature_ablation_volatility_seed42" `
  --seed 42
```

### Remove Momentum

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --ablate_feature momentum `
  --save_results `
  --results_dir ./results_feature_ablation `
  --experiment_name "feature_ablation_momentum_seed42" `
  --seed 42
```

---

## 09.7 / EMBEDDING ABLATION

Purpose: compare PCA-based graph construction against raw feature embeddings.

### PCA Embedding

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --save_results `
  --results_dir ./results_embedding_ablation `
  --experiment_name "embedding_pca_seed42" `
  --seed 42
```

### Raw Embedding

```powershell
python -m core.main `
  --run_mode headless `
  --model stgnn `
  --graph_model gcn `
  --target_stock AAPL `
  --dataset_name custom `
  --custom_tickers AAPL MSFT NVDA `
  --prediction_window 1d `
  --interval 1h `
  --seq_len 8 `
  --batch_size 16 `
  --stgnn_epochs 5 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed raw `
  --save_results `
  --results_dir ./results_embedding_ablation `
  --experiment_name "embedding_raw_seed42" `
  --seed 42
```

---

## 09.8 / MULTI-SEED SWEEP

Purpose: estimate whether observed differences survive random initialisation.

```powershell
$seeds = @(42, 43, 44, 45, 46)

foreach ($s in $seeds) {
  python -m core.main `
    --run_mode headless `
    --model stgnn `
    --graph_model gcn `
    --target_stock AAPL `
    --dataset_name custom `
    --custom_tickers AAPL MSFT NVDA `
    --prediction_window 1d `
    --interval 1h `
    --seq_len 8 `
    --batch_size 16 `
    --stgnn_epochs 5 `
    --k 3 `
    --graph_mode knn_mst `
    --graph_embed pca `
    --save_results `
    --results_dir ./results_multiseed `
    --experiment_name "stgnn_gcn_seed_$s" `
    --seed $s
}
```

---

## 09.9 / AGGREGATE RESULTS

Main aggregate files:

```text
./results_multiseed/runs/experiments.csv
./results_multiseed/runs/experiments.jsonl
```

Minimal check:

```powershell
Get-Content ./results_multiseed/runs/experiments.csv
```

Load with Python:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./results_multiseed/runs/experiments.csv'); print(df.head()); print(df.columns.tolist())"
```

Rank by dense macro-F1:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./results_multiseed/runs/experiments.csv'); print(df.sort_values('metrics.macro_f1_dense', ascending=False)[['model','ticker','seed','graph_model','metrics.macro_f1_dense','metrics.sharpe','metrics.final_equity']].head(20))"
```

Inspect columns:

```powershell
python -c "import pandas as pd; df=pd.read_csv('./results_multiseed/runs/experiments.csv'); print('\n'.join(df.columns))"
```

---

## 10 / VERIFICATION ORDER

Use short runs first:

```text
epochs = 2
tickers = AAPL MSFT NVDA
seq_len = 8
interval = 1h
prediction_window = 1d
```

Recommended order:

```text
01 lstm
02 gru
03 panel_lstm
04 panel_gru
05 gcn
06 gat
07 graphsage
08 nnconv
09 stgnn --graph_model gcn
10 stgnn --graph_model gat
11 stgnn --graph_model graphsage
12 stgnn --graph_model nnconv
```

---

## 11 / SMOKE TEST MATRIX

Run after changes to runners, evaluation, persistence, or graph construction.

```powershell
$baseModels = @(
  "lstm",
  "gru",
  "panel_lstm",
  "panel_gru",
  "gcn",
  "gat",
  "graphsage",
  "nnconv"
)

foreach ($m in $baseModels) {
  Write-Host "RUNNING $m"

  python -m core.main `
    --run_mode headless `
    --model $m `
    --target_stock AAPL `
    --dataset_name custom `
    --custom_tickers AAPL MSFT NVDA `
    --prediction_window 1d `
    --interval 1h `
    --seq_len 8 `
    --batch_size 16 `
    --stgnn_epochs 2 `
    --lstm_epochs 2 `
    --save_results `
    --results_dir ./results_smoke `
    --experiment_name "smoke_$m" `
    --seed 42
}

$graphModels = @(
  "gcn",
  "gat",
  "graphsage",
  "nnconv"
)

foreach ($gm in $graphModels) {
  Write-Host "RUNNING STGNN + $gm"

  python -m core.main `
    --run_mode headless `
    --model stgnn `
    --graph_model $gm `
    --target_stock AAPL `
    --dataset_name custom `
    --custom_tickers AAPL MSFT NVDA `
    --prediction_window 1d `
    --interval 1h `
    --seq_len 8 `
    --batch_size 16 `
    --stgnn_epochs 2 `
    --save_results `
    --results_dir ./results_smoke `
    --experiment_name "smoke_stgnn_$gm" `
    --seed 42
}
```

---

## 12 / HEADLESS RESULT SUITES

Use these PowerShell templates to generate structured result suites without the
GUI. Start with the pilot settings, inspect the CSV outputs, then scale up.

Outputs are written under:

```text
<results_dir>/runs/experiments.csv
<results_dir>/runs/experiments.jsonl
<results_dir>/runs/<model>/...
<results_dir>/graph_logging/...
```

For queue-style/batch summaries, also inspect:

```text
<results_dir>/runs/queue_runs/<queue_run_id>/seed_results.csv
<results_dir>/runs/queue_runs/<queue_run_id>/summary_by_model.csv
```

### 12.1 / Shared Setup

```powershell
$DATASET = "sp500"
$TOP_N = 50
$SEEDS = 42..46
$TICKERS = Import-Csv .\static\universes\sp500_tickers.csv |
  Select-Object -First $TOP_N |
  ForEach-Object { $_.ticker }

$COMMON = @(
  "--run_mode", "headless",
  "--device", "cuda",
  "--dataset_name", $DATASET,
  "--top_n", "$TOP_N",
  "--prediction_window", "1d",
  "--interval", "1h",
  "--seq_len", "8",
  "--batch_size", "16",
  "--graph_mode", "knn_mst",
  "--graph_embed", "pca",
  "--save_results"
)
```

Pilot settings:

```powershell
$TOP_N = 5
$SEEDS = 42..43

$COMMON += @(
  "--lstm_epochs", "2",
  "--stgnn_epochs", "5"
)
```

For final runs, use the intended epoch counts and seed range.

### 12.2 / Recurrent Suite

Purpose: temporal baselines without graph message passing.

```powershell
$MODELS = @("lstm", "gru", "panel_lstm", "panel_gru")

foreach ($model in $MODELS) {
  foreach ($ticker in $TICKERS) {
    foreach ($seed in $SEEDS) {
      python -m core.main @COMMON `
        --model $model `
        --target_stock $ticker `
        --seed $seed `
        --results_dir ".\results_suites\recurrent"
    }
  }
}
```

### 12.3 / Static Graph k-Sensitivity Suite

Purpose: isolate the effect of graph neighbourhood density for static graph
baselines.

```powershell
$GRAPH_MODELS = @("gcn", "gat", "graphsage", "nnconv")
$KS = @(1, 3, 5, 10)

foreach ($k in $KS) {
  foreach ($model in $GRAPH_MODELS) {
    foreach ($ticker in $TICKERS) {
      foreach ($seed in $SEEDS) {
        python -m core.main @COMMON `
          --model $model `
          --target_stock $ticker `
          --seed $seed `
          --k $k `
          --results_dir ".\results_suites\graph_k$k"
      }
    }
  }
}
```

Keep fixed during this suite:

```text
dataset
top_n
target stocks
seeds
prediction_window
seq_len
batch_size
graph_mode = knn_mst
graph_embed = pca
```

### 12.4 / STGNN Backend Suite

Purpose: compare temporal + graph integration across graph backends.

```powershell
$BACKENDS = @("gcn", "gat", "graphsage", "nnconv")
$K = 3

foreach ($backend in $BACKENDS) {
  foreach ($ticker in $TICKERS) {
    foreach ($seed in $SEEDS) {
      python -m core.main @COMMON `
        --model "stgnn" `
        --graph_model $backend `
        --target_stock $ticker `
        --seed $seed `
        --k $K `
        --results_dir ".\results_suites\stgnn_$backend"
    }
  }
}
```

### 12.5 / STGNN Ablation Suite

Run this after choosing a candidate backend and `k` from the previous suites.

```powershell
$BEST_BACKEND = "gcn"
$BEST_K = 3
$GRAPH_ABLATIONS = @("none", "identity", "empty")

foreach ($ablation in $GRAPH_ABLATIONS) {
  foreach ($ticker in $TICKERS) {
    foreach ($seed in $SEEDS) {
      python -m core.main @COMMON `
        --model "stgnn" `
        --graph_model $BEST_BACKEND `
        --target_stock $ticker `
        --seed $seed `
        --k $BEST_K `
        --graph_ablation $ablation `
        --results_dir ".\results_suites\stgnn_ablation_$ablation"
    }
  }
}
```

Feature ablations should be run separately:

```powershell
$FEATURE_ABLATIONS = @("return", "volatility", "momentum")

foreach ($feature in $FEATURE_ABLATIONS) {
  foreach ($ticker in $TICKERS) {
    foreach ($seed in $SEEDS) {
      python -m core.main @COMMON `
        --model "stgnn" `
        --graph_model $BEST_BACKEND `
        --target_stock $ticker `
        --seed $seed `
        --k $BEST_K `
        --ablate_feature $feature `
        --results_dir ".\results_suites\stgnn_feature_ablation_$feature"
    }
  }
}
```

### 12.6 / Result Checks

Before using a suite in a manuscript, check:

```text
all expected ticker x seed x model rows are present
failed rows are explained
experiments.csv contains k, graph_mode, graph_embed, graph_ablation, ablate_feature
graph stats exist for graph-aware models
metric_macro_f1_dense / metric_roc_auc_dense / trade metrics are populated
energy and train time are populated
```

Use the suite structure for the results narrative:

```text
1. recurrent baselines
2. static graph baselines with k-sensitivity
3. STGNN backend comparison
4. STGNN graph and feature ablations
```

---

## 13 / INTERPRETATION NOTES

Do **not** evaluate this framework on raw accuracy alone.

Compare:

```text
dense classification metrics
trade-aligned metrics
strategy metrics
graph diagnostics
compute cost
stability across seeds
```

A graph-aware model is only meaningfully better when it improves performance under comparable:

```text
data
features
target
seed
training conditions
```

One strong run is not evidence.

Prefer:

```text
multi-seed comparisons
ablations
controlled baselines
```
