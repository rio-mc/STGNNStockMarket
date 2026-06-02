# Installation Guide

This project is intended to be installed directly from a terminal into a Python
virtual environment.

```text
Option A: Local CPU         No NVIDIA GPU required
Option B: Local Windows GPU CUDA-enabled PyTorch
```

Do not install both CPU and GPU Torch stacks into the same environment.

---

## 1. Requirements

```text
Python 3.12
Git
PowerShell
```

For GPU runs, you also need:

```text
NVIDIA GPU
Recent NVIDIA driver
```

---

## 2. Clone From GitHub

```powershell
git clone https://github.com/rio-mc/STGNNStockMarket.git
cd STGNNStockMarket
```

If you already cloned the repository:

```powershell
git pull
```

---

## 3. Create And Activate A Virtual Environment

Run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade the installer tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Dependency Files

The project separates shared dependencies from the PyTorch stack.

```text
requirements.txt              Shared app dependencies, no Torch
requirements-local-cpu.txt    Local CPU Torch stack
requirements-local-cu121.txt  CUDA 12.1 Torch stack
requirements-local-cu130.txt  CUDA 13.0 Torch stack, required for RTX 50-series / sm_120
```

Install `requirements.txt` first, then install exactly one Torch stack. The
CUDA stack is user-selected; do not assume `cu130` unless your GPU and driver
need it.

```powershell
python -m pip install -r requirements.txt
```

---

## 5. Option A: Local CPU

Use this path if no NVIDIA GPU is available.

```powershell
python -m pip install -r requirements-local-cpu.txt `
  --index-url https://download.pytorch.org/whl/cpu
```

Install PyTorch Geometric:

```powershell
python -m pip install torch-geometric
```

Verify the install:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected CUDA result:

```text
False
```

---

## 6. Option B: Local Windows GPU

Use this path if you have an NVIDIA GPU and want CUDA acceleration.

First inspect the driver/runtime advertised by NVIDIA:

```powershell
nvidia-smi
```

Then choose a PyTorch CUDA wheel family supported by your driver and GPU. For
example, older RTX GPUs may use `cu121`; RTX 50-series GPUs need a stack that
contains `sm_120`, such as current `cu130` wheels.

```powershell
# Example: RTX 50-series / CUDA 13.0 stack
$CUDA_WHEEL = "cu130"
$TORCH_REQ = "requirements-local-cu130.txt"
$PYG_TAG = "torch-2.11.0+cu130"

# Example: older CUDA 12.1 stack
# $CUDA_WHEEL = "cu121"
# $TORCH_REQ = "requirements-local-cu121.txt"
# $PYG_TAG = "torch-2.2.2+cu121"

python -m pip install -r $TORCH_REQ `
  --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL"
```

Install PyTorch Geometric. The exact optional binary packages depend on what
the PyG project publishes for your Python, OS, Torch version, and CUDA wheel
family. Install `torch_geometric` first, then add matching extension wheels
only when they exist for your selected `$PYG_TAG`.

```powershell
python -m pip install torch_geometric pyg_lib `
  -f "https://data.pyg.org/whl/$PYG_TAG.html"

# Optional, only if matching wheels exist for your selected $PYG_TAG:
# python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv `
#   -f "https://data.pyg.org/whl/$PYG_TAG.html"
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected CUDA result:

```text
True
```

For RTX 50-series, also verify that `sm_120` appears:

```powershell
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Verify PyTorch Geometric:

```powershell
python -c "import torch_geometric; print('PyG OK')"
```

---

## 7. Recommended First Run

Use a small configuration first:

```text
tickers = AAPL MSFT NVDA
seq_len = 8
batch_size = 16
epochs = 2
prediction_window = 1d
interval = 1h
```

Run LSTM before graph models:

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
  --experiment_name "local_smoke_lstm" `
  --seed 42
```

Then test STGNN:

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
  --stgnn_epochs 2 `
  --k 3 `
  --graph_mode knn_mst `
  --graph_embed pca `
  --save_results `
  --results_dir ./results_smoke `
  --experiment_name "local_smoke_stgnn_gcn" `
  --seed 42
```

---

## 8. Common Python Issues

### PowerShell blocks Activate.ps1

If activation is blocked by execution policy, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the venv again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Wrong Python version

Check the interpreter before installing dependencies:

```powershell
python --version
where python
```

Use Python 3.12 for this project.

### Multiple Torch stacks installed

If you accidentally installed both CPU and GPU Torch packages, recreate the
virtual environment:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Then install either the CPU stack or the GPU stack, not both.

---

## 9. Rule Of Thumb

```text
One repository clone.
One virtual environment.
One Torch stack.
```
