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
requirements-local-cu121.txt  Local Windows GPU Torch stack
```

Install `requirements.txt` first, then install exactly one Torch stack.

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

Install the CUDA-enabled PyTorch stack:

```powershell
python -m pip install -r requirements-local-cu121.txt `
  --index-url https://download.pytorch.org/whl/cu121
```

Install PyTorch Geometric GPU packages:

```powershell
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric `
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected CUDA result:

```text
True
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
