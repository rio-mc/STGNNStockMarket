# Installation Guide

This project supports three install paths:

```text
Option A: Docker GPU        Recommended for reproducible GPU experiments
Option B: Local Windows GPU Manual install with CUDA-enabled PyTorch
Option C: Local CPU         No NVIDIA GPU required
```

---

## 1. Requirements

### General

```text
Python 3.12
Git
PowerShell
```

### For Docker GPU

```text
Docker Desktop
WSL 2
NVIDIA GPU driver
NVIDIA GPU with Docker access
```

### For Local GPU

```text
NVIDIA GPU
Recent NVIDIA driver
Python 3.12 virtual environment
```

---

## 2. Dependency Files

The project separates dependencies by environment.

```text
requirements.txt              Shared app dependencies, no Torch
requirements-docker.txt       Docker-only PyG packages, no Torch
requirements-local-cu121.txt  Local Windows GPU Torch stack
requirements-local-cpu.txt    Local CPU Torch stack
```

Do not install all requirement files at once.

---

## 3. Option A: Docker GPU

This is the recommended path for reproducible experiments.

Docker uses a PyTorch base image:

```dockerfile
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
```

That image already includes:

```text
Python
PyTorch
CUDA runtime
cuDNN
```

Because of this, Docker should not install `torch` from a requirements file.

### 3.1 Verify Docker

```powershell
docker --version
docker info
```

### 3.2 Verify GPU access from Docker

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

This should show your NVIDIA GPU.

### 3.3 Build the image

Run from the project root:

```powershell
docker build -t stgnn-gpu .
```

### 3.4 Verify PyTorch CUDA

```powershell
docker run --rm --gpus all stgnn-gpu python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```text
torch.cuda.is_available() -> True
```

### 3.5 Verify PyTorch Geometric

```powershell
docker run --rm --gpus all stgnn-gpu python -c "import torch_geometric; import torch_scatter; import torch_sparse; print('PyG OK')"
```

### 3.6 Run a smoke test

```powershell
docker run --rm --gpus all stgnn-gpu python -m core.main `
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
  --experiment_name "docker_smoke_lstm" `
  --seed 42
```

---

## 4. Option B: Local Windows GPU

Use this if you want to run directly on Windows without Docker.

### 4.1 Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4.2 Upgrade installer tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4.3 Install shared dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4.4 Install PyTorch GPU stack

```powershell
python -m pip install -r requirements-local-cu121.txt `
  --index-url https://download.pytorch.org/whl/cu121
```

### 4.5 Install PyTorch Geometric GPU packages

```powershell
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric `
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

### 4.6 Verify local GPU install

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 4.7 Verify PyG

```powershell
python -c "import torch_geometric; import torch_scatter; import torch_sparse; print('PyG OK')"
```

---

## 5. Option C: Local CPU

Use this if no NVIDIA GPU is available.

### 5.1 Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5.2 Install shared dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 5.3 Install CPU PyTorch

```powershell
python -m pip install -r requirements-local-cpu.txt `
  --index-url https://download.pytorch.org/whl/cpu
```

### 5.4 Verify CPU install

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

```text
False
```

---

## 6. Common Docker Issues

### Docker command not found

Docker Desktop is not installed or not on PATH.

```powershell
winget install -e --id Docker.DockerDesktop
```

Restart Windows after installation.

### WSL must be updated

Run:

```powershell
wsl --update
wsl --shutdown
```

Restart Docker Desktop.

### Docker daemon not running

Start Docker Desktop manually, then check:

```powershell
docker info
```

### GPU not visible in Docker

Test:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, update the NVIDIA driver and restart Docker Desktop.

---

## 7. Common Python Issues

### Do not mix Docker and local Torch files

Docker gets Torch from the Docker image.

Local installs get Torch from local requirement files.

Do not run this inside the Docker image:

```powershell
python -m pip install -r requirements-local-cu121.txt
```

### Do not install every requirements file

Correct:

```text
Docker:
  requirements.txt
  requirements-docker.txt

Local GPU:
  requirements.txt
  requirements-local-cu121.txt
  PyG wheel command

Local CPU:
  requirements.txt
  requirements-local-cpu.txt
```

Incorrect:

```text
pip install -r requirements.txt
pip install -r requirements-docker.txt
pip install -r requirements-local-cu121.txt
pip install -r requirements-local-cpu.txt
```

---

## 8. Recommended First Run

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

## 9. Rule of Thumb

```text
Docker owns Docker Torch.
Local owns local Torch.
Never install two Torch stacks into the same environment.
```