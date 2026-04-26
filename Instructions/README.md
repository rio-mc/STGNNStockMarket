# STGNN Stock Market Prediction

Spatio-temporal graph neural network framework for stock trend prediction, with recurrent and graph-based baselines.

---

## Overview

This project compares:

- **LSTM / GRU** → single-asset temporal models  
- **PANEL_GRU** → multi-asset temporal (no graph)  
- **STGNN** → spatio-temporal graph model  

The goal is to isolate the gain from **graph structure vs temporal modelling alone**.

---

## Setup Options

### Recommended: Local virtual environment (development)

### Alternative: Docker (reproducibility)

---

# Local Setup (Recommended)

## 1. Create virtual environment

```bash
python -m venv .venv

.venv\Scripts\Activate.ps1
```

## 2. Install dependecies

```bash
pip install --upgrade pip
pip install -r instructions/requirements.txt

# GPU / PyTorch CUDA 12.1 example
pip install -r instructions/requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## 3. Run application

```bash
python -m core.main
```