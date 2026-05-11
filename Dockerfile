# GPU runtime with PyTorch preinstalled
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# --- System deps ---
# Tkinter GUI + X11 libs
# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    python3-tk \
    tk \
    tcl \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# --- Python deps: shared non-Torch dependencies ---
# Torch is already provided by the PyTorch base image.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt

# --- PyTorch Geometric GPU wheels matching installed Torch + CUDA ---
# Docker uses requirements-docker.txt, not requirements-gpu.txt.
# requirements-docker.txt should contain PyG packages only.
COPY requirements-docker.txt /app/requirements-docker.txt

RUN python - <<'PY'
import sys
import subprocess
import torch

torch_ver = torch.__version__.split("+")[0]
cuda = torch.version.cuda

if cuda is None:
    raise RuntimeError("This PyTorch build does not report CUDA support.")

cuda_suffix = "cu" + cuda.replace(".", "")
wheel_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_suffix}.html"

print("Torch:", torch.__version__)
print("CUDA:", cuda)
print("PyG wheel index:", wheel_url)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-f",
    wheel_url,
    "-r",
    "/app/requirements-docker.txt",
])
PY

# --- Copy project ---
COPY . /app

# --- Default: safer for Docker experiments ---
CMD ["python", "-m", "core.main", "--run_mode", "headless"]