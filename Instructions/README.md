# STGNN GUI (Docker + CUDA)

This repo can be run inside a GPU-enabled Docker container **with the Tkinter GUI**.  
You can also run everything locally (recommended if you’re developing and already have Python + deps installed).

## Which should I use?
- **Docker (recommended for reproducibility / easiest setup):** best if you want “it runs” with fewer dependency issues.
- **Local install:** best if you’re actively developing and don’t want GUI/X11 in a container.

---

## Prerequisites

### Linux (recommended)
- NVIDIA GPU drivers installed
- Docker installed
- NVIDIA Container Toolkit installed (enables `--gpus all`)

### Windows / macOS
- Windows: use **WSL2** + Docker Desktop + an X server (e.g., VcXsrv)
- macOS: you generally need an X server (XQuartz). GPU passthrough is not as straightforward as Linux.

If you only care about running on Linux with a local desktop, the steps below are the smoothest.

---

## Docker: Build the image

From the repo root (where the Dockerfile is):

```bash
docker build -t stgnn-gui:cuda .
