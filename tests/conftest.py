from pathlib import Path
import shutil

import pytest


@pytest.fixture
def workspace_tmp(request):
    """Workspace-local temp path for Windows environments with locked %TEMP%."""

    root = Path(__file__).resolve().parent / ".runtime"
    path = root / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
